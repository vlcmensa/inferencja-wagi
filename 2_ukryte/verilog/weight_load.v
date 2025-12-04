/*
Project 29: Weight Loading via UART
Objective: Load trained neural network weights from PC to FPGA BRAM.

Protocol:
  - Start marker: 0xAA 0x55 (two bytes)
  - Data bytes: weights and biases (sent in order)
  - End marker: 0x55 0xAA (two bytes)

LED Status:
  - led[0]: Blinks when receiving bytes
  - led[1]: HIGH when waiting for start marker
  - led[2]: HIGH when receiving data
  - led[3]: HIGH when transfer complete (success)
  - led[4]: HIGH if error (too much data / overflow)
  - led[7:5]: unused
  - led[15:8]: Lower 8 bits of current address (progress indicator)

Memory Layout (example for 784->16->16->10 network):
  Address 0-12543:     L1 weights (12544 bytes)
  Address 12544-12607: L1 biases (64 bytes, 16 x 4-byte)
  Address 12608-12863: L2 weights (256 bytes)
  Address 12864-12927: L2 biases (64 bytes)
  Address 12928-13087: L3 weights (160 bytes)
  Address 13088-13127: L3 biases (40 bytes)
  Total: 13128 bytes
*/

// =============================================================================
// TOP MODULE - Use this for synthesis (only external pins)
// =============================================================================
module weight_load_top (
    input wire clk,              // 100 MHz System Clock
    input wire rst,              // Reset Button
    input wire rx,               // UART RX Line (from PC)
    output wire [15:0] led       // Debug LEDs
);

    // Internal signals (for future inference module connection)
    wire [7:0] read_data;
    wire transfer_done;
    wire [13:0] data_size;
    
    // For testing: tie read_addr to 0 (not reading anything yet)
    wire [13:0] read_addr = 14'd0;

    // Instantiate the weight loader
    weight_load u_weight_load (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .read_addr(read_addr),
        .read_data(read_data),
        .transfer_done(transfer_done),
        .data_size(data_size),
        .led(led)
    );

endmodule

// =============================================================================
// WEIGHT LOAD MODULE - Has internal ports for inference connection
// =============================================================================
module weight_load (
    input wire clk,              // 100 MHz System Clock
    input wire rst,              // Reset Button
    input wire rx,               // UART RX Line (from PC)
    
    // Read port for inference
    input wire [13:0] read_addr, // Address to read from
    output reg [7:0] read_data,  // Data at read address
    
    // Status outputs
    output reg transfer_done,    // HIGH when transfer complete
    output reg [13:0] data_size, // Number of bytes received
    
    // Debug LEDs
    output reg [15:0] led
);

    // Parameters
    parameter CLK_FREQ = 100_000_000;
    parameter BAUD_RATE = 115200;
    parameter WEIGHT_SIZE = 16384;  // 16KB buffer (enough for 13128 bytes)
    
    // Protocol markers
    localparam START_BYTE1 = 8'hAA;
    localparam START_BYTE2 = 8'h55;
    localparam END_BYTE1 = 8'h55;
    localparam END_BYTE2 = 8'hAA;
    
    // State machine states
    localparam STATE_WAIT_START1 = 0;  // Waiting for first start byte (0xAA)
    localparam STATE_WAIT_START2 = 1;  // Waiting for second start byte (0x55)
    localparam STATE_RECEIVING = 2;     // Receiving data bytes
    localparam STATE_CHECK_END = 3;     // Checking if this is end marker
    localparam STATE_DONE = 4;          // Transfer complete
    localparam STATE_ERROR = 5;         // Error occurred

    // Internal Signals
    wire [7:0] rx_data;
    wire rx_ready;
    
    // Block RAM for weights and biases
    (* ram_style = "block" *) reg [7:0] weight_bram [0:WEIGHT_SIZE-1];
    
    // State and control registers
    reg [2:0] state;
    reg [13:0] write_addr;
    reg [7:0] prev_byte;        // Store previous byte for end marker detection
    reg blink_toggle;           // For LED blinking
    
    // UART Receiver Instance
    uart_rx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_rx (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .data(rx_data),
        .ready(rx_ready)
    );

    // Synchronous read from BRAM (required for proper BRAM inference)
    always @(posedge clk) begin
        read_data <= weight_bram[read_addr];
    end

    // Main state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_WAIT_START1;
            write_addr <= 0;
            prev_byte <= 0;
            transfer_done <= 0;
            data_size <= 0;
            blink_toggle <= 0;
            led <= 16'b0000_0000_0000_0010;  // led[1] = waiting for start
        end else begin
            
            case (state)
                // ============================================
                // Wait for first start byte (0xAA)
                // ============================================
                STATE_WAIT_START1: begin
                    led[1] <= 1;  // Waiting indicator
                    led[2] <= 0;
                    led[3] <= 0;
                    
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        if (rx_data == START_BYTE1) begin
                            state <= STATE_WAIT_START2;
                        end
                    end
                end
                
                // ============================================
                // Wait for second start byte (0x55)
                // ============================================
                STATE_WAIT_START2: begin
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        if (rx_data == START_BYTE2) begin
                            // Valid start sequence received
                            state <= STATE_RECEIVING;
                            write_addr <= 0;
                            led[1] <= 0;
                            led[2] <= 1;  // Receiving indicator
                        end else if (rx_data == START_BYTE1) begin
                            // Another 0xAA, stay in this state
                            state <= STATE_WAIT_START2;
                        end else begin
                            // Invalid sequence, go back to waiting
                            state <= STATE_WAIT_START1;
                        end
                    end
                end
                
                // ============================================
                // Receiving data bytes
                // ============================================
                STATE_RECEIVING: begin
                    led[2] <= 1;  // Receiving indicator
                    led[15:8] <= write_addr[7:0];  // Show progress
                    
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        // Check for potential end marker
                        if (prev_byte == END_BYTE1 && rx_data == END_BYTE2) begin
                            // End marker detected!
                            // Don't store the end marker bytes
                            // write_addr was already incremented for prev_byte (0x55)
                            // We need to "undo" that - the last real data byte count is write_addr - 1
                            state <= STATE_DONE;
                            data_size <= write_addr;  // This includes the 0x55 we stored
                            transfer_done <= 1;
                            led[2] <= 0;
                            led[3] <= 1;  // Success indicator
                        end else begin
                            // Store the previous byte (if not first byte)
                            if (write_addr > 0 || prev_byte != 0) begin
                                // Check for overflow
                                if (write_addr >= WEIGHT_SIZE) begin
                                    state <= STATE_ERROR;
                                    led[4] <= 1;  // Error indicator
                                end else begin
                                    weight_bram[write_addr] <= prev_byte;
                                    write_addr <= write_addr + 1;
                                end
                            end
                            prev_byte <= rx_data;
                        end
                    end
                end
                
                // ============================================
                // Transfer complete
                // ============================================
                STATE_DONE: begin
                    led[3] <= 1;  // Success - stays on
                    led[15:8] <= data_size[7:0];  // Show final count
                    transfer_done <= 1;
                    // Stay in this state until reset
                end
                
                // ============================================
                // Error state
                // ============================================
                STATE_ERROR: begin
                    led[4] <= 1;  // Error indicator stays on
                    // Stay in this state until reset
                end
                
                default: begin
                    state <= STATE_WAIT_START1;
                end
            endcase
        end
    end

endmodule

// -----------------------------------------------------------------------------
// UART Receiver Module (115200 baud compatible)
// -----------------------------------------------------------------------------
module uart_rx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input wire clk,
    input wire rst,
    input wire rx,
    output reg [7:0] data,
    output reg ready
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    // State definitions
    localparam STATE_IDLE = 0;
    localparam STATE_START = 1;
    localparam STATE_DATA = 2;
    localparam STATE_STOP = 3;

    reg [1:0] state;
    reg [13:0] clk_cnt;
    reg [2:0] bit_cnt;
    reg [7:0] rx_shift;         // Shift register for incoming bits
    reg rx_sync1, rx_sync2;     // Double-flop synchronizer for metastability

    // Synchronize RX input (2-stage synchronizer)
    always @(posedge clk) begin
        rx_sync1 <= rx;
        rx_sync2 <= rx_sync1;
    end

    // UART receive state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            clk_cnt <= 0;
            bit_cnt <= 0;
            data <= 0;
            rx_shift <= 0;
            ready <= 0;
        end else begin
            ready <= 0;  // Default: ready is only high for 1 cycle
            
            case (state)
                // ----------------------------------------
                // IDLE: Wait for start bit (falling edge)
                // ----------------------------------------
                STATE_IDLE: begin
                    clk_cnt <= 0;
                    bit_cnt <= 0;
                    if (rx_sync2 == 0) begin
                        state <= STATE_START;
                    end
                end
                
                // ----------------------------------------
                // START: Verify start bit at middle
                // ----------------------------------------
                STATE_START: begin
                    if (clk_cnt == (CLKS_PER_BIT / 2)) begin
                        if (rx_sync2 == 0) begin
                            // Valid start bit
                            clk_cnt <= 0;
                            state <= STATE_DATA;
                        end else begin
                            // False start (noise)
                            state <= STATE_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // DATA: Sample 8 data bits (LSB first)
                // ----------------------------------------
                STATE_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT) begin
                        clk_cnt <= 0;
                        rx_shift[bit_cnt] <= rx_sync2;
                        
                        if (bit_cnt == 7) begin
                            bit_cnt <= 0;
                            state <= STATE_STOP;
                        end else begin
                            bit_cnt <= bit_cnt + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // STOP: Wait for stop bit, output data
                // ----------------------------------------
                STATE_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT) begin
                        clk_cnt <= 0;
                        state <= STATE_IDLE;
                        data <= rx_shift;  // Transfer to output
                        ready <= 1;        // Signal data valid
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
