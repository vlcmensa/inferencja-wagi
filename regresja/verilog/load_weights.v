/*
================================================================================
Softmax Regression Weight Loader via UART
================================================================================

Loads trained neural network weights and biases from PC to FPGA BRAM.

Model: Softmax Regression (Logistic Regression) for MNIST
  - Input:  784 pixels (28x28 image, 8-bit unsigned)
  - Output: 10 classes (digits 0-9)
  - Weights: 784 x 10 = 7840 values (8-bit signed)
  - Biases:  10 values (32-bit signed, stored as 4 bytes each little-endian)

Protocol:
  - Start marker: 0xAA 0x55 (two bytes)
  - Data bytes: weights (7840 bytes) + biases (40 bytes) = 7880 bytes
  - End marker: 0x55 0xAA (two bytes)

Memory Layout:
  Address 0-7839:     Weights (7840 bytes, 8-bit signed)
                      Organized as weight[output_class][input_pixel]
                      output_class = addr / 784
                      input_pixel  = addr % 784
  Address 7840-7879:  Biases (40 bytes, 10 x 4-byte little-endian)
                      bias[class] at addresses 7840 + class*4 to 7840 + class*4 + 3

LED Status:
  - led[0]:    Blinks when receiving bytes
  - led[1]:    HIGH when waiting for start marker
  - led[2]:    HIGH when receiving data
  - led[3]:    HIGH when transfer complete (success)
  - led[4]:    HIGH if error (overflow)
  - led[7:5]:  Unused
  - led[15:8]: Lower 8 bits of current address (progress indicator)

================================================================================
*/

// =============================================================================
// TOP MODULE - Use this for synthesis (only external pins)
// =============================================================================
module load_weights_top (
    input wire clk,              // 100 MHz System Clock
    input wire rst,              // Reset Button (active high)
    input wire rx,               // UART RX Line (from PC)
    output wire [15:0] led       // Debug LEDs
);

    // Internal signals for inference module connection (directly exposed here)
    wire [12:0] weight_addr;
    wire [7:0]  weight_data;
    wire [3:0]  bias_addr;
    wire [31:0] bias_data;
    wire        transfer_done;
    
    // For testing: tie read addresses to 0
    assign weight_addr = 13'd0;
    assign bias_addr = 4'd0;

    // Instantiate the weight loader
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weight_rd_addr(weight_addr),
        .weight_rd_data(weight_data),
        .bias_rd_addr(bias_addr),
        .bias_rd_data(bias_data),
        .transfer_done(transfer_done),
        .led(led)
    );

endmodule


// =============================================================================
// WEIGHT LOADER MODULE - Has read ports for inference module
// =============================================================================
module weight_loader (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    input wire rx,                // UART RX Line
    
    // Read port for weights (inference module)
    input wire [12:0] weight_rd_addr,  // 0 to 7839 (13 bits needed)
    output reg [7:0]  weight_rd_data,  // 8-bit signed weight
    
    // Read port for biases (inference module)
    input wire [3:0]  bias_rd_addr,    // 0 to 9 (4 bits)
    output reg [31:0] bias_rd_data,    // 32-bit signed bias
    
    // Status
    output reg transfer_done,
    
    // Debug LEDs
    output reg [15:0] led
);

    // Parameters
    parameter CLK_FREQ = 100_000_000;
    parameter BAUD_RATE = 115200;
    
    // Memory sizes
    localparam WEIGHT_SIZE = 7840;   // 784 x 10 weights
    localparam BIAS_SIZE = 10;       // 10 biases
    localparam TOTAL_BYTES = 7880;   // 7840 + 40
    
    // Protocol markers
    localparam START_BYTE1 = 8'hAA;
    localparam START_BYTE2 = 8'h55;
    localparam END_BYTE1 = 8'h55;
    localparam END_BYTE2 = 8'hAA;
    
    // State machine states
    localparam STATE_WAIT_START1 = 3'd0;
    localparam STATE_WAIT_START2 = 3'd1;
    localparam STATE_RECEIVING   = 3'd2;
    localparam STATE_CHECK_END   = 3'd3;
    localparam STATE_DONE        = 3'd4;
    localparam STATE_ERROR       = 3'd5;

    // UART signals
    wire [7:0] rx_data;
    wire rx_ready;
    
    // Block RAM for weights (8-bit values)
    (* ram_style = "block" *) reg [7:0] weight_bram [0:WEIGHT_SIZE-1];
    
    // Block RAM for biases (32-bit values, stored as 4 separate bytes then combined)
    (* ram_style = "block" *) reg [31:0] bias_bram [0:BIAS_SIZE-1];
    
    // State and control registers
    reg [2:0] state;
    reg [13:0] write_addr;       // Address for writing (0 to 7879)
    reg [7:0] prev_byte;         // Previous byte for end marker detection
    reg blink_toggle;            // For LED blinking
    
    // Bias assembly registers (4 bytes -> 32 bits)
    reg [1:0] bias_byte_cnt;     // Which byte of bias we're receiving (0-3)
    reg [31:0] bias_temp;        // Temporary register for assembling bias
    
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

    // Synchronous read from weight BRAM
    always @(posedge clk) begin
        weight_rd_data <= weight_bram[weight_rd_addr];
    end
    
    // Synchronous read from bias BRAM
    always @(posedge clk) begin
        bias_rd_data <= bias_bram[bias_rd_addr];
    end

    // Main state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_WAIT_START1;
            write_addr <= 0;
            prev_byte <= 0;
            transfer_done <= 0;
            bias_byte_cnt <= 0;
            bias_temp <= 0;
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
                    led[4] <= 0;
                    
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
                            bias_byte_cnt <= 0;
                            bias_temp <= 0;
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
                    led[2] <= 1;
                    led[15:8] <= write_addr[7:0];  // Show progress
                    
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        // First, store the previous byte if we have data pending
                        if (write_addr > 0 && write_addr <= TOTAL_BYTES) begin
                            if (write_addr - 1 < WEIGHT_SIZE) begin
                                // Storing weight
                                weight_bram[write_addr - 1] <= prev_byte;
                            end else begin
                                // Storing bias byte
                                // Assemble 4 bytes into 32-bit value (little-endian)
                                case (bias_byte_cnt)
                                    2'd0: bias_temp[7:0]   <= prev_byte;
                                    2'd1: bias_temp[15:8]  <= prev_byte;
                                    2'd2: bias_temp[23:16] <= prev_byte;
                                    2'd3: begin
                                        bias_temp[31:24] <= prev_byte;
                                        // Write complete bias to BRAM
                                        // Bias index = (write_addr - 1 - WEIGHT_SIZE) / 4
                                        bias_bram[(write_addr - 1 - WEIGHT_SIZE) >> 2] <= 
                                            {prev_byte, bias_temp[23:0]};
                                    end
                                endcase
                                bias_byte_cnt <= bias_byte_cnt + 1;
                            end
                        end
                        
                        // Check for end marker after storing
                        if (prev_byte == END_BYTE1 && rx_data == END_BYTE2) begin
                            // End marker detected!
                            state <= STATE_DONE;
                            transfer_done <= 1;
                            led[2] <= 0;
                            led[3] <= 1;  // Success indicator
                        end else begin
                            // Check for overflow
                            if (write_addr >= TOTAL_BYTES + 1) begin
                                state <= STATE_ERROR;
                                led[4] <= 1;
                            end else begin
                                write_addr <= write_addr + 1;
                                prev_byte <= rx_data;
                            end
                        end
                    end
                end
                
                // ============================================
                // Transfer complete
                // ============================================
                STATE_DONE: begin
                    led[3] <= 1;  // Success - stays on
                    led[15:8] <= 8'hFF;  // All progress LEDs on
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


// =============================================================================
// UART Receiver Module (115200 baud compatible)
// =============================================================================
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
    localparam STATE_IDLE  = 2'd0;
    localparam STATE_START = 2'd1;
    localparam STATE_DATA  = 2'd2;
    localparam STATE_STOP  = 2'd3;

    reg [1:0] state;
    reg [15:0] clk_cnt;           // Clock counter (16 bits for flexibility)
    reg [2:0] bit_cnt;            // Bit counter (0-7)
    reg [7:0] rx_shift;           // Shift register for incoming bits
    reg rx_sync1, rx_sync2;       // Double-flop synchronizer

    // Synchronize RX input (2-stage synchronizer for metastability)
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

