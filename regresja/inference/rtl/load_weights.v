/*
================================================================================
Softmax Regression Weight Loader via Routed UART
================================================================================

Loads trained neural network weights and biases from PC to FPGA BRAM.

Model: Softmax Regression (Logistic Regression) for MNIST
  - Input:  784 pixels (28x28 image, 8-bit unsigned)
  - Output: 10 classes (digits 0-9)
  - Weights: 784 x 10 = 7840 values (8-bit signed)
  - Biases:  10 values (32-bit signed, stored as 4 bytes each little-endian)

Protocol:
  - Start marker: 0xAA 0x55 (handled by uart_router)
  - Data bytes: weights (7840 bytes) + biases (40 bytes) = 7880 bytes
  - End marker: 0x55 0xAA (handled by uart_router)

NOTE: This module NO LONGER has its own uart_rx!
      It receives pre-routed data from uart_router.

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
    
    // Routed UART signals
    wire [7:0] weight_rx_data;
    wire weight_rx_ready;
    
    // For testing: tie read addresses to 0
    assign weight_addr = 13'd0;
    assign bias_addr = 4'd0;

    // UART Router (single uart_rx for the system)
    uart_router u_uart_router (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weights_loaded(transfer_done),
        .weight_rx_data(weight_rx_data),
        .weight_rx_ready(weight_rx_ready),
        .image_rx_data(),      // Not used in standalone test
        .image_rx_ready(),     // Not used in standalone test
        .cmd_rx_data(),        // Not used in standalone test
        .cmd_rx_ready()        // Not used in standalone test
    );

    // Instantiate the weight loader
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx_data(weight_rx_data),
        .rx_ready(weight_rx_ready),
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
    
    // Routed UART interface (from uart_router)
    input wire [7:0] rx_data,     // Routed RX data
    input wire rx_ready,          // Routed RX ready signal
    
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

    // Memory sizes
    localparam WEIGHT_SIZE = 7840;   // 784 x 10 weights
    localparam BIAS_SIZE = 10;       // 10 biases
    localparam TOTAL_BYTES = 7880;   // 7840 + 40
    
    // Protocol markers (for end detection)
    localparam END_BYTE1 = 8'h55;
    localparam END_BYTE2 = 8'hAA;
    
    // State machine states
    localparam STATE_WAIT_DATA   = 3'd0;  // Waiting for first data byte
    localparam STATE_RECEIVING   = 3'd1;
    localparam STATE_DONE        = 3'd2;
    localparam STATE_ERROR       = 3'd3;

    // Block RAM for weights (8-bit values)
    (* ram_style = "block" *) reg [7:0] weight_bram [0:WEIGHT_SIZE-1];
    
    // Block RAM for biases (32-bit values, stored as 4 separate bytes then combined)
    (* ram_style = "block" *) reg [31:0] bias_bram [0:BIAS_SIZE-1];
    
    // State and control registers
    reg [2:0] state;
    reg [13:0] write_addr;       // Address for writing (0 to 7879)
    reg [7:0] prev_byte;         // Previous byte for end marker detection
    reg blink_toggle;            // For LED blinking
    reg first_byte;              // Flag to track if we've received the first data byte
    
    // Bias assembly registers (4 bytes -> 32 bits)
    reg [1:0] bias_byte_cnt;     // Which byte of bias we're receiving (0-3)
    reg [31:0] bias_temp;        // Temporary register for assembling bias

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
            state <= STATE_WAIT_DATA;
            write_addr <= 0;
            prev_byte <= 0;
            transfer_done <= 0;
            bias_byte_cnt <= 0;
            bias_temp <= 0;
            blink_toggle <= 0;
            first_byte <= 1;
            led <= 16'b0000_0000_0000_0010;  // led[1] = waiting for start
        end else begin
            
            case (state)
                // ============================================
                // Wait for first data byte (start marker handled by router)
                // ============================================
                STATE_WAIT_DATA: begin
                    led[1] <= 1;  // Waiting indicator
                    led[2] <= 0;
                    led[3] <= 0;
                    led[4] <= 0;
                    
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        // First byte received from router means start sequence was valid
                        // The router sends us the 0x55 (second start marker) first,
                        // then the actual data bytes
                        if (first_byte) begin
                            // This is the 0x55 from start marker, skip it
                            first_byte <= 0;
                        end else begin
                            // Real data starts here
                            state <= STATE_RECEIVING;
                            write_addr <= 0;
                            bias_byte_cnt <= 0;
                            bias_temp <= 0;
                            led[1] <= 0;
                            led[2] <= 1;  // Receiving indicator
                            
                            // Store first real data byte
                            prev_byte <= rx_data;
                            write_addr <= 1;
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
                    state <= STATE_WAIT_DATA;
                end
            endcase
        end
    end

endmodule
