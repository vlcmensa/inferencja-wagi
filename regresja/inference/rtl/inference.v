/*
================================================================================
Softmax Regression Inference Module
================================================================================

Computes digit classification for MNIST images using softmax regression.

Model:
  - Input:  784 pixels (28x28 image, 8-bit signed, preprocessed)
  - Output: Predicted digit (0-9)
  
Computation for each class i (0-9):
  score[i] = sum(input[j] * weight[i][j], j=0..783) + bias[i]
  
  Where:
  - input[j]    : 8-bit signed (-128 to 127, preprocessed by Python)
  - weight[i][j]: 8-bit signed (-128 to 127)
  - bias[i]     : 32-bit signed
  - score[i]    : 32-bit signed (accumulator)

Final output = argmax(score[0..9])

Architecture:
  - Sequential processing: one MAC per clock cycle
  - For each class: 784 multiply-accumulate operations
  - Total: 784 x 10 = 7840 cycles for inference
  - At 100 MHz: ~78.4 µs per image

Interface:
  - Input image loaded via UART (protocol: 0xBB 0x66, 784 bytes, 0x66 0xBB)
  - Weights/biases from weight_loader module
  - Result displayed on LEDs and optional 7-segment

================================================================================
*/

module inference (
    input wire clk,
    input wire rst,
    
    // Weight memory interface
    output reg [12:0] weight_addr,     // 0 to 7839
    input wire [7:0]  weight_data,     // 8-bit signed weight
    
    // Bias memory interface
    output reg [3:0]  bias_addr,       // 0 to 9
    input wire [31:0] bias_data,       // 32-bit signed bias
    
    // Control
    input wire        weights_ready,   // HIGH when weights are loaded
    input wire        start_inference, // Pulse to start inference
    input wire [7:0]  input_pixel,     // Current input pixel value
    output reg [9:0]  input_addr,      // Address of pixel to read (0-783)
    
    // Outputs
    output reg [3:0]  predicted_digit, // 0-9 result
    output reg        inference_done,  // HIGH when inference complete
    output reg        busy             // HIGH during inference
);

    // State machine states
    localparam STATE_IDLE           = 3'd0;
    localparam STATE_LOAD_BIAS      = 3'd1;
    localparam STATE_COMPUTE        = 3'd2;
    localparam STATE_ADD_BIAS       = 3'd3;
    localparam STATE_COMPARE        = 3'd4;
    localparam STATE_NEXT_CLASS     = 3'd5;
    localparam STATE_DONE           = 3'd6;

    // Registers
    reg [2:0] state;
    reg [3:0] current_class;           // Current output class (0-9)
    reg [9:0] current_pixel;           // Current input pixel index (0-783)
    reg signed [31:0] accumulator;     // Running sum for current class
    reg signed [31:0] current_bias;    // Bias for current class
    reg signed [31:0] max_score;       // Maximum score seen so far
    reg [3:0] max_class;               // Class with maximum score
    
    // Pipeline registers for timing
    reg signed [7:0] weight_reg;       // Registered weight (signed)
    reg signed [7:0] pixel_reg;        // Registered input pixel (signed, preprocessed)
    reg signed [15:0] product;         // Multiplication result
    
    // Number of pixels per image
    localparam NUM_PIXELS = 784;
    localparam NUM_CLASSES = 10;

    // Main state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            current_class <= 0;
            current_pixel <= 0;
            accumulator <= 0;
            current_bias <= 0;
            max_score <= 32'h80000000;  // Minimum 32-bit signed value
            max_class <= 0;
            predicted_digit <= 0;
            inference_done <= 0;
            busy <= 0;
            weight_addr <= 0;
            bias_addr <= 0;
            input_addr <= 0;
            weight_reg <= 0;
            pixel_reg <= 0;
            product <= 0;
        end else begin
            
            // Default: inference_done is pulse
            inference_done <= 0;
            
            case (state)
                // ============================================
                // IDLE: Wait for start signal
                // ============================================
                STATE_IDLE: begin
                    busy <= 0;
                    if (start_inference && weights_ready) begin
                        state <= STATE_LOAD_BIAS;
                        current_class <= 0;
                        current_pixel <= 0;
                        accumulator <= 0;
                        max_score <= 32'h80000000;
                        max_class <= 0;
                        busy <= 1;
                        
                        // Request first bias
                        bias_addr <= 0;
                    end
                end
                
                // ============================================
                // LOAD_BIAS: Load bias for current class
                // ============================================
                STATE_LOAD_BIAS: begin
                    // Bias data available after 1 cycle delay
                    current_bias <= $signed(bias_data);
                    accumulator <= 0;
                    current_pixel <= 0;
                    
                    // CRITICAL: Reset pipeline registers to avoid garbage accumulation
                    weight_reg <= 0;
                    pixel_reg <= 0;
                    product <= 0;
                    
                    // Set up first weight and pixel addresses
                    // weight_addr = current_class * 784 + current_pixel
                    weight_addr <= current_class * NUM_PIXELS;
                    input_addr <= 0;
                    
                    state <= STATE_COMPUTE;
                end
                
                // ============================================
                // COMPUTE: Multiply-accumulate loop
                // ============================================
                STATE_COMPUTE: begin
                    // Pipeline stage 1: Register inputs
                    weight_reg <= $signed(weight_data);
                    pixel_reg <= input_pixel;
                    
                    // Pipeline stage 2: Multiply
                    // signed 8-bit weight * signed 8-bit pixel (preprocessed)
                    // Python sends signed int8 as two's complement bytes
                    // Result is signed 16-bit
                    product <= $signed(weight_reg) * $signed(pixel_reg);
                    
                    // Pipeline stage 3: Accumulate
                    accumulator <= accumulator + {{16{product[15]}}, product};
                    
                    // Advance to next pixel
                    if (current_pixel < NUM_PIXELS - 1) begin
                        current_pixel <= current_pixel + 1;
                        weight_addr <= weight_addr + 1;
                        input_addr <= current_pixel + 1;
                    end else begin
                        // Need 2 more cycles to flush pipeline
                        state <= STATE_ADD_BIAS;
                    end
                end
                
                // ============================================
                // ADD_BIAS: Flush pipeline and add bias
                // ============================================
                STATE_ADD_BIAS: begin
                    // Last multiply
                    weight_reg <= $signed(weight_data);
                    pixel_reg <= input_pixel;
                    product <= $signed(weight_reg) * $signed(pixel_reg);
                    accumulator <= accumulator + {{16{product[15]}}, product};
                    
                    state <= STATE_COMPARE;
                end
                
                // ============================================
                // COMPARE: Compute final product (no accumulation needed here)
                // ============================================
                STATE_COMPARE: begin
                    // Recompute product for W[783]*P[783]
                    // (weight_reg and pixel_reg already hold these values)
                    product <= $signed(weight_reg) * $signed(pixel_reg);
                    
                    // NOTE: accumulator already has 783 products (0..782) from STATE_ADD_BIAS
                    // We don't accumulate here - product will be added in final_score
                    
                    state <= STATE_NEXT_CLASS;
                end
                
                // ============================================
                // NEXT_CLASS: Calculate final score and compare
                // ============================================
                STATE_NEXT_CLASS: begin : next_class_block
                    // Final score = accumulator (783 products: 0..782)
                    //             + product (W[783]*P[783])
                    //             + bias
                    // Total: all 784 products + bias
                    reg signed [31:0] final_score;
                    final_score = accumulator + {{16{product[15]}}, product} + current_bias;
                    
                    // Update maximum
                    if (final_score > max_score) begin
                        max_score <= final_score;
                        max_class <= current_class;
                    end
                    
                    if (current_class < NUM_CLASSES - 1) begin
                        // Move to next class
                        current_class <= current_class + 1;
                        bias_addr <= current_class + 1;
                        state <= STATE_LOAD_BIAS;
                    end else begin
                        // All classes processed
                        state <= STATE_DONE;
                    end
                end
                
                // ============================================
                // DONE: Output result
                // ============================================
                STATE_DONE: begin
                    predicted_digit <= max_class;
                    inference_done <= 1;
                    busy <= 0;
                    state <= STATE_IDLE;
                end
                
                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule


// =============================================================================
// TOP MODULE - Complete System with Weight Loading and Inference
// =============================================================================
module softmax_regression_top (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    input wire rx,                // UART RX Line
    output wire tx,               // UART TX Line (optional, for debug)
    output wire [15:0] led,       // Status LEDs
    output wire [6:0] seg,        // 7-segment display segments
    output wire [3:0] an,         // 7-segment display anodes
    input wire [15:0] sw,         // Switches (for control/debug)
    input wire btnU,              // Up button (optional)
    input wire btnD,              // Down button (optional)
    input wire btnL,              // Left button (optional)
    input wire btnR               // Right button (optional)
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    
    // Weight loader signals
    wire [12:0] weight_rd_addr;
    wire [7:0]  weight_rd_data;
    wire [3:0]  bias_rd_addr;
    wire [31:0] bias_rd_data;
    wire        weights_loaded;
    wire [15:0] loader_led;
    
    // Inference signals
    wire [12:0] inf_weight_addr;
    wire [3:0]  inf_bias_addr;
    wire [9:0]  inf_input_addr;
    wire [7:0]  inf_input_pixel;
    wire [3:0]  predicted_digit;
    wire        inference_done;
    wire        inference_busy;
    
    // Image RAM signals
    wire [9:0]  img_wr_addr;
    wire [7:0]  img_wr_data;
    wire        img_wr_en;
    wire        img_loaded;
    
    // Control signals
    wire start_inference_pulse;
    
    // Predicted digit RAM signals
    wire digit_ram_wr_en;
    wire [7:0] digit_ram_rd_data;
    
    // UART TX signals for digit reader
    wire [7:0] tx_data;
    wire tx_send;
    wire tx_busy;
    wire tx_out;
    
    // UART RX signals for digit reader
    wire [7:0] digit_rx_data;
    wire digit_rx_ready;
    
    // LED and display registers
    reg [15:0] led_reg;
    
    // Display digit signals
    wire [3:0] digit_left;   // From memory (persistent)
    wire [3:0] digit_right;  // Current predicted_digit
    
    // Edge detector for inference_done (to create write pulse)
    reg inference_done_prev;
    
    // =========================================================================
    // Weight Loader
    // =========================================================================
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weight_rd_addr(weight_rd_addr),
        .weight_rd_data(weight_rd_data),
        .bias_rd_addr(bias_rd_addr),
        .bias_rd_data(bias_rd_data),
        .transfer_done(weights_loaded),
        .led(loader_led)
    );
    
    // =========================================================================
    // Image Loader (separate UART protocol)
    // =========================================================================
    image_loader u_image_loader (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weights_loaded(weights_loaded),
        .wr_addr(img_wr_addr),
        .wr_data(img_wr_data),
        .wr_en(img_wr_en),
        .image_loaded(img_loaded)
    );
    
    // =========================================================================
    // Image RAM (784 bytes)
    // =========================================================================
    image_ram u_image_ram (
        .clk(clk),
        .wr_addr(img_wr_addr),
        .wr_data(img_wr_data),
        .wr_en(img_wr_en),
        .rd_addr(inf_input_addr),
        .rd_data(inf_input_pixel)
    );
    
    // =========================================================================
    // Inference Module
    // =========================================================================
    
    // Connect weight/bias addresses from inference to loader
    assign weight_rd_addr = inf_weight_addr;
    assign bias_rd_addr = inf_bias_addr;
    
    inference u_inference (
        .clk(clk),
        .rst(rst),
        .weight_addr(inf_weight_addr),
        .weight_data(weight_rd_data),
        .bias_addr(inf_bias_addr),
        .bias_data(bias_rd_data),
        .weights_ready(weights_loaded),
        .start_inference(start_inference_pulse),
        .input_pixel(inf_input_pixel),
        .input_addr(inf_input_addr),
        .predicted_digit(predicted_digit),
        .inference_done(inference_done),
        .busy(inference_busy)
    );
    
    // =========================================================================
    // Auto-start inference when image is loaded
    // =========================================================================
    reg img_loaded_prev;
    always @(posedge clk) begin
        if (rst)
            img_loaded_prev <= 0;
        else
            img_loaded_prev <= img_loaded;
    end
    
    assign start_inference_pulse = img_loaded && !img_loaded_prev;
    
    // =========================================================================
    // Edge detector for inference_done (to write predicted digit to RAM)
    // =========================================================================
    always @(posedge clk) begin
        if (rst)
            inference_done_prev <= 0;
        else
            inference_done_prev <= inference_done;
    end
    
    assign digit_ram_wr_en = inference_done && !inference_done_prev;
    
    // =========================================================================
    // Predicted Digit RAM
    // =========================================================================
    predicted_digit_ram u_predicted_digit_ram (
        .clk(clk),
        .wr_en(digit_ram_wr_en),
        .wr_data(predicted_digit),
        .rd_addr(1'b0),  // Always read from address 0
        .rd_data(digit_ram_rd_data)
    );
    
    // =========================================================================
    // UART RX for Digit Reader (separate instance)
    // =========================================================================
    uart_rx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_digit_rx (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .data(digit_rx_data),
        .ready(digit_rx_ready)
    );
    
    // =========================================================================
    // UART TX for responses
    // =========================================================================
    uart_tx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_digit_tx (
        .clk(clk),
        .rst(rst),
        .data(tx_data),
        .send(tx_send),
        .tx(tx_out),
        .busy(tx_busy)
    );
    
    // =========================================================================
    // Digit Reader
    // =========================================================================
    digit_reader u_digit_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(digit_rx_data),
        .rx_ready(digit_rx_ready),
        .digit_data(digit_ram_rd_data),
        .tx_data(tx_data),
        .tx_send(tx_send),
        .tx_busy(tx_busy)
    );
    
    // =========================================================================
    // Digit Display Reader - Reads from predicted_digit_ram
    // =========================================================================
    digit_display_reader u_digit_display_reader (
        .clk(clk),
        .rst(rst),
        .digit_ram_data(digit_ram_rd_data),
        .display_digit(digit_left)
    );
    
    // =========================================================================
    // LED Control
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            led_reg <= 0;
        end else begin
            // Show loader status when loading, inference result when done
            if (!weights_loaded) begin
                led_reg <= loader_led;
            end else begin
                led_reg[3:0] <= predicted_digit;
                led_reg[4] <= inference_busy;
                led_reg[5] <= inference_done;
                led_reg[6] <= img_loaded;
                led_reg[7] <= weights_loaded;
                led_reg[15:8] <= loader_led[15:8];
            end
        end
    end
    
    assign led = led_reg;
    
    // =========================================================================
    // 7-Segment Display
    // =========================================================================
    // Rightmost digit shows current predicted_digit, leftmost shows stored value from memory
    assign digit_right = predicted_digit;
    
    seven_segment_display u_display (
        .clk(clk),
        .rst(rst),
        .digit_left(digit_left),
        .digit_right(digit_right),
        .seg(seg),
        .an(an)
    );
    
    // Connect TX output
    assign tx = tx_out;

endmodule


// =============================================================================
// Image Loader - Receives 784-byte image via UART
// =============================================================================
module image_loader (
    input wire clk,
    input wire rst,
    input wire rx,
    input wire weights_loaded,    // Only accept images after weights loaded
    
    output reg [9:0] wr_addr,
    output reg [7:0] wr_data,
    output reg wr_en,
    output reg image_loaded
);

    // Protocol: 0xBB 0x66, 784 bytes, 0x66 0xBB
    localparam IMG_START1 = 8'hBB;
    localparam IMG_START2 = 8'h66;
    localparam IMG_END1 = 8'h66;
    localparam IMG_END2 = 8'hBB;
    localparam IMG_SIZE = 784;

    // States
    localparam STATE_WAIT_START1 = 3'd0;
    localparam STATE_WAIT_START2 = 3'd1;
    localparam STATE_RECEIVING   = 3'd2;
    localparam STATE_DONE        = 3'd3;

    // UART receiver (shared with weight loader - need to instantiate or use external)
    // For simplicity, we'll use the same rx line and filter by markers
    // In practice, you'd share the uart_rx module
    
    wire [7:0] rx_data;
    wire rx_ready;
    
    // Instantiate separate UART receiver for image loading
    uart_rx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_img_rx (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .data(rx_data),
        .ready(rx_ready)
    );
    
    reg [2:0] state;
    reg [9:0] byte_count;
    reg [7:0] prev_byte;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_WAIT_START1;
            wr_addr <= 0;
            wr_data <= 0;
            wr_en <= 0;
            byte_count <= 0;
            prev_byte <= 0;
            image_loaded <= 0;
        end else begin
            wr_en <= 0;
            image_loaded <= 0;
            
            // Only process if weights are loaded
            if (weights_loaded) begin
                case (state)
                    STATE_WAIT_START1: begin
                        if (rx_ready && rx_data == IMG_START1) begin
                            state <= STATE_WAIT_START2;
                        end
                    end
                    
                    STATE_WAIT_START2: begin
                        if (rx_ready) begin
                            if (rx_data == IMG_START2) begin
                                state <= STATE_RECEIVING;
                                byte_count <= 0;
                                prev_byte <= 0;
                            end else if (rx_data == IMG_START1) begin
                                state <= STATE_WAIT_START2;
                            end else begin
                                state <= STATE_WAIT_START1;
                            end
                        end
                    end
                    
                    STATE_RECEIVING: begin
                        if (rx_ready) begin
                            // First, store the previous byte if we have data pending
                            if (byte_count > 0 && byte_count <= IMG_SIZE) begin
                                wr_addr <= byte_count - 1;
                                wr_data <= prev_byte;
                                wr_en <= 1;
                            end
                            
                            // Check for end marker after storing
                            if (prev_byte == IMG_END1 && rx_data == IMG_END2 && byte_count >= IMG_SIZE) begin
                                state <= STATE_DONE;
                                image_loaded <= 1;
                            end else begin
                                // Advance counter and store current byte for next iteration
                                byte_count <= byte_count + 1;
                                prev_byte <= rx_data;
                            end
                        end
                    end
                    
                    STATE_DONE: begin
                        // Go back to waiting for next image
                        state <= STATE_WAIT_START1;
                    end
                endcase
            end
        end
    end

endmodule


// =============================================================================
// Image RAM (784 bytes)
// =============================================================================
module image_ram (
    input wire clk,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [9:0] rd_addr,
    output reg [7:0] rd_data
);

    (* ram_style = "block" *) reg [7:0] ram [0:783];
    
    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end
    
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule


// =============================================================================
// Digit Display Reader - Reads digit from predicted_digit_ram
// =============================================================================
module digit_display_reader (
    input wire clk,
    input wire rst,
    input wire [7:0] digit_ram_data,  // Data from predicted_digit_ram
    output reg [3:0] display_digit    // 4-bit digit extracted from memory
);

    always @(posedge clk) begin
        if (rst) begin
            display_digit <= 0;
        end else begin
            // Extract lower 4 bits (digit is stored in lower nibble)
            display_digit <= digit_ram_data[3:0];
        end
    end

endmodule


// =============================================================================
// 7-Segment Display Controller
// =============================================================================
module seven_segment_display (
    input wire clk,
    input wire rst,
    input wire [3:0] digit_left,   // Leftmost digit (from memory)
    input wire [3:0] digit_right,  // Rightmost digit (current)
    output reg [6:0] seg,           // Segments a-g (active low)
    output reg [3:0] an             // Anodes (active low)
);

    // Time-multiplexing counter for switching between digits
    // At 100 MHz, divide to ~1 kHz refresh rate (50,000 cycles per digit)
    localparam REFRESH_DIV = 17'd50000;
    
    reg [16:0] refresh_counter;
    reg digit_select;  // 0 = left, 1 = right
    
    // 7-segment encoding function
    function [6:0] seg_encode;
        input [3:0] digit;
        begin
            case (digit)
                4'd0: seg_encode = 7'b1000000;
                4'd1: seg_encode = 7'b1111001;
                4'd2: seg_encode = 7'b0100100;
                4'd3: seg_encode = 7'b0110000;
                4'd4: seg_encode = 7'b0011001;
                4'd5: seg_encode = 7'b0010010;
                4'd6: seg_encode = 7'b0000010;
                4'd7: seg_encode = 7'b1111000;
                4'd8: seg_encode = 7'b0000000;
                4'd9: seg_encode = 7'b0010000;
                default: seg_encode = 7'b0111111;  // Dash for invalid
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            refresh_counter <= 0;
            digit_select <= 0;
            an <= 4'b1111;  // All off
            seg <= 7'b1111111;
        end else begin
            // Increment refresh counter
            if (refresh_counter >= REFRESH_DIV - 1) begin
                refresh_counter <= 0;
                digit_select <= ~digit_select;  // Toggle between left and right
            end else begin
                refresh_counter <= refresh_counter + 1;
            end
            
            // Select which digit to display
            if (digit_select == 0) begin
                // Display leftmost digit (an[3])
                an <= 4'b0111;  // an[3] active (leftmost)
                seg <= seg_encode(digit_left);
            end else begin
                // Display rightmost digit (an[0])
                an <= 4'b1110;  // an[0] active (rightmost)
                seg <= seg_encode(digit_right);
            end
        end
    end

endmodule


// =============================================================================
// Predicted Digit RAM - Stores the predicted digit at a fixed address
// =============================================================================
// Memory Layout:
//   Address 0: Predicted digit (4-bit value, stored as 8-bit byte)
//
// This BRAM is separate from weight/bias/image memories.
// When inference completes, the predicted digit (0-9) is written to address 0.
// =============================================================================
module predicted_digit_ram (
    input wire clk,
    input wire wr_en,           // Write enable (pulse when inference_done)
    input wire [3:0] wr_data,   // Predicted digit value (0-9)
    input wire rd_addr,         // Read address (always 0 for now)
    output reg [7:0] rd_data    // Read data (digit in lower 4 bits)
);

    // Single byte BRAM
    (* ram_style = "block" *) reg [7:0] ram [0:0];
    
    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[0] <= {4'b0, wr_data};  // Store 4-bit digit in lower nibble
        end
    end
    
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule


// =============================================================================
// UART Transmitter Module (115200 baud compatible)
// =============================================================================
module uart_tx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input wire clk,
    input wire rst,
    input wire [7:0] data,      // Byte to transmit
    input wire send,            // Pulse to start transmission
    output reg tx,              // UART TX output line
    output reg busy             // HIGH while transmitting
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    // State definitions
    localparam STATE_IDLE  = 2'd0;
    localparam STATE_START = 2'd1;
    localparam STATE_DATA  = 2'd2;
    localparam STATE_STOP  = 2'd3;

    reg [1:0] state;
    reg [15:0] clk_cnt;           // Clock counter
    reg [2:0] bit_cnt;            // Bit counter (0-7)
    reg [7:0] tx_shift;           // Shift register for outgoing bits

    // UART transmit state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            clk_cnt <= 0;
            bit_cnt <= 0;
            tx <= 1'b1;  // Idle state is HIGH
            tx_shift <= 0;
            busy <= 0;
        end else begin
            case (state)
                // ----------------------------------------
                // IDLE: Wait for send signal
                // ----------------------------------------
                STATE_IDLE: begin
                    tx <= 1'b1;  // Idle state is HIGH
                    busy <= 0;
                    if (send) begin
                        state <= STATE_START;
                        tx_shift <= data;
                        clk_cnt <= 0;
                        busy <= 1;
                    end
                end
                
                // ----------------------------------------
                // START: Send start bit (LOW)
                // ----------------------------------------
                STATE_START: begin
                    tx <= 1'b0;  // Start bit is LOW
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        bit_cnt <= 0;
                        state <= STATE_DATA;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // DATA: Send 8 data bits (LSB first)
                // ----------------------------------------
                STATE_DATA: begin
                    tx <= tx_shift[bit_cnt];
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
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
                // STOP: Send stop bit (HIGH)
                // ----------------------------------------
                STATE_STOP: begin
                    tx <= 1'b1;  // Stop bit is HIGH
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        state <= STATE_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule


// =============================================================================
// Digit Reader - Handles UART read requests for predicted digit
// =============================================================================
// Protocol:
//   Request: Send byte 0xCC to request predicted digit
//   Response: FPGA sends 1 byte containing the predicted digit (0-9) in lower 4 bits
// =============================================================================
module digit_reader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,        // UART RX data
    input wire rx_ready,              // UART RX ready signal
    input wire [7:0] digit_data,      // Read data from predicted_digit_ram
    output reg [7:0] tx_data,         // Data to send via UART TX
    output reg tx_send,               // Pulse to start UART TX transmission
    input wire tx_busy                // UART TX busy signal
);

    // Request byte constant
    localparam REQUEST_BYTE = 8'hCC;
    
    // States
    localparam STATE_IDLE = 2'd0;
    localparam STATE_SEND_RESPONSE = 2'd1;
    
    reg [1:0] state;
    reg rx_ready_prev;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            tx_data <= 0;
            tx_send <= 0;
            rx_ready_prev <= 0;
        end else begin
            // Default: tx_send is pulse
            tx_send <= 0;
            
            // Detect rising edge of rx_ready
            rx_ready_prev <= rx_ready;
            
            case (state)
                // ----------------------------------------
                // IDLE: Wait for read request (0xCC)
                // ----------------------------------------
                STATE_IDLE: begin
                    if (rx_ready && !rx_ready_prev) begin
                        // New byte received
                        if (rx_data == REQUEST_BYTE) begin
                            // Valid request - prepare response
                            tx_data <= digit_data;
                            state <= STATE_SEND_RESPONSE;
                        end
                    end
                end
                
                // ----------------------------------------
                // SEND_RESPONSE: Send the digit via UART TX
                // ----------------------------------------
                STATE_SEND_RESPONSE: begin
                    if (!tx_busy) begin
                        // UART TX is idle, send the data
                        tx_send <= 1;
                        state <= STATE_IDLE;
                    end
                end
                
                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule

