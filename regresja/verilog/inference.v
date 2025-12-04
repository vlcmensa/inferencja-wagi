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
    
    // LED and display registers
    reg [15:0] led_reg;
    reg [3:0] display_digit;
    
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
    // LED Control
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            led_reg <= 0;
            display_digit <= 0;
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
            
            if (inference_done) begin
                display_digit <= predicted_digit;
            end
        end
    end
    
    assign led = led_reg;
    
    // =========================================================================
    // 7-Segment Display
    // =========================================================================
    seven_segment_display u_display (
        .clk(clk),
        .rst(rst),
        .digit(display_digit),
        .seg(seg),
        .an(an)
    );
    
    // TX not used for now
    assign tx = 1'b1;

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
// 7-Segment Display Controller
// =============================================================================
module seven_segment_display (
    input wire clk,
    input wire rst,
    input wire [3:0] digit,       // 0-9 to display
    output reg [6:0] seg,         // Segments a-g (active low)
    output reg [3:0] an           // Anodes (active low)
);

    // Only use rightmost digit (an[0])
    always @(posedge clk) begin
        if (rst) begin
            an <= 4'b1110;  // Only an[0] active
            seg <= 7'b1111111;
        end else begin
            an <= 4'b1110;
            
            // 7-segment encoding (active low)
            // Segment order: gfedcba
            case (digit)
                4'd0: seg <= 7'b1000000;
                4'd1: seg <= 7'b1111001;
                4'd2: seg <= 7'b0100100;
                4'd3: seg <= 7'b0110000;
                4'd4: seg <= 7'b0011001;
                4'd5: seg <= 7'b0010010;
                4'd6: seg <= 7'b0000010;
                4'd7: seg <= 7'b1111000;
                4'd8: seg <= 7'b0000000;
                4'd9: seg <= 7'b0010000;
                default: seg <= 7'b0111111;  // Dash for invalid
            endcase
        end
    end

endmodule

