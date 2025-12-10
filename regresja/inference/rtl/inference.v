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
  - Pure logic module - no UART or I/O
  - Weights/biases from weight_loader module
  - Input pixels from image RAM

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
                    // Pipeline flush cycle 1: multiply existing registers, accumulate previous product
                    // Don't load new data - weight_reg and pixel_reg already contain last values
                    product <= $signed(weight_reg) * $signed(pixel_reg);
                    accumulator <= accumulator + {{16{product[15]}}, product};
                    
                    state <= STATE_COMPARE;
                end
                
                // ============================================
                // COMPARE: Pipeline flush cycle 2 - accumulate the last product
                // ============================================
                STATE_COMPARE: begin
                    // Accumulate the product computed in ADD_BIAS state
                    // This is the final (784th) product
                    accumulator <= accumulator + {{16{product[15]}}, product};
                    
                    state <= STATE_NEXT_CLASS;
                end
                
                // ============================================
                // NEXT_CLASS: Calculate final score and compare
                // ============================================
                STATE_NEXT_CLASS: begin : next_class_block
                    // Final score = accumulator (all 784 products: 0..783)
                    //             + bias
                    // Product was already accumulated in COMPARE state
                    reg signed [31:0] final_score;
                    final_score = accumulator + current_bias;
                    
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
