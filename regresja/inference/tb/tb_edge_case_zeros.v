/*
================================================================================
Test 3: Edge Case - All Zeros
================================================================================

This testbench verifies inference behavior when:
  - All input pixels are 0
  - All weights are 0
  - Only biases are non-zero

Expected Behavior:
  - All accumulators should remain at 0 (since 0 × 0 = 0)
  - Final scores should equal the bias values
  - Predicted digit should be the class with the maximum bias

Test Strategy:
  - Set all weights to 0
  - Set all input pixels to 0
  - Set unique bias values for each class (0 through 9 multiplied by 1000)
  - Verify accumulator stays at 0 during computation
  - Verify predicted digit matches the class with max bias (class 9)

This test validates:
  1. Proper handling of zero inputs
  2. Correct bias addition
  3. Argmax selection based purely on biases
  4. No garbage accumulation in edge cases

================================================================================
*/

`timescale 1ns / 1ps

module tb_edge_case_zeros();

    reg clk;
    reg rst;
    
    // Memory Interfaces
    wire [12:0] weight_addr;
    reg  [7:0]  weight_data;
    wire [3:0]  bias_addr;
    reg  [31:0] bias_data;
    wire [9:0]  input_addr;
    reg  [7:0]  input_pixel;
    
    // Control
    reg weights_ready;
    reg start_inference;
    
    // Outputs
    wire [3:0] predicted_digit;
    wire inference_done;
    wire busy;
    
    // Test monitoring
    reg test_passed;
    integer class_idx;
    
    // Instantiate the Unit Under Test (UUT)
    inference uut (
        .clk(clk),
        .rst(rst),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .bias_addr(bias_addr),
        .bias_data(bias_data),
        .weights_ready(weights_ready),
        .start_inference(start_inference),
        .input_pixel(input_pixel),
        .input_addr(input_addr),
        .predicted_digit(predicted_digit),
        .inference_done(inference_done),
        .busy(busy)
    );
    
    // Clock Generation: 100MHz
    always #5 clk = ~clk;
    
    // ========================================================================
    // Mock Memory: All Zeros for Weights and Inputs
    // ========================================================================
    always @(posedge clk) begin
        // All weights are ZERO
        weight_data <= 8'd0;
        
        // All input pixels are ZERO
        input_pixel <= 8'd0;
        
        // Biases: Different for each class
        // Class 0: bias = 0
        // Class 1: bias = 1000
        // Class 2: bias = 2000
        // ...
        // Class 9: bias = 9000 (this should win)
        bias_data <= {28'd0, bias_addr} * 32'd1000;
    end
    
    // ========================================================================
    // Monitor Accumulator - Should Stay at 0
    // ========================================================================
    always @(posedge clk) begin
        if (!rst && busy) begin
            // During COMPUTE state, accumulator should remain 0
            if (uut.state == 3'd2) begin  // STATE_COMPUTE
                if (uut.accumulator !== 32'd0) begin
                    $display("[%0t] ERROR: Accumulator is non-zero during COMPUTE!", $time);
                    $display("    Class: %0d, Pixel: %0d, Accumulator: %0d",
                            uut.current_class, uut.current_pixel, uut.accumulator);
                    test_passed = 0;
                end
            end
            
            // After ADD_BIAS, accumulator should still be 0 (bias not yet added)
            if (uut.state == 3'd3) begin  // STATE_ADD_BIAS
                if (uut.accumulator !== 32'd0) begin
                    $display("[%0t] ERROR: Accumulator is non-zero in ADD_BIAS!", $time);
                    $display("    Class: %0d, Accumulator: %0d",
                            uut.current_class, uut.accumulator);
                    test_passed = 0;
                end
            end
        end
    end
    
    // ========================================================================
    // Monitor State Transitions
    // ========================================================================
    reg [2:0] prev_state;
    always @(posedge clk) begin
        if (rst) begin
            prev_state <= 0;
        end else if (busy) begin
            if (uut.state != prev_state) begin
                case (uut.state)
                    3'd1: begin // STATE_LOAD_BIAS
                        $display("[%0t] Class %0d: LOAD_BIAS - Bias value: %0d",
                                $time, uut.current_class, bias_data);
                    end
                    
                    3'd2: begin // STATE_COMPUTE
                        $display("[%0t] Class %0d: COMPUTE", $time, uut.current_class);
                    end
                    
                    3'd5: begin // STATE_NEXT_CLASS
                        // Check final score in next cycle (after bias is added)
                        $display("[%0t] Class %0d: NEXT_CLASS - Accumulator: %0d",
                                $time, uut.current_class, uut.accumulator);
                    end
                    
                    3'd6: begin // STATE_DONE
                        $display("[%0t] DONE - Predicted digit: %0d", $time, predicted_digit);
                    end
                endcase
            end
            prev_state = uut.state;
        end
    end
    
    // ========================================================================
    // Test Sequence
    // ========================================================================
    initial begin
        $display("================================================================================");
        $display("Test 3: Edge Case - All Zeros");
        $display("================================================================================");
        $display("Test Configuration:");
        $display("  - All weights:      0");
        $display("  - All input pixels: 0");
        $display("  - Biases:          Class N has bias = N * 1000");
        $display("  - Expected result:  Class 9 (highest bias = 9000)");
        $display("================================================================================\n");
        
        // Initialize
        clk = 0;
        rst = 1;
        weights_ready = 0;
        start_inference = 0;
        weight_data = 0;
        bias_data = 0;
        input_pixel = 0;
        test_passed = 1;
        prev_state = 0;
        
        // Reset
        #100;
        rst = 0;
        #20;
        
        $display("[%0t] Starting Test 3: All Zeros Edge Case\n", $time);
        weights_ready = 1;
        
        // Start inference
        #10 start_inference = 1;
        #10 start_inference = 0;
        
        // Wait for completion
        wait(inference_done);
        
        $display("\n[%0t] Inference complete!", $time);
        
        // ====================================================================
        // Verify Results
        // ====================================================================
        $display("\n================================================================================");
        $display("VERIFICATION RESULTS");
        $display("================================================================================");
        
        // Check predicted digit
        $display("\nPredicted Digit: %0d", predicted_digit);
        
        if (predicted_digit == 4'd9) begin
            $display("  ✓ CORRECT: Class 9 has the highest bias (9000)");
        end else begin
            $display("  ✗ INCORRECT: Expected class 9, got class %0d", predicted_digit);
            test_passed = 0;
        end
        
        // Summary
        $display("\n--------------------------------------------------------------------------------");
        if (test_passed) begin
            $display("*** TEST 3 PASSED ***");
            $display("  ✓ All weights were zero");
            $display("  ✓ All inputs were zero");
            $display("  ✓ Accumulator remained at 0 during computation");
            $display("  ✓ Prediction based purely on biases");
            $display("  ✓ Correctly predicted class with maximum bias (class 9)");
        end else begin
            $display("*** TEST 3 FAILED ***");
            $display("  See errors above for details");
        end
        $display("================================================================================");
        
        #100;
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #1000000;  // 1ms timeout
        $display("\nERROR: Simulation timeout!");
        $finish;
    end

endmodule




