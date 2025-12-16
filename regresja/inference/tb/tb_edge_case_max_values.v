/*
================================================================================
Test 4: Edge Case - Maximum Values
================================================================================

This testbench verifies inference behavior with maximum positive values:
  - All input pixels: +127 (max positive int8)
  - All weights: +127 (max positive int8)
  - Biases: Different per class

Expected Behavior:
  - Each product: 127 × 127 = 16,129 (fits in 16-bit signed)
  - Accumulator per class: 16,129 × 784 = 12,645,136 (fits in 32-bit signed)
  - No overflow should occur (12,645,136 < 2,147,483,647)
  - Final scores: 12,645,136 + bias[class]
  - Predicted digit: class with maximum bias

Test Strategy:
  - Set all weights to +127
  - Set all input pixels to +127
  - Set unique bias values for each class
  - Verify accumulator reaches expected value (12,645,136)
  - Verify no overflow occurs
  - Verify prediction based on bias values

This test validates:
  1. Correct handling of maximum positive values
  2. No spurious overflow with legitimate large values
  3. Correct 32-bit accumulator behavior
  4. Proper signed arithmetic throughout pipeline

Expected Accumulator Value:
  127 × 127 × 784 = 12,645,136

32-bit Signed Range:
  Min: -2,147,483,648
  Max: +2,147,483,647
  ✓ 12,645,136 is well within range (no overflow expected)

================================================================================
*/

`timescale 1ns / 1ps

module tb_edge_case_max_values();

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
    
    // Expected values
    localparam signed [31:0] EXPECTED_ACCUMULATOR = 32'd12645136;  // 127 × 127 × 784
    localparam signed [15:0] PRODUCT_VALUE = 16'd16129;            // 127 × 127
    
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
    // Mock Memory: Maximum Positive Values
    // ========================================================================
    always @(posedge clk) begin
        // All weights are +127 (max positive int8)
        weight_data <= 8'd127;
        
        // All input pixels are +127 (max positive int8)
        input_pixel <= 8'd127;
        
        // Biases: Different for each class
        // Class 0: bias = 0
        // Class 1: bias = 10000
        // Class 2: bias = 20000
        // ...
        // Class 9: bias = 90000 (this should win)
        bias_data <= {28'd0, bias_addr} * 32'd10000;
    end
    
    // ========================================================================
    // Monitor Accumulator - Should reach 12,645,136
    // ========================================================================
    reg accumulator_checked [0:9];
    
    always @(posedge clk) begin
        if (rst) begin
            for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
                accumulator_checked[class_idx] = 0;
            end
        end else if (busy) begin
            // Check accumulator value when transitioning to NEXT_CLASS
            if (uut.state == 3'd5 && !accumulator_checked[uut.current_class]) begin
                accumulator_checked[uut.current_class] = 1;
                
                if (uut.accumulator == EXPECTED_ACCUMULATOR) begin
                    $display("[%0t] Class %0d: Accumulator correct ✓ (value: %0d)",
                            $time, uut.current_class, uut.accumulator);
                end else begin
                    $display("[%0t] Class %0d: ERROR - Accumulator mismatch!",
                            $time, uut.current_class);
                    $display("    Expected: %0d", EXPECTED_ACCUMULATOR);
                    $display("    Got:      %0d", uut.accumulator);
                    $display("    Diff:     %0d", uut.accumulator - EXPECTED_ACCUMULATOR);
                    test_passed = 0;
                end
            end
            
            // Check for overflow during computation
            // Overflow would show as negative value when we expect positive
            if (uut.state == 3'd2) begin  // STATE_COMPUTE
                if (uut.accumulator < 0) begin
                    $display("[%0t] ERROR: Accumulator went negative during COMPUTE!", $time);
                    $display("    Class: %0d, Pixel: %0d, Accumulator: %0d",
                            uut.current_class, uut.current_pixel, uut.accumulator);
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
                        $display("[%0t] Class %0d: COMPUTE (784 MACs with 127×127)",
                                $time, uut.current_class);
                    end
                    
                    3'd5: begin // STATE_NEXT_CLASS
                        $display("[%0t] Class %0d: NEXT_CLASS - Final accumulator: %0d",
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
    // Verify Product Calculation (spot check)
    // ========================================================================
    reg product_checked;
    always @(posedge clk) begin
        if (rst) begin
            product_checked = 0;
        end else if (busy && !product_checked) begin
            // Check product value during first computation
            if (uut.state == 3'd2 && uut.current_pixel == 10'd10) begin
                product_checked = 1;
                if (uut.product == PRODUCT_VALUE) begin
                    $display("[%0t] Product calculation verified: 127 × 127 = %0d ✓",
                            $time, uut.product);
                end else begin
                    $display("[%0t] ERROR: Product mismatch!", $time);
                    $display("    Expected: %0d", PRODUCT_VALUE);
                    $display("    Got:      %0d", uut.product);
                    test_passed = 0;
                end
            end
        end
    end
    
    // ========================================================================
    // Test Sequence
    // ========================================================================
    initial begin
        $display("================================================================================");
        $display("Test 4: Edge Case - Maximum Values");
        $display("================================================================================");
        $display("Test Configuration:");
        $display("  - All weights:      +127 (max positive int8)");
        $display("  - All input pixels: +127 (max positive int8)");
        $display("  - Product per MAC:  127 × 127 = 16,129");
        $display("  - Expected accumulator: 16,129 × 784 = 12,645,136");
        $display("  - Biases:          Class N has bias = N × 10,000");
        $display("  - Expected result:  Class 9 (highest bias = 90,000)");
        $display("  - 32-bit range:    -2,147,483,648 to +2,147,483,647");
        $display("  - Overflow check:  12,645,136 + 90,000 = 12,735,136 < 2^31 ✓");
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
        product_checked = 0;
        
        for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
            accumulator_checked[class_idx] = 0;
        end
        
        // Reset
        #100;
        rst = 0;
        #20;
        
        $display("[%0t] Starting Test 4: Maximum Values Edge Case\n", $time);
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
            $display("  ✓ CORRECT: Class 9 has the highest final score");
            $display("    (12,645,136 + 90,000 = 12,735,136)");
        end else begin
            $display("  ✗ INCORRECT: Expected class 9, got class %0d", predicted_digit);
            test_passed = 0;
        end
        
        // Check all accumulators were verified
        $display("\nAccumulator Verification:");
        for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
            if (accumulator_checked[class_idx]) begin
                $display("  Class %0d: ✓ Verified", class_idx);
            end else begin
                $display("  Class %0d: ✗ Not checked", class_idx);
                test_passed = 0;
            end
        end
        
        // Summary
        $display("\n--------------------------------------------------------------------------------");
        if (test_passed) begin
            $display("*** TEST 4 PASSED ***");
            $display("  ✓ All weights were +127 (max positive)");
            $display("  ✓ All inputs were +127 (max positive)");
            $display("  ✓ Products correctly calculated: 127 × 127 = 16,129");
            $display("  ✓ Accumulator correctly reached: 12,645,136");
            $display("  ✓ No overflow occurred (value within 32-bit signed range)");
            $display("  ✓ Bias addition worked correctly");
            $display("  ✓ Correctly predicted class with maximum final score (class 9)");
        end else begin
            $display("*** TEST 4 FAILED ***");
            $display("  See errors above for details");
        end
        $display("================================================================================");
        
        #100;
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000000;  // 100ms timeout (needs more time due to large computations)
        $display("\nERROR: Simulation timeout!");
        $finish;
    end

endmodule








