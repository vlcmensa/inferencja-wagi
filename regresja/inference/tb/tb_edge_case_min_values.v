/*
================================================================================
Test 5: Edge Case - Minimum Negative Values
================================================================================

This testbench verifies inference behavior with minimum negative values:
  - All input pixels: -128 (min negative int8)
  - All weights: -128 (min negative int8)
  - Biases: Different per class

Expected Behavior:
  - Each product: (-128) × (-128) = +16,384 (POSITIVE result!)
  - Accumulator per class: 16,384 × 784 = 12,845,056 (positive, no overflow)
  - No overflow should occur (12,845,056 < 2,147,483,647)
  - Final scores: 12,845,056 + bias[class]
  - Predicted digit: class with maximum bias

Key Validation:
  This test specifically validates that:
  - Negative × Negative = Positive (signed arithmetic correctness)
  - Two's complement representation works properly
  - Sign extension in accumulator is correct
  - Large positive result from negative operands doesn't overflow

Test Strategy:
  - Set all weights to -128
  - Set all input pixels to -128
  - Set unique bias values for each class
  - Verify accumulator reaches expected POSITIVE value (12,845,056)
  - Verify no overflow occurs
  - Verify prediction based on bias values

This test validates:
  1. Correct handling of minimum negative values (-128)
  2. Proper signed multiplication: (-128) × (-128) = +16,384
  3. No spurious overflow with large positive result from negative operands
  4. Correct 32-bit signed accumulator behavior
  5. Two's complement arithmetic throughout pipeline

Expected Accumulator Value:
  (-128) × (-128) × 784 = 16,384 × 784 = 12,845,056

32-bit Signed Range:
  Min: -2,147,483,648
  Max: +2,147,483,647
  ✓ 12,845,056 is well within range (no overflow expected)

================================================================================
*/

`timescale 1ns / 1ps

module tb_edge_case_min_values();

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
    localparam signed [31:0] EXPECTED_ACCUMULATOR = 32'd12845056;  // (-128) × (-128) × 784 = 16384 × 784
    localparam signed [15:0] PRODUCT_VALUE = 16'd16384;            // (-128) × (-128) = +16384
    
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
    // Mock Memory: Minimum Negative Values
    // ========================================================================
    always @(posedge clk) begin
        // All weights are -128 (min negative int8)
        // In 8-bit two's complement: -128 = 0x80 = 8'b10000000
        weight_data <= 8'h80;  // -128
        
        // All input pixels are -128 (min negative int8)
        input_pixel <= 8'h80;  // -128
        
        // Biases: Different for each class
        // Class 0: bias = 0
        // Class 1: bias = 10000
        // Class 2: bias = 20000
        // ...
        // Class 9: bias = 90000 (this should win)
        bias_data <= {28'd0, bias_addr} * 32'd10000;
    end
    
    // ========================================================================
    // Monitor Accumulator - Should reach 12,845,056 (POSITIVE!)
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
            
            // Check for unexpected negative accumulator
            // With (-128) × (-128), result should be POSITIVE
            if (uut.state == 3'd2 || uut.state == 3'd3 || uut.state == 3'd4) begin  // COMPUTE, ADD_BIAS, COMPARE
                if (uut.accumulator < 0) begin
                    $display("[%0t] ERROR: Accumulator is negative! Should be positive.", $time);
                    $display("    Class: %0d, Pixel: %0d, Accumulator: %0d",
                            uut.current_class, uut.current_pixel, uut.accumulator);
                    $display("    (-128) × (-128) should produce POSITIVE result!");
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
                        $display("[%0t] Class %0d: COMPUTE (784 MACs with (-128)×(-128))",
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
                    $display("[%0t] Product calculation verified: (-128) × (-128) = %0d ✓",
                            $time, uut.product);
                    $display("    Note: Negative × Negative = POSITIVE (correct!)");
                end else begin
                    $display("[%0t] ERROR: Product mismatch!", $time);
                    $display("    Expected: %0d (positive)", PRODUCT_VALUE);
                    $display("    Got:      %0d", uut.product);
                    if (uut.product < 0) begin
                        $display("    ERROR: Product is negative! Should be positive.");
                    end
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
        $display("Test 5: Edge Case - Minimum Negative Values");
        $display("================================================================================");
        $display("Test Configuration:");
        $display("  - All weights:      -128 (min negative int8, 0x80)");
        $display("  - All input pixels: -128 (min negative int8, 0x80)");
        $display("  - Product per MAC:  (-128) × (-128) = +16,384 (POSITIVE!)");
        $display("  - Expected accumulator: 16,384 × 784 = 12,845,056 (POSITIVE)");
        $display("  - Biases:          Class N has bias = N × 10,000");
        $display("  - Expected result:  Class 9 (highest bias = 90,000)");
        $display("  - 32-bit range:    -2,147,483,648 to +2,147,483,647");
        $display("  - Overflow check:  12,845,056 + 90,000 = 12,935,056 < 2^31 ✓");
        $display("");
        $display("KEY TEST: Negative × Negative = Positive (signed arithmetic)");
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
        
        $display("[%0t] Starting Test 5: Minimum Negative Values Edge Case\n", $time);
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
            $display("    (12,845,056 + 90,000 = 12,935,056)");
        end else begin
            $display("  ✗ INCORRECT: Expected class 9, got class %0d", predicted_digit);
            test_passed = 0;
        end
        
        // Check all accumulators were verified
        $display("\nAccumulator Verification:");
        for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
            if (accumulator_checked[class_idx]) begin
                $display("  Class %0d: ✓ Verified (accumulator = 12,845,056)", class_idx);
            end else begin
                $display("  Class %0d: ✗ Not checked", class_idx);
                test_passed = 0;
            end
        end
        
        // Summary
        $display("\n--------------------------------------------------------------------------------");
        if (test_passed) begin
            $display("*** TEST 5 PASSED ***");
            $display("  ✓ All weights were -128 (min negative)");
            $display("  ✓ All inputs were -128 (min negative)");
            $display("  ✓ Products correctly calculated: (-128) × (-128) = +16,384");
            $display("  ✓ Negative × Negative = Positive (signed arithmetic correct!)");
            $display("  ✓ Accumulator correctly reached: 12,845,056 (POSITIVE)");
            $display("  ✓ No overflow occurred (value within 32-bit signed range)");
            $display("  ✓ Bias addition worked correctly");
            $display("  ✓ Correctly predicted class with maximum final score (class 9)");
            $display("  ✓ Two's complement arithmetic validated");
        end else begin
            $display("*** TEST 5 FAILED ***");
            $display("  See errors above for details");
        end
        $display("================================================================================");
        
        #100;
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000000;  // 100ms timeout
        $display("\nERROR: Simulation timeout!");
        $finish;
    end

endmodule








