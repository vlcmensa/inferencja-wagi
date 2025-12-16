/*
================================================================================
Test 6: Edge Case - Mixed Signs (Positive × Negative)
================================================================================

This testbench verifies inference behavior with mixed sign values:
  - All input pixels: +127 (max positive int8)
  - All weights: -128 (min negative int8)
  - Biases: Different per class

Expected Behavior:
  - Each product: 127 × (-128) = -16,256 (NEGATIVE result!)
  - Accumulator per class: -16,256 × 784 = -12,744,704 (negative, no overflow)
  - No overflow should occur (-12,744,704 > -2,147,483,648)
  - Final scores: -12,744,704 + bias[class]
  - Predicted digit: class with maximum (least negative) final score

Key Validation:
  This test specifically validates that:
  - Positive × Negative = Negative (signed arithmetic correctness)
  - Negative accumulator values are handled properly
  - Argmax works correctly with negative scores
  - Bias addition can make negative scores positive or less negative
  - Sign extension and two's complement work with negative results

Test Strategy:
  - Set all weights to -128
  - Set all input pixels to +127
  - Set unique bias values for each class (large enough to differentiate)
  - Verify accumulator reaches expected NEGATIVE value (-12,744,704)
  - Verify no overflow occurs
  - Verify prediction based on maximum final score (least negative or most positive)

This test validates:
  1. Correct handling of mixed signs
  2. Proper signed multiplication: 127 × (-128) = -16,256
  3. Negative accumulator handling
  4. No spurious overflow with negative values
  5. Argmax with negative and/or mixed positive/negative scores
  6. Bias addition can overcome negative accumulator

Expected Accumulator Value:
  127 × (-128) × 784 = -16,256 × 784 = -12,744,704

With Biases (Class N gets bias = N × 2,000,000):
  Class 0: -12,744,704 + 0         = -12,744,704
  Class 1: -12,744,704 + 2,000,000 = -10,744,704
  Class 2: -12,744,704 + 4,000,000 =  -8,744,704
  ...
  Class 6: -12,744,704 + 12,000,000 = -744,704
  Class 7: -12,744,704 + 14,000,000 = 1,255,296 (POSITIVE!)
  Class 8: -12,744,704 + 16,000,000 = 3,255,296 (POSITIVE!)
  Class 9: -12,744,704 + 18,000,000 = 5,255,296 (POSITIVE! - WINNER)

32-bit Signed Range:
  Min: -2,147,483,648
  Max: +2,147,483,647
  ✓ -12,744,704 is well within range (no overflow expected)

================================================================================
*/

`timescale 1ns / 1ps

module tb_edge_case_mixed_signs();

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
    localparam signed [31:0] EXPECTED_ACCUMULATOR = -32'd12744704;  // 127 × (-128) × 784 = -16256 × 784
    localparam signed [15:0] PRODUCT_VALUE = -16'd16256;            // 127 × (-128) = -16256
    
    // Expected final scores
    reg signed [31:0] expected_scores [0:9];
    
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
    // Mock Memory: Mixed Signs
    // ========================================================================
    always @(posedge clk) begin
        // All weights are -128 (min negative int8)
        weight_data <= 8'h80;  // -128
        
        // All input pixels are +127 (max positive int8)
        input_pixel <= 8'd127;
        
        // Biases: Large values to overcome negative accumulator
        // Class N gets bias = N × 2,000,000
        // This ensures class 9 has highest final score
        bias_data <= bias_addr * 32'd2000000;
    end
    
    // ========================================================================
    // Initialize Expected Scores
    // ========================================================================
    initial begin
        for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
            expected_scores[class_idx] = EXPECTED_ACCUMULATOR + (class_idx * 32'd2000000);
        end
    end
    
    // ========================================================================
    // Monitor Accumulator - Should reach -12,744,704 (NEGATIVE!)
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
            
            // Check for unexpected positive accumulator during computation
            // With 127 × (-128), result should be NEGATIVE
            if (uut.state == 3'd2 && uut.current_pixel > 10'd10) begin  // COMPUTE, after initial products
                if (uut.accumulator > 0) begin
                    $display("[%0t] ERROR: Accumulator is positive! Should be negative.", $time);
                    $display("    Class: %0d, Pixel: %0d, Accumulator: %0d",
                            uut.current_class, uut.current_pixel, uut.accumulator);
                    $display("    127 × (-128) should produce NEGATIVE result!");
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
                        $display("[%0t] Class %0d: COMPUTE (784 MACs with 127×(-128))",
                                $time, uut.current_class);
                    end
                    
                    3'd5: begin // STATE_NEXT_CLASS
                        $display("[%0t] Class %0d: NEXT_CLASS - Accumulator: %0d, Expected final score: %0d",
                                $time, uut.current_class, uut.accumulator, expected_scores[uut.current_class]);
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
                    $display("[%0t] Product calculation verified: 127 × (-128) = %0d ✓",
                            $time, uut.product);
                    $display("    Note: Positive × Negative = NEGATIVE (correct!)");
                end else begin
                    $display("[%0t] ERROR: Product mismatch!", $time);
                    $display("    Expected: %0d (negative)", PRODUCT_VALUE);
                    $display("    Got:      %0d", uut.product);
                    if (uut.product > 0) begin
                        $display("    ERROR: Product is positive! Should be negative.");
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
        $display("Test 6: Edge Case - Mixed Signs (Positive × Negative)");
        $display("================================================================================");
        $display("Test Configuration:");
        $display("  - All weights:      -128 (min negative int8, 0x80)");
        $display("  - All input pixels: +127 (max positive int8)");
        $display("  - Product per MAC:  127 × (-128) = -16,256 (NEGATIVE!)");
        $display("  - Expected accumulator: -16,256 × 784 = -12,744,704 (NEGATIVE)");
        $display("  - Biases:          Class N has bias = N × 2,000,000");
        $display("  - Expected result:  Class 9 (highest final score)");
        $display("");
        $display("Expected Final Scores:");
        for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
            $display("    Class %0d: -12,744,704 + %0d = %0d",
                    class_idx, class_idx * 2000000, expected_scores[class_idx]);
        end
        $display("");
        $display("KEY TEST: Positive × Negative = Negative (signed arithmetic)");
        $display("KEY TEST: Argmax with negative and positive scores");
        $display("KEY TEST: Bias can overcome negative accumulator");
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
        
        $display("[%0t] Starting Test 6: Mixed Signs Edge Case\n", $time);
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
            $display("    Final score: %0d", expected_scores[9]);
        end else begin
            $display("  ✗ INCORRECT: Expected class 9, got class %0d", predicted_digit);
            $display("    Expected final score for class 9: %0d", expected_scores[9]);
            if (predicted_digit < 10) begin
                $display("    Actual predicted class %0d score: %0d", 
                        predicted_digit, expected_scores[predicted_digit]);
            end
            test_passed = 0;
        end
        
        // Check all accumulators were verified
        $display("\nAccumulator Verification:");
        for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
            if (accumulator_checked[class_idx]) begin
                $display("  Class %0d: ✓ Verified (accumulator = -12,744,704)", class_idx);
            end else begin
                $display("  Class %0d: ✗ Not checked", class_idx);
                test_passed = 0;
            end
        end
        
        // Display final score analysis
        $display("\nFinal Score Analysis:");
        $display("  Classes 0-6: NEGATIVE final scores");
        $display("  Classes 7-9: POSITIVE final scores (bias overcame negative accumulator)");
        $display("  Class 9:    Highest positive score → Winner");
        
        // Summary
        $display("\n--------------------------------------------------------------------------------");
        if (test_passed) begin
            $display("*** TEST 6 PASSED ***");
            $display("  ✓ All weights were -128 (min negative)");
            $display("  ✓ All inputs were +127 (max positive)");
            $display("  ✓ Products correctly calculated: 127 × (-128) = -16,256");
            $display("  ✓ Positive × Negative = Negative (signed arithmetic correct!)");
            $display("  ✓ Accumulator correctly reached: -12,744,704 (NEGATIVE)");
            $display("  ✓ No overflow occurred (value within 32-bit signed range)");
            $display("  ✓ Negative accumulator handled properly");
            $display("  ✓ Bias addition correctly overcame negative accumulator");
            $display("  ✓ Argmax correctly selected maximum from mixed pos/neg scores");
            $display("  ✓ Correctly predicted class 9 (highest final score)");
        end else begin
            $display("*** TEST 6 FAILED ***");
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








