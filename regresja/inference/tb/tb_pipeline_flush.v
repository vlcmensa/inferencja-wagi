/*
================================================================================
Pipeline Flush Testbench for Inference Module
================================================================================
This testbench verifies:
  1. Exactly 784 products are computed per class (not 783 or 785)
  2. Pipeline registers properly reset between classes
  3. Critical pipeline flush in STATE_ADD_BIAS and STATE_COMPARE
  4. No data carryover between classes due to pipeline registers

Test Strategy:
  - Use different weight/pixel patterns for each class
  - Monitor internal pipeline registers (weight_reg, pixel_reg, product)
  - Count MAC operations per class
  - Verify state transitions occur at correct pixel counts
  - Check accumulator integrity between classes
================================================================================
*/

`timescale 1ns / 1ps

module tb_pipeline_flush();

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

    // Monitoring counters
    integer mac_count [0:9];           // Count of MAC ops per class
    integer class_idx;
    integer prev_class;
    reg [31:0] expected_score [0:9];   // Expected scores for verification
    reg test_passed;
    
    // Previous values for edge detection
    reg [9:0] prev_input_addr;
    reg [2:0] prev_state;
    
    // Variables for memory mock logic
    integer class_num;
    integer pixel_idx;
    
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
    // Mock Memory: Different patterns for each class to detect carryover
    // ========================================================================
    always @(posedge clk) begin
        class_num = weight_addr / 784;
        pixel_idx = weight_addr % 784;
        
        // Pattern: Each class has unique weight pattern
        // Class 0: weight = pixel_idx % 10
        // Class 1: weight = -(pixel_idx % 10)
        // Class 2: weight = 5
        // Class 3: weight = -5
        // ... etc
        case (class_num)
            0: weight_data <= (pixel_idx % 10);
            1: weight_data <= -(pixel_idx % 10);
            2: weight_data <= 8'd5;
            3: weight_data <= -8'd5;
            4: weight_data <= (pixel_idx % 7);
            5: weight_data <= -(pixel_idx % 7);
            6: weight_data <= 8'd3;
            7: weight_data <= -8'd3;
            8: weight_data <= (pixel_idx % 13);
            9: weight_data <= -(pixel_idx % 13);
            default: weight_data <= 8'd0;
        endcase
        
        // Biases: Different for each class
        bias_data <= {28'd0, bias_addr} * 32'd1000;
        
        // Input pixels: Pattern based on position
        input_pixel <= (input_addr % 17) + 1;  // Range: 1-17
    end

    // ========================================================================
    // Monitor Pipeline Registers and Count Operations
    // ========================================================================
    always @(posedge clk) begin
        if (rst) begin
            prev_class = 0;
            prev_state = 0;
            prev_input_addr = 0;
            for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
                mac_count[class_idx] = 0;
            end
        end else if (busy) begin
            // Detect state transitions
            if (uut.state != prev_state) begin
                case (uut.state)
                    3'd1: begin // STATE_LOAD_BIAS
                        $display("[%0t] Class %0d: Entering LOAD_BIAS", $time, uut.current_class);
                        
                        // Check if pipeline registers are reset
                        if (uut.current_class > 0) begin
                            if (uut.weight_reg !== 0 || uut.pixel_reg !== 0 || uut.product !== 0) begin
                                $display("  ERROR: Pipeline registers NOT properly reset!");
                                $display("    weight_reg=%0d, pixel_reg=%0d, product=%0d",
                                        uut.weight_reg, uut.pixel_reg, uut.product);
                                test_passed = 0;
                            end else begin
                                $display("  PASS: Pipeline registers properly reset");
                            end
                        end
                    end
                    
                    3'd2: begin // STATE_COMPUTE
                        $display("[%0t] Class %0d: Entering COMPUTE", $time, uut.current_class);
                    end
                    
                    3'd3: begin // STATE_ADD_BIAS
                        $display("[%0t] Class %0d: Entering ADD_BIAS (Pipeline flush cycle 1)", $time, uut.current_class);
                        $display("    current_pixel=%0d (should be 783)", uut.current_pixel);
                        if (uut.current_pixel != 783) begin
                            $display("    ERROR: current_pixel should be 783!");
                            test_passed = 0;
                        end
                    end
                    
                    3'd4: begin // STATE_COMPARE
                        $display("[%0t] Class %0d: Entering COMPARE (Pipeline flush cycle 2)", $time, uut.current_class);
                    end
                    
                    3'd5: begin // STATE_NEXT_CLASS
                        $display("[%0t] Class %0d: Entering NEXT_CLASS - Computing final score", $time, uut.current_class);
                        $display("    Total MAC ops for this class: %0d", mac_count[uut.current_class]);
                        
                        // Verify exactly 784 MAC operations
                        if (mac_count[uut.current_class] != 784) begin
                            $display("    ERROR: Expected 784 MAC ops, got %0d!", mac_count[uut.current_class]);
                            test_passed = 0;
                        end else begin
                            $display("    PASS: Exactly 784 products computed");
                        end
                    end
                    
                    3'd6: begin // STATE_DONE
                        $display("[%0t] Entering DONE state", $time);
                    end
                endcase
            end
            
            // Count MAC operations based on when data is loaded into pipeline
            // In COMPUTE state, each time input_addr advances, we load a new weight-pixel pair
            if (uut.state == 3'd2) begin  // STATE_COMPUTE
                // Count when input_addr advances (new data loaded)
                if (input_addr != prev_input_addr) begin
                    mac_count[uut.current_class] = mac_count[uut.current_class] + 1;
                end
            end
            
            // Note: ADD_BIAS and COMPARE no longer load new data in the fixed implementation
            // They only flush the pipeline, so we don't count additional MACs there
            // The total count should be exactly 784 from COMPUTE state alone
            
            prev_state = uut.state;
            prev_input_addr = input_addr;
            prev_class = uut.current_class;
        end
    end

    // ========================================================================
    // Monitor Accumulator Reset Between Classes
    // ========================================================================
    always @(posedge clk) begin
        if (!rst && busy) begin
            // Check accumulator is reset when entering LOAD_BIAS
            if (uut.state == 3'd1 && prev_state != 3'd1) begin  // STATE_LOAD_BIAS entry
                if (uut.accumulator != 0) begin
                    $display("[%0t] ERROR: Accumulator not reset in LOAD_BIAS! Value: %0d",
                            $time, uut.accumulator);
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
        $display("Pipeline Flush Test - Verifying MAC operation count and register reset");
        $display("================================================================================");
        
        // Initialize
        clk = 0;
        rst = 1;
        weights_ready = 0;
        start_inference = 0;
        weight_data = 0;
        bias_data = 0;
        input_pixel = 0;
        test_passed = 1;
        
        for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
            mac_count[class_idx] = 0;
        end

        // Reset
        #100;
        rst = 0;
        #20;

        $display("\n[%0t] Starting inference test...", $time);
        weights_ready = 1;
        
        // Start inference
        #10 start_inference = 1;
        #10 start_inference = 0;

        // Wait for completion
        wait(inference_done);
        
        $display("\n[%0t] Inference complete!", $time);
        $display("Predicted digit: %0d", predicted_digit);
        
        // ====================================================================
        // Verify Results
        // ====================================================================
        $display("\n================================================================================");
        $display("VERIFICATION RESULTS");
        $display("================================================================================");
        
        $display("\nMAC Operation Counts per Class:");
        for (class_idx = 0; class_idx < 10; class_idx = class_idx + 1) begin
            $display("  Class %0d: %0d MAC operations %s", 
                    class_idx, 
                    mac_count[class_idx],
                    (mac_count[class_idx] == 784) ? "✓ PASS" : "✗ FAIL");
            if (mac_count[class_idx] != 784) begin
                test_passed = 0;
            end
        end
        
        $display("\n--------------------------------------------------------------------------------");
        if (test_passed) begin
            $display("*** ALL TESTS PASSED ***");
            $display("  ✓ All 10 classes computed exactly 784 products");
            $display("  ✓ Pipeline registers properly reset between classes");
            $display("  ✓ Accumulator properly reset for each class");
            $display("  ✓ Pipeline flush cycles working correctly");
        end else begin
            $display("*** TESTS FAILED ***");
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

