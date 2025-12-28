/*
================================================================================
Top Module - Simple CNN Inference System
================================================================================
This module instantiates all sub-modules and connects them together.
Uses a SINGLE uart_rx instance (in uart_router) for the entire system.

Model Architecture:
  - Input: 784 pixels (28x28 image)
  - Conv Layer: 4 filters, 3x3 kernel, stride 1, no padding -> 26x26x4 feature map
  - ReLU activation
  - Dense Layer: 2704 inputs -> 10 outputs
================================================================================
*/

module top (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    input wire rx,                // UART RX Line
    output wire tx,               // UART TX Line
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
    
    // UART Router signals
    wire [7:0] weight_rx_data;
    wire weight_rx_ready;
    wire [7:0] image_rx_data;
    wire image_rx_ready;
    wire [7:0] cmd_rx_data;
    wire cmd_rx_ready;
    
    // Weight loader write signals
    wire [5:0] conv_w_wr_addr;
    wire [7:0] conv_w_wr_data;
    wire conv_w_wr_en;
    wire [3:0] conv_b_wr_addr;
    wire [31:0] conv_b_wr_data;
    wire conv_b_wr_en;
    wire [14:0] dense_w_wr_addr;
    wire [7:0] dense_w_wr_data;
    wire dense_w_wr_en;
    wire [3:0] dense_b_wr_addr;
    wire [31:0] dense_b_wr_data;
    wire dense_b_wr_en;
    wire weights_loaded;
    
    // Inference read signals for conv weights/biases
    wire [5:0] conv_w_rd_addr;
    wire [7:0] conv_w_rd_data;
    wire [3:0] conv_b_rd_addr;
    wire [31:0] conv_b_rd_data;
    
    // Inference read signals for dense weights/biases
    wire [14:0] dense_w_rd_addr;
    wire [7:0] dense_w_rd_data;
    wire [3:0] dense_b_rd_addr;
    wire [31:0] dense_b_rd_data;
    
    // Image RAM signals
    wire [9:0]  img_wr_addr;
    wire [7:0]  img_wr_data;
    wire        img_wr_en;
    wire        img_loaded;
    wire [9:0]  img_rd_addr;
    wire [7:0]  img_rd_data;
    
    // Inference signals
    wire [3:0]  predicted_digit;
    wire        inference_done;
    wire        start_inference_pulse;
    
    // Feature Map RAM signals (from ram_cnn)
    wire [11:0] fm_addr;
    wire [7:0]  fm_wr_data;
    wire        fm_wr_en;
    wire [7:0]  fm_rd_data;
    
    // Class scores from inference module
    wire signed [31:0] class_score_0;
    wire signed [31:0] class_score_1;
    wire signed [31:0] class_score_2;
    wire signed [31:0] class_score_3;
    wire signed [31:0] class_score_4;
    wire signed [31:0] class_score_5;
    wire signed [31:0] class_score_6;
    wire signed [31:0] class_score_7;
    wire signed [31:0] class_score_8;
    wire signed [31:0] class_score_9;
    
    // Predicted digit RAM signals
    wire digit_ram_wr_en;
    wire [7:0] digit_ram_rd_data;
    
    // Scores RAM signals
    wire scores_ram_wr_en;
    wire [5:0] scores_ram_rd_addr;
    wire [7:0] scores_ram_rd_data;
    
    // UART TX signals - shared between digit and scores readers
    wire [7:0] digit_tx_data;
    wire digit_tx_send;
    wire [7:0] scores_tx_data;
    wire scores_tx_send;
    wire [7:0] tx_data;
    wire tx_send;
    wire tx_busy;
    wire tx_out;
    
    // LED and display registers
    reg [15:0] led_reg;
    
    // Display digit signals
    wire [3:0] digit_left;   // From memory (persistent)
    wire [3:0] digit_right;  // Current predicted_digit
    
    // Edge detector for inference_done (to create write pulse)
    reg inference_done_prev;
    reg img_loaded_prev;
    
    // =========================================================================
    // UART Router - SINGLE uart_rx for the entire system
    // =========================================================================
    uart_router u_uart_router (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weights_loaded(weights_loaded),
        .weight_rx_data(weight_rx_data),
        .weight_rx_ready(weight_rx_ready),
        .image_rx_data(image_rx_data),
        .image_rx_ready(image_rx_ready),
        .cmd_rx_data(cmd_rx_data),
        .cmd_rx_ready(cmd_rx_ready)
    );
    
    // =========================================================================
    // Weight Loader (receives routed data from uart_router)
    // =========================================================================
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx_data(weight_rx_data),
        .rx_ready(weight_rx_ready),
        .transfer_done(weights_loaded),
        .conv_w_addr(conv_w_wr_addr),
        .conv_w_data(conv_w_wr_data),
        .conv_w_en(conv_w_wr_en),
        .conv_b_addr(conv_b_wr_addr),
        .conv_b_data(conv_b_wr_data),
        .conv_b_en(conv_b_wr_en),
        .dense_w_addr(dense_w_wr_addr),
        .dense_w_data(dense_w_wr_data),
        .dense_w_en(dense_w_wr_en),
        .dense_b_addr(dense_b_wr_addr),
        .dense_b_data(dense_b_wr_data),
        .dense_b_en(dense_b_wr_en)
    );
    
    // =========================================================================
    // Conv Weights RAM
    // =========================================================================
    conv_weights_ram u_conv_weights_ram (
        .clk(clk),
        .wr_addr(conv_w_wr_addr),
        .wr_data(conv_w_wr_data),
        .wr_en(conv_w_wr_en),
        .rd_addr(conv_w_rd_addr),
        .rd_data(conv_w_rd_data)
    );
    
    // =========================================================================
    // Conv Biases RAM
    // =========================================================================
    conv_biases_ram u_conv_biases_ram (
        .clk(clk),
        .wr_addr(conv_b_wr_addr),
        .wr_data(conv_b_wr_data),
        .wr_en(conv_b_wr_en),
        .rd_addr(conv_b_rd_addr),
        .rd_data(conv_b_rd_data)
    );
    
    // =========================================================================
    // Dense Biases RAM
    // =========================================================================
    dense_biases_ram u_dense_biases_ram (
        .clk(clk),
        .wr_addr(dense_b_wr_addr),
        .wr_data(dense_b_wr_data),
        .wr_en(dense_b_wr_en),
        .rd_addr(dense_b_rd_addr),
        .rd_data(dense_b_rd_data)
    );
    
    // =========================================================================
    // Image Loader (receives routed data from uart_router)
    // =========================================================================
    image_loader u_image_loader (
        .clk(clk),
        .rst(rst),
        .weights_loaded(weights_loaded),
        .rx_data(image_rx_data),
        .rx_ready(image_rx_ready),
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
        .rd_addr(img_rd_addr),
        .rd_data(img_rd_data)
    );
    
    // =========================================================================
    // CNN RAM (Feature Map + Dense Weights)
    // =========================================================================
    // For dense weights: write during weight loading, read during inference
    wire [14:0] dense_w_ram_addr = dense_w_wr_en ? dense_w_wr_addr : dense_w_rd_addr;
    
    ram_cnn u_ram_cnn (
        .clk(clk),
        .fm_addr(fm_addr),
        .fm_d(fm_wr_data),
        .fm_we(fm_wr_en),
        .fm_q(fm_rd_data),
        .dw_addr(dense_w_ram_addr),
        .dw_d(dense_w_wr_data),
        .dw_we(dense_w_wr_en),
        .dw_q(dense_w_rd_data)
    );
    
    // =========================================================================
    // Inference Module (CNN)
    // =========================================================================
    inference u_inference (
        .clk(clk),
        .rst(rst),
        .start(start_inference_pulse),
        .done(inference_done),
        .predicted_digit(predicted_digit),
        .img_addr(img_rd_addr),
        .img_data(img_rd_data),
        .conv_w_addr(conv_w_rd_addr),
        .conv_w_data(conv_w_rd_data),
        .conv_b_addr(conv_b_rd_addr),
        .conv_b_data(conv_b_rd_data),
        .dense_w_addr(dense_w_rd_addr),
        .dense_w_data(dense_w_rd_data),
        .dense_b_addr(dense_b_rd_addr),
        .dense_b_data(dense_b_rd_data),
        .fm_addr(fm_addr),
        .fm_wr_data(fm_wr_data),
        .fm_wr_en(fm_wr_en),
        .fm_rd_data(fm_rd_data),
        .class_score_0(class_score_0),
        .class_score_1(class_score_1),
        .class_score_2(class_score_2),
        .class_score_3(class_score_3),
        .class_score_4(class_score_4),
        .class_score_5(class_score_5),
        .class_score_6(class_score_6),
        .class_score_7(class_score_7),
        .class_score_8(class_score_8),
        .class_score_9(class_score_9)
    );
    
    // =========================================================================
    // Auto-start inference when image is loaded
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            img_loaded_prev <= 0;
        end else begin
            img_loaded_prev <= img_loaded;
        end
    end
    
    assign start_inference_pulse = img_loaded && !img_loaded_prev;
    
    // =========================================================================
    // Edge detector for inference_done (to write predicted digit to RAM)
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            inference_done_prev <= 0;
        end else begin
            inference_done_prev <= inference_done;
        end
    end
    
    assign digit_ram_wr_en = inference_done && !inference_done_prev;
    assign scores_ram_wr_en = inference_done && !inference_done_prev;
    
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
    // Scores RAM
    // =========================================================================
    scores_ram u_scores_ram (
        .clk(clk),
        .wr_en(scores_ram_wr_en),
        .score_0(class_score_0),
        .score_1(class_score_1),
        .score_2(class_score_2),
        .score_3(class_score_3),
        .score_4(class_score_4),
        .score_5(class_score_5),
        .score_6(class_score_6),
        .score_7(class_score_7),
        .score_8(class_score_8),
        .score_9(class_score_9),
        .rd_addr(scores_ram_rd_addr),
        .rd_data(scores_ram_rd_data)
    );
    
    // =========================================================================
    // UART TX for responses
    // =========================================================================
    uart_tx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_uart_tx (
        .clk(clk),
        .rst(rst),
        .data(tx_data),
        .send(tx_send),
        .tx(tx_out),
        .busy(tx_busy)
    );
    
    // =========================================================================
    // Digit Reader (0xCC protocol) - uses routed command interface
    // =========================================================================
    digit_reader u_digit_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(cmd_rx_data),
        .rx_ready(cmd_rx_ready),
        .digit_data(digit_ram_rd_data),
        .tx_data(digit_tx_data),
        .tx_send(digit_tx_send),
        .tx_busy(tx_busy)
    );
    
    // =========================================================================
    // Scores Reader (0xCD protocol) - uses routed command interface
    // =========================================================================
    scores_reader u_scores_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(cmd_rx_data),
        .rx_ready(cmd_rx_ready),
        .scores_data(scores_ram_rd_data),
        .scores_addr(scores_ram_rd_addr),
        .tx_data(scores_tx_data),
        .tx_send(scores_tx_send),
        .tx_busy(tx_busy)
    );
    
    // =========================================================================
    // TX Arbiter - Multiplex between digit_reader and scores_reader
    // =========================================================================
    assign tx_data = digit_tx_send ? digit_tx_data : scores_tx_data;
    assign tx_send = digit_tx_send | scores_tx_send;
    
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
            led_reg[3:0] <= predicted_digit;
            led_reg[4] <= inference_done;
            led_reg[5] <= img_loaded;
            led_reg[6] <= weights_loaded;
        end
    end
    
    assign led = led_reg;
    
    // =========================================================================
    // 7-Segment Display
    // =========================================================================
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


