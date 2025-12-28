/*
================================================================================
Weight Loader - CNN Architecture
================================================================================
Loads weights and biases for:
  - Conv Layer: 4 filters, 3x3 kernel (36 weights + 4 biases)
  - Dense Layer: 2704 inputs -> 10 outputs (27040 weights + 10 biases)
Total: 27132 bytes
================================================================================
*/

module weight_loader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,
    input wire rx_ready,
    output reg transfer_done,
    
    // Memory Write Interfaces
    output reg [5:0] conv_w_addr, // 36 bytes (0-35)
    output reg [7:0] conv_w_data,
    output reg conv_w_en,
    
    output reg [3:0] conv_b_addr, // 4 biases (written byte by byte -> assembled)
    output reg [31:0] conv_b_data, // We will output full 32-bit here
    output reg conv_b_en,

    output reg [14:0] dense_w_addr, // 27040 bytes
    output reg [7:0] dense_w_data,
    output reg dense_w_en,

    output reg [3:0] dense_b_addr,
    output reg [31:0] dense_b_data,
    output reg dense_b_en
);
    // Address Ranges (Sequential Arrival)
    localparam CONV_W_SIZE = 36;
    localparam CONV_B_SIZE = 16;
    localparam DENSE_W_SIZE = 27040;
    localparam DENSE_B_SIZE = 40;
    localparam TOTAL = CONV_W_SIZE + CONV_B_SIZE + DENSE_W_SIZE + DENSE_B_SIZE;

    reg [15:0] global_addr;
    reg [1:0] byte_idx; // For assembling 32-bit biases
    reg [31:0] bias_accum;
    reg first_byte_flag;

    always @(posedge clk) begin
        if (rst) begin
            transfer_done <= 0;
            global_addr <= 0;
            byte_idx <= 0;
            first_byte_flag <= 1;
            conv_w_en <= 0; conv_b_en <= 0; dense_w_en <= 0; dense_b_en <= 0;
        end else if (rx_ready) begin
            if (first_byte_flag) first_byte_flag <= 0; // Skip 0x55 marker from router
            else begin
                // 1. Conv Weights (0 - 35)
                if (global_addr < CONV_W_SIZE) begin
                    conv_w_addr <= global_addr;
                    conv_w_data <= rx_data;
                    conv_w_en <= 1;
                end 
                // 2. Conv Biases (36 - 51)
                else if (global_addr < CONV_W_SIZE + CONV_B_SIZE) begin
                    bias_accum[byte_idx*8 +: 8] <= rx_data;
                    if (byte_idx == 3) begin
                        conv_b_addr <= (global_addr - CONV_W_SIZE) >> 2;
                        conv_b_data <= {rx_data, bias_accum[23:0]};
                        conv_b_en <= 1;
                        byte_idx <= 0;
                    end else byte_idx <= byte_idx + 1;
                end
                // 3. Dense Weights (52 - 27091)
                else if (global_addr < CONV_W_SIZE + CONV_B_SIZE + DENSE_W_SIZE) begin
                    dense_w_addr <= (global_addr - (CONV_W_SIZE + CONV_B_SIZE));
                    dense_w_data <= rx_data;
                    dense_w_en <= 1;
                end
                // 4. Dense Biases (27092 - 27131)
                else if (global_addr < TOTAL) begin
                    bias_accum[byte_idx*8 +: 8] <= rx_data;
                    if (byte_idx == 3) begin
                        dense_b_addr <= (global_addr - (CONV_W_SIZE + CONV_B_SIZE + DENSE_W_SIZE)) >> 2;
                        dense_b_data <= {rx_data, bias_accum[23:0]};
                        dense_b_en <= 1;
                        byte_idx <= 0;
                    end else byte_idx <= byte_idx + 1;
                end

                global_addr <= global_addr + 1;
                
                // End Detection
                if (global_addr >= TOTAL - 1) transfer_done <= 1;
            end
        end else begin
            conv_w_en <= 0; conv_b_en <= 0; dense_w_en <= 0; dense_b_en <= 0;
        end
    end
endmodule


