/*
================================================================================
Conv Layer RAM Modules
================================================================================
- Conv Weights: 36 bytes (4 filters * 3*3 weights)
- Conv Biases: 16 bytes (4 biases * 4 bytes each)
================================================================================
*/

// Conv Weights RAM (36 bytes - distributed RAM is fine)
module conv_weights_ram (
    input wire clk,
    input wire [5:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [5:0] rd_addr,
    output reg [7:0] rd_data
);

    (* ram_style = "distributed" *) reg [7:0] ram [0:35];
    
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
        rd_data <= ram[rd_addr];
    end

endmodule

// Conv Biases RAM (4 values, 32-bit each = 16 bytes)
module conv_biases_ram (
    input wire clk,
    input wire [3:0] wr_addr,
    input wire [31:0] wr_data,
    input wire wr_en,
    input wire [3:0] rd_addr,
    output reg [31:0] rd_data
);

    (* ram_style = "block" *) reg [31:0] ram [0:3];
    
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
        rd_data <= ram[rd_addr];
    end

endmodule


