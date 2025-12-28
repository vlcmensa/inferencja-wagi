/*
================================================================================
Dense Biases RAM (10 values, 32-bit each = 40 bytes)
================================================================================
*/

module dense_biases_ram (
    input wire clk,
    input wire [3:0] wr_addr,
    input wire [31:0] wr_data,
    input wire wr_en,
    input wire [3:0] rd_addr,
    output reg [31:0] rd_data
);

    (* ram_style = "block" *) reg [31:0] ram [0:9];
    
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
        rd_data <= ram[rd_addr];
    end

endmodule


