/*
================================================================================
Image RAM (784 bytes)
================================================================================
*/

module image_ram (
    input wire clk,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [9:0] rd_addr,
    output reg [7:0] rd_data
);

    (* ram_style = "block" *) reg [7:0] ram [0:783];
    
    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end
    
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule


