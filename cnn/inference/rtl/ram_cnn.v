/*
================================================================================
CNN RAM Modules
================================================================================
Storage for:
  - Feature Map: 2704 bytes (26*26*4)
  - Dense Weights: 27040 bytes
================================================================================
*/

module ram_cnn (
    input clk,
    // Feature Map RAM (2704 bytes)
    input [11:0] fm_addr,
    input [7:0] fm_d,
    input fm_we,
    output reg [7:0] fm_q,
    
    // Dense Weights RAM (27040 bytes)
    input [14:0] dw_addr,
    input [7:0] dw_d,
    input dw_we,
    output reg [7:0] dw_q
);
    // Inferred BRAMs
    (* ram_style = "block" *) reg [7:0] fm_ram [0:2703];
    (* ram_style = "block" *) reg [7:0] dw_ram [0:27039];

    always @(posedge clk) begin
        if (fm_we) fm_ram[fm_addr] <= fm_d;
        fm_q <= fm_ram[fm_addr];
    end

    always @(posedge clk) begin
        if (dw_we) dw_ram[dw_addr] <= dw_d;
        dw_q <= dw_ram[dw_addr];
    end
endmodule


