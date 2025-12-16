/*
================================================================================
UART Receiver Module (115200 baud compatible)
================================================================================
Single shared UART RX for the entire system.
Other modules should NOT instantiate their own uart_rx.
================================================================================
*/

module uart_rx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input wire clk,
    input wire rst,
    input wire rx,
    output reg [7:0] data,
    output reg ready
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    // State definitions
    localparam STATE_IDLE  = 2'd0;
    localparam STATE_START = 2'd1;
    localparam STATE_DATA  = 2'd2;
    localparam STATE_STOP  = 2'd3;

    reg [1:0] state;
    reg [15:0] clk_cnt;           // Clock counter (16 bits for flexibility)
    reg [2:0] bit_cnt;            // Bit counter (0-7)
    reg [7:0] rx_shift;           // Shift register for incoming bits
    reg rx_sync1, rx_sync2;       // Double-flop synchronizer

    // Synchronize RX input (2-stage synchronizer for metastability)
    always @(posedge clk) begin
        rx_sync1 <= rx;
        rx_sync2 <= rx_sync1;
    end

    // UART receive state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            clk_cnt <= 0;
            bit_cnt <= 0;
            data <= 0;
            rx_shift <= 0;
            ready <= 0;
        end else begin
            ready <= 0;  // Default: ready is only high for 1 cycle
            
            case (state)
                // ----------------------------------------
                // IDLE: Wait for start bit (falling edge)
                // ----------------------------------------
                STATE_IDLE: begin
                    clk_cnt <= 0;
                    bit_cnt <= 0;
                    if (rx_sync2 == 0) begin
                        state <= STATE_START;
                    end
                end
                
                // ----------------------------------------
                // START: Verify start bit at middle
                // ----------------------------------------
                STATE_START: begin
                    if (clk_cnt == (CLKS_PER_BIT / 2)) begin
                        if (rx_sync2 == 0) begin
                            // Valid start bit
                            clk_cnt <= 0;
                            state <= STATE_DATA;
                        end else begin
                            // False start (noise)
                            state <= STATE_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // DATA: Sample 8 data bits (LSB first)
                // ----------------------------------------
                STATE_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT) begin
                        clk_cnt <= 0;
                        rx_shift[bit_cnt] <= rx_sync2;
                        
                        if (bit_cnt == 7) begin
                            bit_cnt <= 0;
                            state <= STATE_STOP;
                        end else begin
                            bit_cnt <= bit_cnt + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // STOP: Wait for stop bit, output data
                // ----------------------------------------
                STATE_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT) begin
                        clk_cnt <= 0;
                        state <= STATE_IDLE;
                        data <= rx_shift;  // Transfer to output
                        ready <= 1;        // Signal data valid
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
