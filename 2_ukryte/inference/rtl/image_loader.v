/*
================================================================================
Image Loader - Receives 784-byte image via routed UART data
================================================================================
Protocol: Receives data bytes from uart_router (start/end markers handled there)
          Data format: 784 pixel bytes
          
NOTE: This module NO LONGER has its own uart_rx!
      It receives pre-routed data from uart_router.
================================================================================
*/

module image_loader (
    input wire clk,
    input wire rst,
    input wire weights_loaded,        // Only accept images after weights loaded
    
    // Routed UART interface (from uart_router)
    input wire [7:0] rx_data,         // Routed RX data
    input wire rx_ready,              // Routed RX ready signal
    
    // Image RAM write interface
    output reg [9:0] wr_addr,
    output reg [7:0] wr_data,
    output reg wr_en,
    output reg image_loaded
);

    // Protocol: Data bytes only (markers handled by uart_router)
    // End marker detection: 0x66 0xBB after 784 bytes
    localparam IMG_END1 = 8'h66;
    localparam IMG_END2 = 8'hBB;
    localparam IMG_SIZE = 784;

    // States
    localparam STATE_RECEIVING = 2'd0;
    localparam STATE_DONE      = 2'd1;

    reg [1:0] state;
    reg [9:0] byte_count;
    reg [7:0] prev_byte;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_RECEIVING;
            wr_addr <= 0;
            wr_data <= 0;
            wr_en <= 0;
            byte_count <= 0;
            prev_byte <= 0;
            image_loaded <= 0;
        end else begin
            wr_en <= 0;
            image_loaded <= 0;
            
            // Only process if weights are loaded
            if (weights_loaded) begin
                case (state)
                    STATE_RECEIVING: begin
                        if (rx_ready) begin
                            // First, store the previous byte if we have data pending
                            if (byte_count > 0 && byte_count <= IMG_SIZE) begin
                                wr_addr <= byte_count - 1;
                                wr_data <= prev_byte;
                                wr_en <= 1;
                            end
                            
                            // Check for end marker after storing
                            if (prev_byte == IMG_END1 && rx_data == IMG_END2 && byte_count >= IMG_SIZE) begin
                                state <= STATE_DONE;
                                image_loaded <= 1;
                            end else begin
                                // Advance counter and store current byte for next iteration
                                byte_count <= byte_count + 1;
                                prev_byte <= rx_data;
                            end
                        end
                    end
                    
                    STATE_DONE: begin
                        // Go back to waiting for next image
                        state <= STATE_RECEIVING;
                        byte_count <= 0;
                        prev_byte <= 0;
                    end
                endcase
            end
        end
    end

endmodule
