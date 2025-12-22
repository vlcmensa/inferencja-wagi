/*
================================================================================
UART Router - Single RX Demultiplexer for the Entire System
================================================================================

This module is the ONLY uart_rx instance in the system. It receives all bytes
and routes them to the appropriate handler based on protocol markers:

Protocols:
  1. Weight Loading:  0xAA 0x55 ... data ... 0x55 0xAA
  2. Image Loading:   0xBB 0x66 ... data ... 0x66 0xBB
  3. Digit Request:   0xCC (single byte command)
  4. Scores Request:  0xCD (single byte command)

Architecture:
  - Single uart_rx receives ALL bytes
  - Protocol state machine determines where bytes go
  - Data is routed to weight_loader, image_loader, or command handlers

Model: Two-Hidden-Layer Neural Network for MNIST
  - Layer 1: 784 -> 16 (12,544 weights + 64 biases)
  - Layer 2: 16 -> 16 (256 weights + 64 biases)
  - Layer 3: 16 -> 10 (160 weights + 40 biases)
  - Total: 13,128 bytes (12,960 weights + 168 biases)

================================================================================
*/

module uart_router (
    input wire clk,
    input wire rst,
    input wire rx,                    // UART RX line (from PC)
    
    // Status inputs
    input wire weights_loaded,        // HIGH when weights are fully loaded
    
    // Weight loader interface
    output reg [7:0] weight_rx_data,
    output reg weight_rx_ready,
    
    // Image loader interface
    output reg [7:0] image_rx_data,
    output reg image_rx_ready,
    
    // Command interface (for digit_reader and scores_reader)
    output reg [7:0] cmd_rx_data,
    output reg cmd_rx_ready
);

    // Protocol markers
    localparam WEIGHT_START1 = 8'hAA;
    localparam WEIGHT_START2 = 8'h55;
    localparam WEIGHT_END1   = 8'h55;
    localparam WEIGHT_END2   = 8'hAA;
    
    localparam IMAGE_START1 = 8'hBB;
    localparam IMAGE_START2 = 8'h66;
    localparam IMAGE_END1   = 8'h66;
    localparam IMAGE_END2   = 8'hBB;
    
    localparam CMD_DIGIT_READ  = 8'hCC;
    localparam CMD_SCORES_READ = 8'hCD;
    
    // Data sizes
    localparam WEIGHT_DATA_SIZE = 13128;  // 12,960 weights + 168 bias bytes
    localparam IMAGE_DATA_SIZE = 784;     // 784 pixels
    
    // Router states
    localparam STATE_IDLE          = 4'd0;
    localparam STATE_WAIT_WEIGHT2  = 4'd1;
    localparam STATE_RECEIVING_WEIGHTS = 4'd2;
    localparam STATE_WAIT_IMAGE2   = 4'd3;
    localparam STATE_RECEIVING_IMAGE = 4'd4;
    
    // Shared UART RX
    wire [7:0] rx_data;
    wire rx_ready;
    
    uart_rx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_uart_rx (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .data(rx_data),
        .ready(rx_ready)
    );
    
    // State registers
    reg [3:0] state;
    reg [14:0] byte_count;  // Counter for data bytes (up to 13,128, needs 15 bits)
    reg [7:0] prev_byte;    // For detecting end markers
    
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            byte_count <= 0;
            prev_byte <= 0;
            weight_rx_data <= 0;
            weight_rx_ready <= 0;
            image_rx_data <= 0;
            image_rx_ready <= 0;
            cmd_rx_data <= 0;
            cmd_rx_ready <= 0;
        end else begin
            // Default: all ready signals are pulses (1 cycle only)
            weight_rx_ready <= 0;
            image_rx_ready <= 0;
            cmd_rx_ready <= 0;
            
            case (state)
                // ============================================================
                // IDLE: Detect protocol start markers or single-byte commands
                // ============================================================
                STATE_IDLE: begin
                    if (rx_ready) begin
                        case (rx_data)
                            // Weight loading start marker (first byte)
                            WEIGHT_START1: begin
                                if (!weights_loaded) begin
                                    state <= STATE_WAIT_WEIGHT2;
                                end
                                // If weights already loaded, ignore
                            end
                            
                            // Image loading start marker (first byte)
                            IMAGE_START1: begin
                                if (weights_loaded) begin
                                    state <= STATE_WAIT_IMAGE2;
                                end
                                // If weights not loaded, ignore
                            end
                            
                            // Single-byte commands (only valid after weights loaded)
                            CMD_DIGIT_READ, CMD_SCORES_READ: begin
                                if (weights_loaded) begin
                                    cmd_rx_data <= rx_data;
                                    cmd_rx_ready <= 1;
                                end
                            end
                            
                            default: begin
                                // Unknown byte in IDLE, ignore
                            end
                        endcase
                    end
                end
                
                // ============================================================
                // Wait for second weight start marker (0x55)
                // ============================================================
                STATE_WAIT_WEIGHT2: begin
                    if (rx_ready) begin
                        if (rx_data == WEIGHT_START2) begin
                            // Valid start sequence, begin receiving weights
                            state <= STATE_RECEIVING_WEIGHTS;
                            byte_count <= 0;
                            prev_byte <= 0;
                            
                            // Forward start markers to weight_loader
                            weight_rx_data <= WEIGHT_START2;
                            weight_rx_ready <= 1;
                        end else if (rx_data == WEIGHT_START1) begin
                            // Another 0xAA, stay waiting for 0x55
                            state <= STATE_WAIT_WEIGHT2;
                        end else begin
                            // Invalid sequence, go back to idle
                            state <= STATE_IDLE;
                        end
                    end
                end
                
                // ============================================================
                // Receiving weight data
                // ============================================================
                STATE_RECEIVING_WEIGHTS: begin
                    if (rx_ready) begin
                        // Forward byte to weight_loader
                        weight_rx_data <= rx_data;
                        weight_rx_ready <= 1;
                        
                        // Increment counter
                        byte_count <= byte_count + 1;
                        prev_byte <= rx_data;

                        // === CRITICAL FIX START ===
                        // Only check for end marker if we have received at least the expected amount of data.
                        // This prevents random weight values that happen to be 0x55 0xAA from 
                        // terminating the transfer early.
                        if (byte_count >= WEIGHT_DATA_SIZE) begin
                            if (prev_byte == WEIGHT_END1 && rx_data == WEIGHT_END2) begin
                                // End of weight transfer
                                state <= STATE_IDLE;
                            end
                        end
                        // === CRITICAL FIX END ===
                        
                        // Safety: Hard abort if way too many bytes (prevents locking up if marker is missed)
                        if (byte_count > WEIGHT_DATA_SIZE + 10) begin
                            state <= STATE_IDLE;
                        end
                    end
                end
                
                // ============================================================
                // Wait for second image start marker (0x66)
                // ============================================================
                STATE_WAIT_IMAGE2: begin
                    if (rx_ready) begin
                        if (rx_data == IMAGE_START2) begin
                            // Valid start sequence, begin receiving image
                            state <= STATE_RECEIVING_IMAGE;
                            byte_count <= 0;
                            prev_byte <= 0;
                            
                            // Forward start markers to image_loader
                            image_rx_data <= IMAGE_START2;
                            image_rx_ready <= 1;
                        end else if (rx_data == IMAGE_START1) begin
                            // Another 0xBB, stay waiting for 0x66
                            state <= STATE_WAIT_IMAGE2;
                        end else begin
                            // Invalid sequence, go back to idle
                            state <= STATE_IDLE;
                        end
                    end
                end
                
                // ============================================================
                // Receiving image data
                // ============================================================
                STATE_RECEIVING_IMAGE: begin
                    if (rx_ready) begin
                        // Forward byte to image_loader
                        image_rx_data <= rx_data;
                        image_rx_ready <= 1;
                        
                        // Increment counter
                        byte_count <= byte_count + 1;
                        prev_byte <= rx_data;

                        // === CRITICAL FIX START ===
                        // Only check for end marker if we have received enough data.
                        if (byte_count >= IMAGE_DATA_SIZE) begin
                            if (prev_byte == IMAGE_END1 && rx_data == IMAGE_END2) begin
                                // End of image transfer
                                state <= STATE_IDLE;
                            end
                        end
                        // === CRITICAL FIX END ===

                        // Safety
                        if (byte_count > IMAGE_DATA_SIZE + 10) begin
                            state <= STATE_IDLE;
                        end
                    end
                end
                
                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule