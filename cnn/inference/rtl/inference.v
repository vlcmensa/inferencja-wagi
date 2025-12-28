module inference (
    input wire clk,
    input wire rst,
    input wire start,
    output reg done,
    output reg [3:0] predicted_digit,

    // Interface to Input Image RAM (28x28)
    output reg [9:0] img_addr,
    input wire [7:0] img_data,

    // Interface to Conv Weights (Distributed RAM)
    output reg [5:0] conv_w_addr,
    input wire [7:0] conv_w_data,
    
    // Interface to Conv Biases
    output reg [3:0] conv_b_addr,
    input wire [31:0] conv_b_data,

    // Interface to Dense Weights (BRAM)
    output reg [14:0] dense_w_addr,
    input wire [7:0] dense_w_data,

    // Interface to Dense Biases
    output reg [3:0] dense_b_addr,
    input wire [31:0] dense_b_data,

    // Feature Map Storage
    output reg [11:0] fm_addr,
    output reg [7:0] fm_wr_data,
    output reg fm_wr_en,
    input wire [7:0] fm_rd_data,
    
    // Scores output
    output reg signed [31:0] class_score_0,
    output reg signed [31:0] class_score_1,
    output reg signed [31:0] class_score_2,
    output reg signed [31:0] class_score_3,
    output reg signed [31:0] class_score_4,
    output reg signed [31:0] class_score_5,
    output reg signed [31:0] class_score_6,
    output reg signed [31:0] class_score_7,
    output reg signed [31:0] class_score_8,
    output reg signed [31:0] class_score_9
);

    localparam IDLE = 0, LOAD_CONV_BIAS = 1, CONV_MULT = 2, CONV_SAVE = 3, LOAD_DENSE_BIAS = 4, DENSE_MULT = 5, DENSE_NEXT = 6, DONE = 7;
    reg [3:0] state;

    // Conv Iterators
    reg [1:0] filter_idx; // 0-3
    reg [4:0] row, col;   // 0-25
    reg [1:0] ky, kx;     // 0-2 (kernel)
    reg signed [31:0] acc;
    
    // Dense Iterators
    reg [3:0] class_idx; // 0-9
    reg [11:0] flat_idx; // 0-2703 (26*26*4)

    // Max Score Logic
    reg signed [31:0] max_score;
    reg signed [31:0] temp;

    // Helper wires for Pipelining (Next Address Calculation)
    reg [1:0] next_kx, next_ky;
    
    // Combinational logic for kernel counters
    always @(*) begin
        if (kx == 2) begin
            next_kx = 0;
            next_ky = ky + 1;
        end else begin
            next_kx = kx + 1;
            next_ky = ky;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            fm_wr_en <= 0;
            predicted_digit <= 0;
            img_addr <= 0;
            conv_w_addr <= 0;
            conv_b_addr <= 0;
            dense_w_addr <= 0;
            dense_b_addr <= 0;
            fm_addr <= 0;
            fm_wr_data <= 0;
            acc <= 0;
            filter_idx <= 0;
            row <= 0; col <= 0;
            ky <= 0; kx <= 0;
            class_idx <= 0;
            flat_idx <= 0;
            max_score <= 32'h80000000;
        end else begin
            fm_wr_en <= 0;

            case(state)
                IDLE: begin 
                    done <= 0;
                    if(start) begin
                        state <= LOAD_CONV_BIAS;
                        filter_idx <= 0; row <= 0; col <= 0;
                        max_score <= 32'h80000000;
                    end
                end

                // --- CONVOLUTION PHASE ---
                LOAD_CONV_BIAS: begin
                    conv_b_addr <= filter_idx;
                    acc <= $signed(conv_b_data); 
                    ky <= 0; kx <= 0;
                    
                    // [FIX] PRE-CHARGE ADDRESSES for the first item (0,0)
                    // This ensures data is valid when we enter CONV_MULT next cycle
                    img_addr <= (row + 0) * 28 + (col + 0);
                    conv_w_addr <= filter_idx * 9 + 0;
                    
                    state <= CONV_MULT;
                end

                CONV_MULT: begin
                    // 1. Accumulate using data requested in PREVIOUS cycle
                    // This data is now valid because we set the address in LOAD or the prev MULT cycle
                    acc <= acc + $signed(img_data) * $signed(conv_w_data);

                    // 2. Setup Address for NEXT cycle (Pipeline)
                    if (kx == 2 && ky == 2) begin
                        state <= CONV_SAVE;
                    end else begin
                        // Look ahead to next coordinate
                        img_addr <= (row + next_ky) * 28 + (col + next_kx);
                        conv_w_addr <= filter_idx * 9 + next_ky * 3 + next_kx;
                        
                        // Advance counters
                        kx <= next_kx;
                        ky <= next_ky;
                    end
                end

                CONV_SAVE: begin
                    temp = acc >>> 7; // Shift right by 7
                    if (temp < 0) temp = 0; // ReLU
                    if (temp > 127) temp = 127; // Saturation
                    
                    fm_addr <= filter_idx * 676 + row * 26 + col;
                    fm_wr_data <= temp[7:0];
                    fm_wr_en <= 1;

                    // Iterate
                    if (col == 25 && row == 25 && filter_idx == 3) begin
                        state <= LOAD_DENSE_BIAS;
                        class_idx <= 0;
                    end else if (col == 25 && row == 25) begin
                        filter_idx <= filter_idx + 1; row <= 0; col <= 0;
                        state <= LOAD_CONV_BIAS;
                    end else if (col == 25) begin
                        row <= row + 1; col <= 0;
                        state <= LOAD_CONV_BIAS;
                    end else begin
                        col <= col + 1;
                        state <= LOAD_CONV_BIAS;
                    end
                end

                // --- DENSE PHASE ---
                LOAD_DENSE_BIAS: begin
                    dense_b_addr <= class_idx;
                    acc <= $signed(dense_b_data);
                    flat_idx <= 0;
                    
                    // [FIX] PRE-CHARGE ADDRESSES for first dense weight (idx 0)
                    fm_addr <= 0;
                    dense_w_addr <= class_idx * 2704 + 0;
                    
                    state <= DENSE_MULT;
                end

                DENSE_MULT: begin
                    // 1. Accumulate current data (from address set in prev cycle)
                    acc <= acc + $signed(fm_rd_data) * $signed(dense_w_data);

                    // 2. Setup Address for NEXT cycle
                    if (flat_idx == 2703) begin
                        state <= DENSE_NEXT;
                    end else begin
                        flat_idx <= flat_idx + 1;
                        fm_addr <= flat_idx + 1;
                        dense_w_addr <= class_idx * 2704 + (flat_idx + 1);
                    end
                end

                DENSE_NEXT: begin
                    // Store Score
                    case (class_idx)
                        0: class_score_0 <= acc;
                        1: class_score_1 <= acc;
                        2: class_score_2 <= acc;
                        3: class_score_3 <= acc;
                        4: class_score_4 <= acc;
                        5: class_score_5 <= acc;
                        6: class_score_6 <= acc;
                        7: class_score_7 <= acc;
                        8: class_score_8 <= acc;
                        9: class_score_9 <= acc;
                    endcase

                    if (class_idx == 0 || acc > max_score) begin
                        max_score <= acc;
                        predicted_digit <= class_idx;
                    end

                    if (class_idx == 9) state <= DONE;
                    else begin
                        class_idx <= class_idx + 1;
                        state <= LOAD_DENSE_BIAS;
                    end
                end

                DONE: done <= 1;
            endcase
        end
    end
endmodule