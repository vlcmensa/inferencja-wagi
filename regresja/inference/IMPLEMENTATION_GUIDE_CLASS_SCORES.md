# Implementation Guide: Class Scores Storage Feature

## Context and Motivation

**Problem**: The existing inference system only saves the predicted digit (0-9) to BRAM, which is useful for basic accuracy testing. However, for detailed analysis, debugging, and understanding model confidence, we need access to all 10 class scores (the raw logits before argmax).

**Solution**: Extend the system to:
1. Store all 10 class scores (32-bit signed integers) in a separate BRAM
2. Add UART protocol (0xCD) to read these scores
3. Update Python test scripts to retrieve and display scores

**Design Decisions**:
- Use **separate BRAM** for scores (not expanding predicted_digit_ram) for cleaner separation of concerns
- Store scores in **little-endian format** for cross-platform compatibility
- Use **0xCD** as request byte (extends existing 0xCC protocol)
- Implement **TX arbiter** to share UART between digit and scores readers

---

## Implementation Steps

### Step 1: Modify `inference.v` - Add Score Outputs

**File**: `regresja/inference/rtl/inference.v`

#### Change 1.1: Add output ports for class scores

**Location**: Module declaration, after line 58 (after `busy` output)

**Find:**
```verilog
    // Outputs
    output reg [3:0]  predicted_digit, // 0-9 result
    output reg        inference_done,  // HIGH when inference complete
    output reg        busy             // HIGH during inference
);
```

**Replace with:**
```verilog
    // Outputs
    output reg [3:0]  predicted_digit, // 0-9 result
    output reg        inference_done,  // HIGH when inference complete
    output reg        busy,            // HIGH during inference
    
    // Class scores output (for accuracy analysis)
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
```

#### Change 1.2: Add register array to store scores

**Location**: After line 77 (after `max_class` register declaration)

**Find:**
```verilog
    // Registers
    reg [2:0] state;
    reg [3:0] current_class;           // Current output class (0-9)
    reg [9:0] current_pixel;           // Current input pixel index (0-783)
    reg signed [31:0] accumulator;     // Running sum for current class
    reg signed [31:0] current_bias;    // Bias for current class
    reg signed [31:0] max_score;       // Maximum score seen so far
    reg [3:0] max_class;               // Class with maximum score
```

**Replace with:**
```verilog
    // Registers
    reg [2:0] state;
    reg [3:0] current_class;           // Current output class (0-9)
    reg [9:0] current_pixel;           // Current input pixel index (0-783)
    reg signed [31:0] accumulator;     // Running sum for current class
    reg signed [31:0] current_bias;    // Bias for current class
    reg signed [31:0] max_score;       // Maximum score seen so far
    reg [3:0] max_class;               // Class with maximum score
    
    // Array to store all class scores for output
    reg signed [31:0] class_scores [0:9];
```

#### Change 1.3: Initialize score registers and array

**Location**: In reset block, after line 106 (after `product <= 0;`)

**Find:**
```verilog
    // Main state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            current_class <= 0;
            current_pixel <= 0;
            accumulator <= 0;
            current_bias <= 0;
            max_score <= 32'h80000000;  // Minimum 32-bit signed value
            max_class <= 0;
            predicted_digit <= 0;
            inference_done <= 0;
            busy <= 0;
            weight_addr <= 0;
            bias_addr <= 0;
            input_addr <= 0;
            weight_reg <= 0;
            pixel_reg <= 0;
            product <= 0;
        end else begin
```

**Replace with:**
```verilog
    // Main state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            current_class <= 0;
            current_pixel <= 0;
            accumulator <= 0;
            current_bias <= 0;
            max_score <= 32'h80000000;  // Minimum 32-bit signed value
            max_class <= 0;
            predicted_digit <= 0;
            inference_done <= 0;
            busy <= 0;
            weight_addr <= 0;
            bias_addr <= 0;
            input_addr <= 0;
            weight_reg <= 0;
            pixel_reg <= 0;
            product <= 0;
            
            // Reset all class scores
            class_scores[0] <= 0;
            class_scores[1] <= 0;
            class_scores[2] <= 0;
            class_scores[3] <= 0;
            class_scores[4] <= 0;
            class_scores[5] <= 0;
            class_scores[6] <= 0;
            class_scores[7] <= 0;
            class_scores[8] <= 0;
            class_scores[9] <= 0;
            
            // Reset score outputs
            class_score_0 <= 0;
            class_score_1 <= 0;
            class_score_2 <= 0;
            class_score_3 <= 0;
            class_score_4 <= 0;
            class_score_5 <= 0;
            class_score_6 <= 0;
            class_score_7 <= 0;
            class_score_8 <= 0;
            class_score_9 <= 0;
        end else begin
```

#### Change 1.4: Store score in STATE_NEXT_CLASS

**Location**: In STATE_NEXT_CLASS, after line 213 (after `final_score = accumulator + current_bias;`)

**Find:**
```verilog
                STATE_NEXT_CLASS: begin : next_class_block
                    // Final score = accumulator (all 784 products: 0..783)
                    //             + bias
                    // Product was already accumulated in COMPARE state
                    reg signed [31:0] final_score;
                    final_score = accumulator + current_bias;
                    
                    // Update maximum
                    if (final_score > max_score) begin
```

**Replace with:**
```verilog
                STATE_NEXT_CLASS: begin : next_class_block
                    // Final score = accumulator (all 784 products: 0..783)
                    //             + bias
                    // Product was already accumulated in COMPARE state
                    reg signed [31:0] final_score;
                    final_score = accumulator + current_bias;
                    
                    // Store the score for this class
                    class_scores[current_class] <= final_score;
                    
                    // Update maximum
                    if (final_score > max_score) begin
```

#### Change 1.5: Output scores in STATE_DONE

**Location**: In STATE_DONE (around line 235)

**Find:**
```verilog
                STATE_DONE: begin
                    predicted_digit <= max_class;
                    inference_done <= 1;
                    busy <= 0;
                    state <= STATE_IDLE;
                end
```

**Replace with:**
```verilog
                STATE_DONE: begin
                    predicted_digit <= max_class;
                    inference_done <= 1;
                    busy <= 0;
                    
                    // Copy all scores to output ports
                    class_score_0 <= class_scores[0];
                    class_score_1 <= class_scores[1];
                    class_score_2 <= class_scores[2];
                    class_score_3 <= class_scores[3];
                    class_score_4 <= class_scores[4];
                    class_score_5 <= class_scores[5];
                    class_score_6 <= class_scores[6];
                    class_score_7 <= class_scores[7];
                    class_score_8 <= class_scores[8];
                    class_score_9 <= class_scores[9];
                    
                    state <= STATE_IDLE;
                end
```

---

### Step 2: Add `scores_ram` Module to `rams.v`

**File**: `regresja/inference/rtl/rams.v`

**Location**: At the end of the file, after the `predicted_digit_ram` module (after line 70)

**Find:**
```verilog
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule


```

**Replace with:**
```verilog
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule


// =============================================================================
// Scores RAM - Stores all 10 class scores for accuracy analysis
// =============================================================================
// Memory Layout:
//   Address 0-3:   Class 0 score (32-bit signed, little-endian)
//   Address 4-7:   Class 1 score
//   Address 8-11:  Class 2 score
//   Address 12-15: Class 3 score
//   Address 16-19: Class 4 score
//   Address 20-23: Class 5 score
//   Address 24-27: Class 6 score
//   Address 28-31: Class 7 score
//   Address 32-35: Class 8 score
//   Address 36-39: Class 9 score
//
// Total: 40 bytes (10 scores × 4 bytes each)
//
// This BRAM stores all class scores when inference completes.
// Scores are written in little-endian format for easy reading via UART.
// =============================================================================
module scores_ram (
    input wire clk,
    input wire wr_en,                    // Write enable (pulse when inference_done)
    input wire signed [31:0] score_0,    // Class 0 score
    input wire signed [31:0] score_1,    // Class 1 score
    input wire signed [31:0] score_2,    // Class 2 score
    input wire signed [31:0] score_3,    // Class 3 score
    input wire signed [31:0] score_4,    // Class 4 score
    input wire signed [31:0] score_5,    // Class 5 score
    input wire signed [31:0] score_6,    // Class 6 score
    input wire signed [31:0] score_7,    // Class 7 score
    input wire signed [31:0] score_8,    // Class 8 score
    input wire signed [31:0] score_9,    // Class 9 score
    input wire [5:0] rd_addr,            // Read address (0-39)
    output reg [7:0] rd_data             // Read data (1 byte)
);

    // 40-byte BRAM (10 scores × 4 bytes each)
    (* ram_style = "block" *) reg [7:0] ram [0:39];
    
    // Synchronous write - write all 10 scores at once (little-endian)
    always @(posedge clk) begin
        if (wr_en) begin
            // Class 0 (addresses 0-3)
            ram[0]  <= score_0[7:0];
            ram[1]  <= score_0[15:8];
            ram[2]  <= score_0[23:16];
            ram[3]  <= score_0[31:24];
            
            // Class 1 (addresses 4-7)
            ram[4]  <= score_1[7:0];
            ram[5]  <= score_1[15:8];
            ram[6]  <= score_1[23:16];
            ram[7]  <= score_1[31:24];
            
            // Class 2 (addresses 8-11)
            ram[8]  <= score_2[7:0];
            ram[9]  <= score_2[15:8];
            ram[10] <= score_2[23:16];
            ram[11] <= score_2[31:24];
            
            // Class 3 (addresses 12-15)
            ram[12] <= score_3[7:0];
            ram[13] <= score_3[15:8];
            ram[14] <= score_3[23:16];
            ram[15] <= score_3[31:24];
            
            // Class 4 (addresses 16-19)
            ram[16] <= score_4[7:0];
            ram[17] <= score_4[15:8];
            ram[18] <= score_4[23:16];
            ram[19] <= score_4[31:24];
            
            // Class 5 (addresses 20-23)
            ram[20] <= score_5[7:0];
            ram[21] <= score_5[15:8];
            ram[22] <= score_5[23:16];
            ram[23] <= score_5[31:24];
            
            // Class 6 (addresses 24-27)
            ram[24] <= score_6[7:0];
            ram[25] <= score_6[15:8];
            ram[26] <= score_6[23:16];
            ram[27] <= score_6[31:24];
            
            // Class 7 (addresses 28-31)
            ram[28] <= score_7[7:0];
            ram[29] <= score_7[15:8];
            ram[30] <= score_7[23:16];
            ram[31] <= score_7[31:24];
            
            // Class 8 (addresses 32-35)
            ram[32] <= score_8[7:0];
            ram[33] <= score_8[15:8];
            ram[34] <= score_8[23:16];
            ram[35] <= score_8[31:24];
            
            // Class 9 (addresses 36-39)
            ram[36] <= score_9[7:0];
            ram[37] <= score_9[15:8];
            ram[38] <= score_9[23:16];
            ram[39] <= score_9[31:24];
        end
    end
    
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule


```

---

### Step 3: Create `scores_reader.v` - UART Protocol Handler

**File**: `regresja/inference/rtl/scores_reader.v` (NEW FILE)

**Full contents:**

```verilog
/*
================================================================================
Scores Reader - Handles UART read requests for all class scores
================================================================================
Protocol:
  Request: Send byte 0xCD to request all class scores
  Response: FPGA sends 40 bytes containing all 10 scores (little-endian)
  
Memory Layout:
  Bytes 0-3:   Class 0 score (int32, little-endian)
  Bytes 4-7:   Class 1 score
  Bytes 8-11:  Class 2 score
  Bytes 12-15: Class 3 score
  Bytes 16-19: Class 4 score
  Bytes 20-23: Class 5 score
  Bytes 24-27: Class 6 score
  Bytes 28-31: Class 7 score
  Bytes 32-35: Class 8 score
  Bytes 36-39: Class 9 score
================================================================================
*/

module scores_reader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,        // UART RX data
    input wire rx_ready,              // UART RX ready signal
    input wire [7:0] scores_data,     // Read data from scores_ram
    output reg [5:0] scores_addr,     // Address to read from scores_ram (0-39)
    output reg [7:0] tx_data,         // Data to send via UART TX
    output reg tx_send,               // Pulse to start UART TX transmission
    input wire tx_busy                // UART TX busy signal
);

    // Request byte constant
    localparam REQUEST_BYTE = 8'hCD;
    localparam NUM_BYTES = 40;  // 10 scores × 4 bytes each
    
    // States
    localparam STATE_IDLE = 2'd0;
    localparam STATE_READ_BYTE = 2'd1;
    localparam STATE_SEND_BYTE = 2'd2;
    localparam STATE_WAIT_TX = 2'd3;
    
    reg [1:0] state;
    reg [5:0] byte_counter;  // 0 to 39
    reg rx_ready_prev;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            tx_data <= 0;
            tx_send <= 0;
            rx_ready_prev <= 0;
            byte_counter <= 0;
            scores_addr <= 0;
        end else begin
            // Default: tx_send is pulse
            tx_send <= 0;
            
            // Detect rising edge of rx_ready
            rx_ready_prev <= rx_ready;
            
            case (state)
                // ----------------------------------------
                // IDLE: Wait for read request (0xCD)
                // ----------------------------------------
                STATE_IDLE: begin
                    if (rx_ready && !rx_ready_prev) begin
                        // New byte received
                        if (rx_data == REQUEST_BYTE) begin
                            // Valid request - start reading scores
                            byte_counter <= 0;
                            scores_addr <= 0;
                            state <= STATE_READ_BYTE;
                        end
                    end
                end
                
                // ----------------------------------------
                // READ_BYTE: Request byte from scores_ram
                // ----------------------------------------
                STATE_READ_BYTE: begin
                    // Wait 1 cycle for BRAM read (synchronous read)
                    state <= STATE_SEND_BYTE;
                end
                
                // ----------------------------------------
                // SEND_BYTE: Send the byte via UART TX
                // ----------------------------------------
                STATE_SEND_BYTE: begin
                    // Data from scores_ram is now available
                    tx_data <= scores_data;
                    
                    if (!tx_busy) begin
                        // UART TX is idle, send the data
                        tx_send <= 1;
                        state <= STATE_WAIT_TX;
                    end
                    // If TX is busy, wait in this state
                end
                
                // ----------------------------------------
                // WAIT_TX: Wait for transmission to complete
                // ----------------------------------------
                STATE_WAIT_TX: begin
                    if (!tx_busy) begin
                        // Transmission complete
                        if (byte_counter < NUM_BYTES - 1) begin
                            // More bytes to send
                            byte_counter <= byte_counter + 1;
                            scores_addr <= byte_counter + 1;
                            state <= STATE_READ_BYTE;
                        end else begin
                            // All 40 bytes sent
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
```

---

### Step 4: Update `top.v` - Wire Everything Together

**File**: `regresja/inference/rtl/top.v`

#### Change 4.1: Add class score signals

**Location**: After line 44 (after `inference_busy` declaration)

**Find:**
```verilog
    // Inference signals
    wire [12:0] inf_weight_addr;
    wire [3:0]  inf_bias_addr;
    wire [9:0]  inf_input_addr;
    wire [7:0]  inf_input_pixel;
    wire [3:0]  predicted_digit;
    wire        inference_done;
    wire        inference_busy;
```

**Replace with:**
```verilog
    // Inference signals
    wire [12:0] inf_weight_addr;
    wire [3:0]  inf_bias_addr;
    wire [9:0]  inf_input_addr;
    wire [7:0]  inf_input_pixel;
    wire [3:0]  predicted_digit;
    wire        inference_done;
    wire        inference_busy;
    
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
```

#### Change 4.2: Add scores RAM and TX arbiter signals

**Location**: After line 57 (after `digit_ram_rd_data` declaration)

**Find:**
```verilog
    // Predicted digit RAM signals
    wire digit_ram_wr_en;
    wire [7:0] digit_ram_rd_data;
    
    // UART TX signals for digit reader
    wire [7:0] tx_data;
    wire tx_send;
    wire tx_busy;
    wire tx_out;
    
    // UART RX signals for digit reader
    wire [7:0] digit_rx_data;
    wire digit_rx_ready;
```

**Replace with:**
```verilog
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
    
    // UART RX signals - shared between digit and scores readers
    wire [7:0] digit_rx_data;
    wire digit_rx_ready;
```

#### Change 4.3: Connect class scores to inference module

**Location**: In u_inference instantiation (around line 128-142)

**Find:**
```verilog
    inference u_inference (
        .clk(clk),
        .rst(rst),
        .weight_addr(inf_weight_addr),
        .weight_data(weight_rd_data),
        .bias_addr(inf_bias_addr),
        .bias_data(bias_rd_data),
        .weights_ready(weights_loaded),
        .start_inference(start_inference_pulse),
        .input_pixel(inf_input_pixel),
        .input_addr(inf_input_addr),
        .predicted_digit(predicted_digit),
        .inference_done(inference_done),
        .busy(inference_busy)
    );
```

**Replace with:**
```verilog
    inference u_inference (
        .clk(clk),
        .rst(rst),
        .weight_addr(inf_weight_addr),
        .weight_data(weight_rd_data),
        .bias_addr(inf_bias_addr),
        .bias_data(bias_rd_data),
        .weights_ready(weights_loaded),
        .start_inference(start_inference_pulse),
        .input_pixel(inf_input_pixel),
        .input_addr(inf_input_addr),
        .predicted_digit(predicted_digit),
        .inference_done(inference_done),
        .busy(inference_busy),
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
```

#### Change 4.4: Add scores_ram instantiation

**Location**: After predicted_digit_ram instantiation (after line 178)

**Find:**
```verilog
    assign digit_ram_wr_en = inference_done && !inference_done_prev;
    
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
```

**Replace with:**
```verilog
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
```

#### Change 4.5: Add scores_reader and TX arbiter

**Location**: After digit_reader instantiation (after line 221)

**Find:**
```verilog
    // =========================================================================
    // Digit Reader
    // =========================================================================
    digit_reader u_digit_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(digit_rx_data),
        .rx_ready(digit_rx_ready),
        .digit_data(digit_ram_rd_data),
        .tx_data(tx_data),
        .tx_send(tx_send),
        .tx_busy(tx_busy)
    );
```

**Replace with:**
```verilog
    // =========================================================================
    // Digit Reader (0xCC protocol)
    // =========================================================================
    digit_reader u_digit_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(digit_rx_data),
        .rx_ready(digit_rx_ready),
        .digit_data(digit_ram_rd_data),
        .tx_data(digit_tx_data),
        .tx_send(digit_tx_send),
        .tx_busy(tx_busy)
    );
    
    // =========================================================================
    // Scores Reader (0xCD protocol)
    // =========================================================================
    scores_reader u_scores_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(digit_rx_data),
        .rx_ready(digit_rx_ready),
        .scores_data(scores_ram_rd_data),
        .scores_addr(scores_ram_rd_addr),
        .tx_data(scores_tx_data),
        .tx_send(scores_tx_send),
        .tx_busy(tx_busy)
    );
    
    // =========================================================================
    // TX Arbiter - Multiplex between digit_reader and scores_reader
    // =========================================================================
    // Simple OR-based arbiter (only one will be active at a time due to
    // different request bytes 0xCC vs 0xCD)
    assign tx_data = digit_tx_send ? digit_tx_data : scores_tx_data;
    assign tx_send = digit_tx_send | scores_tx_send;
```

---

### Step 5: Update Python Test Script

**File**: `regresja/testing/test_fpga_integration.py`

#### Change 5.1: Add struct import

**Location**: Import section (line 1-7)

**Find:**
```python
import serial
import time
import sys
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix
```

**Replace with:**
```python
import serial
import time
import sys
import os
import struct
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix
```

#### Change 5.2: Add scores request constant

**Location**: Protocol constants section (line 15-18)

**Find:**
```python
# Protocol Constants
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])
DIGIT_READ_REQUEST = bytes([0xCC])
```

**Replace with:**
```python
# Protocol Constants
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])
DIGIT_READ_REQUEST = bytes([0xCC])
SCORES_READ_REQUEST = bytes([0xCD])
```

#### Change 5.3: Add score reading function

**Location**: After preprocess_image function, before run_integration_test (around line 60)

**Add this new function:**
```python
def read_scores_from_fpga(ser):
    """Read all 10 class scores from FPGA via 0xCD protocol.
    
    Returns:
        numpy array of 10 int32 scores, or None on error
    """
    ser.reset_input_buffer()
    ser.write(SCORES_READ_REQUEST)
    
    # Read 40 bytes (10 scores × 4 bytes each)
    resp = ser.read(40)
    
    if len(resp) != 40:
        print(f"Error: Expected 40 bytes, got {len(resp)}")
        return None
    
    # Unpack as 10 signed 32-bit integers (little-endian)
    scores = struct.unpack('<10i', resp)
    return np.array(scores, dtype=np.int32)
```

#### Change 5.4: Add detailed test function

**Location**: After run_integration_test function (around line 124)

**Add this new function:**
```python
def run_detailed_test_with_scores(num_samples=10):
    """Run a detailed test that reads and displays class scores for analysis.
    
    Args:
        num_samples: Number of images to test with detailed score output
    """
    X_test, y_test, mean, scale = load_data_and_scaler()
    
    indices = range(num_samples)
    
    print(f"\nStarting Detailed Test with Scores on {num_samples} images...")
    print(f"Port: {COM_PORT}, Baud: {BAUD_RATE}")
    print("=" * 80)
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(1)
        ser.reset_input_buffer()
        
        for i, idx in enumerate(indices):
            target_label = y_test[idx]
            img_data = preprocess_image(X_test[idx], mean, scale)
            
            # Send Image
            ser.write(IMG_START_MARKER)
            ser.write(img_data.tobytes())
            ser.write(IMG_END_MARKER)
            ser.flush()
            
            time.sleep(0.05)
            
            # Read predicted digit
            ser.reset_input_buffer()
            ser.write(DIGIT_READ_REQUEST)
            resp = ser.read(1)
            
            if len(resp) != 1:
                print(f"Error: Timeout on image {i}")
                continue
                
            pred = int.from_bytes(resp, byteorder='little') & 0x0F
            
            # Read all class scores
            time.sleep(0.01)  # Small delay between requests
            scores = read_scores_from_fpga(ser)
            
            if scores is None:
                print(f"Error: Failed to read scores for image {i}")
                continue
            
            # Display results
            print(f"\nImage {i} (Index {idx}):")
            print(f"  True Label:      {target_label}")
            print(f"  Predicted:       {pred}")
            print(f"  Status:          {'✓ CORRECT' if pred == target_label else '✗ WRONG'}")
            print(f"  Class Scores:")
            
            # Show scores with highlighting
            for class_idx in range(10):
                score = scores[class_idx]
                marker = " <-- PREDICTED" if class_idx == pred else ""
                marker += " (TRUE)" if class_idx == target_label else ""
                print(f"    Class {class_idx}: {score:12d}{marker}")
            
            # Show score differences
            max_score = np.max(scores)
            second_max = np.partition(scores, -2)[-2]
            confidence = max_score - second_max
            print(f"  Confidence Margin: {confidence} (max - 2nd_max)")
            print("-" * 80)
        
        ser.close()
        
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
```

#### Change 5.5: Add CLI argument parser

**Location**: In main block (around line 126-127)

**Find:**
```python
if __name__ == "__main__":
    run_integration_test()
```

**Replace with:**
```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FPGA inference with MNIST')
    parser.add_argument('--mode', choices=['quick', 'detailed'], default='quick',
                        help='Test mode: quick (1000 images, accuracy only) or detailed (10 images with scores)')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples for detailed mode (default: 10)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_integration_test()
    else:
        run_detailed_test_with_scores(num_samples=args.samples)
```

---

## Testing and Verification

### 1. Verilog Linting
After making all changes, verify no syntax errors:
```bash
# Check all modified Verilog files
# Should show "No linter errors found" for each
```

### 2. Synthesis Check
Synthesize the design in Vivado/Quartus to ensure:
- All modules instantiate correctly
- No unconnected ports
- BRAM inference is correct (should use block RAM, not distributed)

### 3. Python Test
```bash
# Quick test (existing functionality should still work)
python regresja/testing/test_fpga_integration.py --mode quick

# Detailed test with scores
python regresja/testing/test_fpga_integration.py --mode detailed --samples 5
```

### 4. Expected Behavior
- 0xCC request → 1 byte response (predicted digit)
- 0xCD request → 40 bytes response (all scores)
- Scores should be signed 32-bit integers
- argmax(scores) should equal predicted_digit

---

## Key Design Principles

1. **Separation of Concerns**: Scores RAM is separate from predicted digit RAM
2. **Little-Endian Format**: Standard for cross-platform compatibility
3. **Synchronous Design**: All BRAM operations are synchronous
4. **TX Arbitration**: Simple OR-based (safe because protocols are mutually exclusive)
5. **Pipeline Awareness**: scores_reader accounts for 1-cycle BRAM read latency

---

## Protocol Summary

| Command | Request | Response | Purpose |
|---------|---------|----------|---------|
| Read Digit | 0xCC | 1 byte | Get predicted digit (0-9) |
| Read Scores | 0xCD | 40 bytes | Get all 10 class scores (int32, little-endian) |

---

## Resource Usage

- **New BRAM**: 40 bytes (scores_ram)
- **New Logic**: ~200 LUTs (scores_reader state machine)
- **New Registers**: 10 × 32-bit (class score outputs) + state machine registers

---

## Notes for Future Implementation

1. **Order matters**: Follow steps 1-5 in sequence
2. **Test incrementally**: After each step, check for syntax errors
3. **Backup first**: These changes modify core inference and top modules
4. **UART timing**: 0.01s delay between 0xCC and 0xCD requests is recommended
5. **Little-endian**: Python's `struct.unpack('<10i', ...)` expects little-endian

---

## Common Issues and Solutions

### Issue: Synthesis fails with "port not found"
**Solution**: Ensure all new ports in inference module are connected in top.v

### Issue: FPGA returns wrong number of bytes for 0xCD
**Solution**: Check scores_reader state machine byte_counter logic

### Issue: Scores are all zeros
**Solution**: Verify scores_ram_wr_en is pulsing when inference_done fires

### Issue: TX conflicts between digit and scores
**Solution**: Ensure 0xCC and 0xCD requests are not sent simultaneously

---

## Files Modified Summary

1. **regresja/inference/rtl/inference.v** - Added score outputs and storage
2. **regresja/inference/rtl/rams.v** - Added scores_ram module
3. **regresja/inference/rtl/scores_reader.v** - Created new UART handler
4. **regresja/inference/rtl/top.v** - Wired all modules together
5. **regresja/testing/test_fpga_integration.py** - Added score reading capability

Total: 4 modified files, 1 new file

