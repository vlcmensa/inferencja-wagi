# Inference Module Deep Dive

A comprehensive line-by-line explanation of `inference.v` - the core computation module for MNIST digit classification using softmax regression on FPGA.

---

## Table of Contents
1. [Overview](#overview)
2. [Module Interface](#module-interface)
3. [Architecture & Data Flow](#architecture--data-flow)
4. [State Machine](#state-machine)
5. [Pipeline Operation](#pipeline-operation)
6. [Line-by-Line Walkthrough](#line-by-line-walkthrough)
7. [Timing Diagrams](#timing-diagrams)

---

## Overview

### Purpose
The `inference` module implements the core computation for softmax regression (logistic regression) inference on MNIST handwritten digits. It computes:

```
For each class i (0-9):
    score[i] = Σ(input[j] × weight[i][j]) + bias[i]
             j=0..783

prediction = argmax(score[0..9])
```

### Key Characteristics
- **Pure Computation Module**: No UART, no I/O - only math
- **Sequential Processing**: One multiply-accumulate (MAC) per clock cycle
- **Pipelined**: 3-stage pipeline for timing optimization
- **Performance**: 7840 cycles per inference (~78.4 µs at 100 MHz)
- **Data Types**:
  - Inputs: 8-bit signed (-128 to 127)
  - Weights: 8-bit signed (-128 to 127)
  - Products: 16-bit signed
  - Accumulator: 32-bit signed
  - Biases: 32-bit signed

---

## Module Interface

### Ports (Lines 37-59)

```verilog
module inference (
    input wire clk,              // System clock (100 MHz)
    input wire rst,              // Synchronous reset (active high)
    
    // Weight memory interface (Read-only)
    output reg [12:0] weight_addr,    // Address: 0 to 7839 (13 bits)
    input wire [7:0]  weight_data,    // Data: 8-bit signed weight
    
    // Bias memory interface (Read-only)
    output reg [3:0]  bias_addr,      // Address: 0 to 9 (4 bits)
    input wire [31:0] bias_data,      // Data: 32-bit signed bias
    
    // Control signals
    input wire        weights_ready,   // HIGH when weights loaded
    input wire        start_inference, // Pulse HIGH to start
    
    // Image memory interface (Read-only)
    input wire [7:0]  input_pixel,    // Current pixel value
    output reg [9:0]  input_addr,     // Address: 0 to 783 (10 bits)
    
    // Outputs
    output reg [3:0]  predicted_digit, // Result: 0-9
    output reg        inference_done,  // Pulse HIGH when done
    output reg        busy             // HIGH during inference
);
```

**Key Points:**
- All memory interfaces are **synchronous reads** (1 cycle latency)
- Module **requests** addresses, memories **provide** data
- `start_inference` is a **pulse** (one clock cycle HIGH)
- `inference_done` is a **pulse** (one clock cycle HIGH)
- `busy` is a **level** signal (HIGH throughout inference)

---

## Architecture & Data Flow

### Memory Organization

**Weight Memory (7840 bytes):**
```
Address = class_index × 784 + pixel_index
Example: Weight for class 5, pixel 100 → Address 5×784+100 = 4020
```

**Bias Memory (10 entries):**
```
Address = class_index (0-9)
Example: Bias for class 7 → Address 7
```

**Image Memory (784 bytes):**
```
Address = pixel_index (0-783)
```

### Computation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    OUTER LOOP (10 classes)                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              INNER LOOP (784 pixels)                   │  │
│  │                                                         │  │
│  │  1. Load weight[class][pixel]     ← weight_data       │  │
│  │  2. Load input[pixel]             ← input_pixel       │  │
│  │  3. Multiply: product = w × i     [Pipeline Stage]    │  │
│  │  4. Accumulate: acc += product    [Pipeline Stage]    │  │
│  │                                                         │  │
│  │  Repeat 784 times ──────────────────────────────────► │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  5. Add bias: score = accumulator + bias[class]             │
│  6. Compare: if score > max_score, update winner            │
│                                                               │
│  Repeat for all 10 classes ────────────────────────────────►│
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Return predicted_digit = argmax class
```

---

## State Machine

### State Definitions (Lines 62-68)

```verilog
localparam STATE_IDLE           = 3'd0;  // Waiting for start
localparam STATE_LOAD_BIAS      = 3'd1;  // Load bias for current class
localparam STATE_COMPUTE        = 3'd2;  // Main MAC loop (784 cycles)
localparam STATE_ADD_BIAS       = 3'd3;  // Pipeline flush cycle 1
localparam STATE_COMPARE        = 3'd4;  // Pipeline flush cycle 2
localparam STATE_NEXT_CLASS     = 3'd5;  // Compute final score, update max
localparam STATE_DONE           = 3'd6;  // Output result
```

### State Transition Diagram

```
         ┌──────────┐
         │   IDLE   │ ← Start here
         └──────────┘
              │ start_inference=1
              ▼
    ┌─────────────────────┐
    │    LOAD_BIAS        │ ← Load bias[class]
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │     COMPUTE         │ ← 784 MACs
    │   (784 cycles)      │
    └─────────────────────┘
              │ current_pixel = 783
              ▼
    ┌─────────────────────┐
    │    ADD_BIAS         │ ← Pipeline flush cycle 1
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │    COMPARE          │ ← Pipeline flush cycle 2
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   NEXT_CLASS        │ ← Final score, update max
    └─────────────────────┘
              │
              ├─► current_class < 9? ──YES──► Loop to LOAD_BIAS
              │
              NO (all 10 classes done)
              ▼
    ┌─────────────────────┐
    │      DONE           │ ← Output result
    └─────────────────────┘
              │
              ▼ Back to IDLE
```

---

## Pipeline Operation

### The 3-Stage Pipeline

The module uses a **3-stage pipeline** to improve timing and clock frequency:

```
Clock Cycle:     N         N+1        N+2        N+3
               ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
Stage 1:       │ R0  │   │ R1  │   │ R2  │   │ R3  │  Register inputs
               └─────┘   └─────┘   └─────┘   └─────┘
                  │         │         │         │
Stage 2:       ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
               │     │   │ M0  │   │ M1  │   │ M2  │  Multiply
               └─────┘   └─────┘   └─────┘   └─────┘
                             │         │         │
Stage 3:                  ┌─────┐   ┌─────┐   ┌─────┐
                          │     │   │ A0  │   │ A1  │  Accumulate
                          └─────┘   └─────┘   └─────┘

Legend:
  R = Register (load weight_reg, pixel_reg)
  M = Multiply (compute product = weight_reg × pixel_reg)
  A = Accumulate (acc += product)
```

**Key Insight:** At any given cycle, **three different operations** are happening:
- Loading the next weight/pixel pair
- Multiplying the previous pair
- Accumulating the product from 2 cycles ago

### Pipeline Latency Problem

When you finish loading the 784th pixel (pixel 783), **there are still 2 products in the pipeline**:
- Product 782 is being multiplied
- Product 783 hasn't been multiplied yet

**Solution:** Two extra "flush" cycles (ADD_BIAS and COMPARE states) to complete the last 2 products without loading new data.

---

## Line-by-Line Walkthrough

### Header Comments (Lines 1-35)

The header provides critical context:
- **Model description**: Softmax regression for MNIST
- **Computation formula**: score[i] = Σ(input × weight) + bias
- **Data types**: All signed integers with specific bit widths
- **Architecture**: Sequential processing, one MAC per cycle
- **Performance**: 7840 cycles total (~78.4 µs at 100 MHz)

---

### Register Declarations (Lines 70-82)

```verilog
reg [2:0] state;                       // Current FSM state (0-6)
reg [3:0] current_class;               // Current class (0-9)
reg [9:0] current_pixel;               // Current pixel (0-783)
reg signed [31:0] accumulator;         // Running sum (32-bit signed)
reg signed [31:0] current_bias;        // Bias for current class
reg signed [31:0] max_score;           // Highest score so far
reg [3:0] max_class;                   // Class with highest score

// Pipeline registers
reg signed [7:0] weight_reg;           // Registered weight (Stage 1)
reg signed [7:0] pixel_reg;            // Registered pixel (Stage 1)
reg signed [15:0] product;             // Multiply result (Stage 2)
```

**Important:**
- `signed` keyword is crucial - tells Verilog to treat as two's complement
- Pipeline registers hold intermediate values between stages

---

### Constants (Lines 84-86)

```verilog
localparam NUM_PIXELS = 784;   // 28×28 image
localparam NUM_CLASSES = 10;   // Digits 0-9
```

These make the code more readable and easier to modify.

---

## State-by-State Detailed Explanation

---

### STATE 0: IDLE (Lines 116-130)

**Purpose:** Wait for the start signal.

**What happens:**
1. Set `busy = 0` to indicate module is idle
2. Wait for `start_inference` pulse AND `weights_ready` to be HIGH
3. When both conditions met:
   - Transition to `STATE_LOAD_BIAS`
   - Initialize all counters to 0
   - Initialize `max_score` to minimum 32-bit signed value (`0x80000000` = -2,147,483,648)
   - Set `busy = 1`
   - Request first bias by setting `bias_addr = 0`

```verilog
STATE_IDLE: begin
    busy <= 0;
    if (start_inference && weights_ready) begin
        state <= STATE_LOAD_BIAS;
        current_class <= 0;
        current_pixel <= 0;
        accumulator <= 0;
        max_score <= 32'h80000000;  // Most negative 32-bit value
        max_class <= 0;
        busy <= 1;
        bias_addr <= 0;              // Request bias for class 0
    end
end
```

**Why initialize max_score to minimum?**
- Ensures ANY real score will be larger
- First class will always become the initial "winner"

---

### STATE 1: LOAD_BIAS (Lines 132-152)

**Purpose:** Load the bias for the current class and prepare for MAC loop.

**What happens:**
1. **Capture bias**: `current_bias <= $signed(bias_data)`
   - The bias data arrives from memory (requested in previous state)
   - `$signed()` ensures proper sign interpretation

2. **Reset accumulator**: `accumulator <= 0`
   - Start fresh for this class

3. **CRITICAL: Reset pipeline registers:**
   ```verilog
   weight_reg <= 0;
   pixel_reg <= 0;
   product <= 0;
   ```
   - **Why?** Prevents garbage from previous class leaking into current class
   - Without this, the first product of the new class would be contaminated

4. **Request first weight and pixel:**
   ```verilog
   weight_addr <= current_class * NUM_PIXELS;  // First weight for this class
   input_addr <= 0;                             // First pixel
   ```

5. Transition to `STATE_COMPUTE`

**Memory Latency:**
- Weight and pixel data will arrive **1 cycle later** (in COMPUTE state)
- This is fine because pipeline needs time to start up

**Example:**
- For class 3, `weight_addr = 3 × 784 = 2352`
- This points to the first weight for class 3

---

### STATE 2: COMPUTE (Lines 154-180)

**Purpose:** Main multiply-accumulate loop. Runs for exactly 784 cycles.

**What happens each cycle:**

#### Stage 1: Register Inputs (Lines 159-160)
```verilog
weight_reg <= $signed(weight_data);
pixel_reg <= input_pixel;
```
- Load the current weight and pixel from memory
- `$signed()` ensures proper two's complement interpretation

#### Stage 2: Multiply (Lines 162-166)
```verilog
product <= $signed(weight_reg) * $signed(pixel_reg);
```
- Multiply the **previous cycle's** weight and pixel
- Result is 16-bit signed (8-bit × 8-bit = 16-bit max)

#### Stage 3: Accumulate (Line 169)
```verilog
accumulator <= accumulator + {{16{product[15]}}, product};
```
- Add the **two cycles ago** product to accumulator
- `{{16{product[15]}}, product}` = **sign extension** to 32 bits
  - Takes bit 15 (sign bit) and replicates it 16 times
  - Example: `16'hFF80` (negative) → `32'hFFFFFF80`
  - Example: `16'h007F` (positive) → `32'h0000007F`

#### Address Management (Lines 172-179)
```verilog
if (current_pixel < NUM_PIXELS - 1) begin
    current_pixel <= current_pixel + 1;
    weight_addr <= weight_addr + 1;
    input_addr <= current_pixel + 1;
end else begin
    // Done with 784 pixels
    state <= STATE_ADD_BIAS;
end
```

**Important:** When `current_pixel = 783` (last pixel):
- We DON'T increment addresses anymore
- We transition to ADD_BIAS
- But pipeline still has 2 products to finish!

**Timeline Example:**
```
Cycle   current_pixel   Action
-----   -------------   ------
  1          0          Load w[0], p[0]
  2          1          Load w[1], p[1],  Multiply w[0]×p[0]
  3          2          Load w[2], p[2],  Multiply w[1]×p[1],  Acc += w[0]×p[0]
  ...
 783        782          Load w[782], p[782]
 784        783          Load w[783], p[783]  ← LAST LOAD
                         → Transition to ADD_BIAS
```

At cycle 784, we've loaded all 784 pairs, but:
- Product 782 hasn't been accumulated yet
- Product 783 hasn't been multiplied yet

---

### STATE 3: ADD_BIAS (Lines 182-192)

**Purpose:** Pipeline flush cycle 1 - multiply the last pair and accumulate the second-to-last product.

**What happens:**

```verilog
STATE_ADD_BIAS: begin
    // Continue pipeline operations WITHOUT loading new data
    product <= $signed(weight_reg) * $signed(pixel_reg);
    accumulator <= accumulator + {{16{product[15]}}, product};
    
    state <= STATE_COMPARE;
end
```

**Critical point:** We DON'T load new data:
- `weight_reg` and `pixel_reg` still contain the 783rd weight/pixel pair
- We multiply them (computing product 783)
- We accumulate the existing product (product 782)

**Timeline continues:**
```
Cycle   State          Action
-----   -----          ------
 785    ADD_BIAS       Multiply w[783]×p[783],  Acc += w[782]×p[782]
```

**Why the name "ADD_BIAS"?**
- Historical - originally this was where bias was added
- Now it's just a pipeline flush cycle
- The actual bias addition happens in NEXT_CLASS

---

### STATE 4: COMPARE (Lines 194-203)

**Purpose:** Pipeline flush cycle 2 - accumulate the final (784th) product.

**What happens:**

```verilog
STATE_COMPARE: begin
    // Accumulate the product computed in ADD_BIAS state
    // This is the final (784th) product
    accumulator <= accumulator + {{16{product[15]}}, product};
    
    state <= STATE_NEXT_CLASS;
end
```

**No new multiplications** - only accumulation of the last product.

**Timeline continues:**
```
Cycle   State          Action
-----   -----          ------
 786    COMPARE        Acc += w[783]×p[783]  ← FINAL ACCUMULATION
```

**After this cycle:**
- `accumulator` contains the sum of ALL 784 products
- Ready to add bias and compare with max

**Why the name "COMPARE"?**
- Historical - originally comparison happened here
- Now comparison happens in NEXT_CLASS

---

### STATE 5: NEXT_CLASS (Lines 205-230)

**Purpose:** Compute final score, update maximum, and decide next action.

**What happens:**

#### 1. Compute Final Score (Lines 212-213)
```verilog
reg signed [31:0] final_score;
final_score = accumulator + current_bias;
```
- Note: This is a **combinational** variable (not registered)
- Computes within the same cycle

#### 2. Update Maximum (Lines 216-219)
```verilog
if (final_score > max_score) begin
    max_score <= final_score;
    max_class <= current_class;
end
```
- **Argmax implementation**: Keep track of highest score seen
- If current class beats the record, update both score and class

#### 3. Loop Control (Lines 221-229)
```verilog
if (current_class < NUM_CLASSES - 1) begin
    // More classes to process
    current_class <= current_class + 1;
    bias_addr <= current_class + 1;      // Request next bias
    state <= STATE_LOAD_BIAS;            // Loop back
end else begin
    // All 10 classes done
    state <= STATE_DONE;
end
```

**Loop structure:**
- Classes 0-8: Loop back to LOAD_BIAS for next class
- Class 9: All done, go to DONE

**Timeline for one complete class:**
```
Cycle   State           current_class
-----   -----           -------------
  1     LOAD_BIAS              0
  2     COMPUTE (1)            0
  ...
 785    COMPUTE (784)          0
 786    ADD_BIAS               0
 787    COMPARE                0
 788    NEXT_CLASS             0  ← Check: 0 < 9? Yes, loop
```

Then repeat for classes 1-9.

---

### STATE 6: DONE (Lines 232-240)

**Purpose:** Output the final result and return to IDLE.

**What happens:**

```verilog
STATE_DONE: begin
    predicted_digit <= max_class;     // Output winner (0-9)
    inference_done <= 1;               // Pulse HIGH for 1 cycle
    busy <= 0;                         // No longer busy
    state <= STATE_IDLE;               // Go back to waiting
end
```

**Key points:**
- `predicted_digit` gets latched to the argmax class
- `inference_done` is a **pulse** - HIGH for only 1 cycle
- Automatically returns to IDLE (no waiting)

**External modules can:**
- Read `predicted_digit` anytime after `inference_done` pulse
- Value remains stable until next inference starts

---

## Reset Behavior (Lines 90-106)

**Synchronous reset** (active HIGH):

```verilog
if (rst) begin
    state <= STATE_IDLE;
    current_class <= 0;
    current_pixel <= 0;
    accumulator <= 0;
    current_bias <= 0;
    max_score <= 32'h80000000;
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
end
```

**Everything gets cleared** - module starts fresh.

---

## Timing Diagrams

### Overall Inference Timing

```
Time:    0  1  2  3  ...  788  789 ...  7879  7880  7881
        ┌──┬──┬──┬──┬─────┬────┬────┬──────┬─────┬─────┬──┐
clk     ┘  └──┘  └──┘     └────┘    └──────┘     └─────┘  └──

start   ──┐  ┌──────────────────────────────────────────────
          └──┘

busy    ─────┐                                          ┌────
             └──────────────────────────────────────────┘

state   IDLE│LB│Compute (784)│AB│CP│NC│LB│...│NC│DONE│IDLE
             └──  Class 0  ───────┘  │...│    └────┘
                                   Class 1-8    Class 9

done    ─────────────────────────────────────────────┐  ┌────
                                                      └──┘
```

### Single Class Processing (788 cycles)

```
Cycle:  1    2    3    4   ...  785   786   787   788
       ┌────┬────┬────┬────┬────┬─────┬─────┬─────┬────┐
State  │ LB │ C0 │ C1 │ C2 │... │C783 │ AB  │ CP  │ NC │
       └────┴────┴────┴────┴────┴─────┴─────┴─────┴────┘

Where:
  LB = LOAD_BIAS
  C0-C783 = COMPUTE cycles 0-783
  AB = ADD_BIAS
  CP = COMPARE
  NC = NEXT_CLASS
```

### Pipeline Detail (COMPUTE state)

```
Cycle:       N      N+1     N+2     N+3
           ┌──────┬───────┬───────┬───────┐
Load       │ W0,P0│ W1,P1 │ W2,P2 │ W3,P3 │
           └──────┴───────┴───────┴───────┘
                    │       │       │
Multiply            │ W0×P0 │ W1×P1 │ W2×P2 │
                    └───────┴───────┴───────┘
                              │       │
Accumulate                    │+W0×P0 │+W1×P1 │
                              └───────┴───────┘

Accumulator:  0      0      W0×P0   W0×P0    W0×P0
Value                              +W1×P1   +W1×P1
                                            +W2×P2
```

---

## Critical Design Decisions

### 1. Why 3-stage pipeline?

**Answer:** Timing closure.

Without pipeline, critical path would be:
```
Memory Read → Multiply → Sign Extend → 32-bit Add → Register
```

With pipeline:
```
Cycle 1: Memory Read → Register
Cycle 2: Multiply → Register
Cycle 3: Sign Extend + Add → Register
```

Each cycle has a shorter critical path → higher clock frequency possible.

---

### 2. Why reset pipeline registers in LOAD_BIAS?

**Answer:** Prevent data contamination between classes.

Without reset:
- `weight_reg` contains last weight from previous class
- `pixel_reg` contains last pixel
- `product` contains their product
- First accumulation of new class would add this garbage value!

With reset:
- All pipeline registers = 0
- First accumulation adds 0 × 0 = 0 (harmless)

---

### 3. Why two flush cycles?

**Answer:** Pipeline depth.

The pipeline is 3 stages deep, which means when you stop loading new data:
- Stage 1 (register) has the last pair
- Stage 2 (multiply) has the second-to-last pair
- Stage 3 (accumulate) has the third-to-last pair

Need 2 extra cycles to finish the last 2 pairs.

---

### 4. Why is accumulator 32-bit?

**Answer:** Prevent overflow.

Worst case accumulator value:
```
Max product: 127 × 127 = 16,129
Max accumulator: 16,129 × 784 = 12,645,136
```

This fits comfortably in 32-bit signed range:
- Maximum: +2,147,483,647
- Our max: +12,645,136 ✓

**But beware:** With real weights and biases, overflow CAN occur if not careful during training/quantization!

---

## Common Pitfalls & Bugs

### 1. Off-by-one MAC count

**Bug:** Only computing 783 or 785 products instead of 784.

**Cause:** 
- Not flushing pipeline correctly
- Loop counter logic error

**Fix:** See testbench `tb_pipeline_flush.v` which verifies exact count.

---

### 2. Pipeline contamination

**Bug:** Wrong predictions, especially on second inference.

**Cause:** Forgetting to reset pipeline registers in LOAD_BIAS.

**Fix:** Always reset `weight_reg`, `pixel_reg`, `product` to 0.

---

### 3. Sign extension errors

**Bug:** Products are treated as unsigned.

**Cause:** Forgetting `$signed()` or `{{16{product[15]}}, product}`.

**Result:** Negative products become huge positive numbers!

**Fix:** Always use proper sign extension for signed arithmetic.

---

### 4. Memory read timing

**Bug:** Using data before it's available.

**Cause:** Forgetting that memory has 1-cycle latency.

**Fix:** Request address in cycle N, use data in cycle N+1.

---

## Verification Strategy

The testbenches verify:

1. **Basic functionality** (`tb_inference.v`)
   - Simple test with known winner

2. **Pipeline correctness** (`tb_pipeline_flush.v`)
   - Exact MAC count (784 per class)
   - Register reset between classes

3. **Edge cases:**
   - **All zeros** (`tb_edge_case_zeros.v`)
   - **Max positive values** (`tb_edge_case_max_values.v`)
   - **Min negative values** (`tb_edge_case_min_values.v`)
   - **Mixed signs** (`tb_edge_case_mixed_signs.v`)

4. **Hardware integration** (`test_fpga_integration.py`)
   - Real FPGA with UART
   - 1000 test images
   - Accuracy measurement

---

## Performance Analysis

### Cycle Count Breakdown

| State | Cycles per Class | Total for 10 Classes |
|-------|------------------|---------------------|
| LOAD_BIAS | 1 | 10 |
| COMPUTE | 784 | 7,840 |
| ADD_BIAS | 1 | 10 |
| COMPARE | 1 | 10 |
| NEXT_CLASS | 1 | 10 |
| **Subtotal** | **788** | **7,880** |
| DONE | 1 | 1 |
| **Total** | - | **7,881** |

### Timing at Different Frequencies

| Clock Freq | Time per Inference | Images per Second |
|------------|-------------------|-------------------|
| 50 MHz | 157.6 µs | 6,345 |
| 100 MHz | 78.8 µs | 12,689 |
| 200 MHz | 39.4 µs | 25,380 |

**Bottleneck:** Not computation, but UART speed for loading images!

---

## Summary

The `inference.v` module is a **well-designed sequential processor** for softmax regression:

✅ **Clean separation** of concerns (pure computation)  
✅ **Pipelined** for timing optimization  
✅ **Systematic** state machine with clear transitions  
✅ **Careful** handling of signed arithmetic  
✅ **Proper** pipeline flushing  
✅ **Efficient** argmax implementation  

**Key Takeaway:** Hardware inference is fundamentally about:
1. Organizing memory access patterns
2. Managing pipeline hazards
3. Handling fixed-point arithmetic carefully
4. Verifying corner cases thoroughly

This module demonstrates all these principles effectively.




