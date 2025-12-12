# Test Documentation

This document explains what each test in the `regresja/testing` directory tests and how to use them.

---

## Overview

The testing infrastructure includes:

### Python Tests (`regresja/testing/`)
1. **`simulate_fpga_inference.py`** - Software simulation of FPGA hardware arithmetic
2. **`test_fpga_integration.py`** - Physical FPGA hardware integration test

### Verilog Testbenches (`regresja/inference/tb/`)
1. **`tb_inference.v`** - Basic inference module functionality test
2. **`tb_pipeline_flush.v`** - Pipeline flush and MAC operation count verification
3. **`tb_edge_case_zeros.v`** - Edge case: all zeros test
4. **`tb_edge_case_max_values.v`** - Edge case: maximum positive values test
5. **`tb_edge_case_min_values.v`** - Edge case: minimum negative values test
6. **`tb_edge_case_mixed_signs.v`** - Edge case: mixed positive/negative signs test

---

## 1. simulate_fpga_inference.py

### Purpose
This script simulates the **exact arithmetic operations** performed by the FPGA hardware in software, including 32-bit signed integer overflow behavior. It helps identify whether overflow in the accumulator is causing accuracy degradation.

### What It Tests
- **32-bit overflow behavior**: Simulates the wrapping that occurs when accumulator values exceed ±2,147,483,648
- **Exact FPGA arithmetic pipeline**: Replicates the hardware's multiply-accumulate operations
- **Full test set evaluation**: Runs inference on all 10,000 MNIST test images
- **Overflow frequency**: Tracks how often overflow occurs during inference

### Hardware Arithmetic Simulation
The script replicates these exact hardware operations from `inference.v`:
- Input pixels: 8-bit signed integers (-128 to 127)
- Weights: 8-bit signed integers (-128 to 127)
- Products: 16-bit signed (8-bit × 8-bit multiplication)
- Accumulator: 32-bit signed **WITH overflow** (wraps at ±2,147,483,648)
- Bias addition: 32-bit signed **WITH overflow**
- Final prediction: argmax(scores)

### Key Functions

#### `load_model_and_scaler()`
- Loads quantized 8-bit weights from `.mem` files
- Loads 32-bit biases from `.mem` files
- Loads scaler parameters (mean and scale) from `.npy` files

#### `preprocess_image_fpga()`
- Normalizes input images from 0-255 to 0-1
- Applies standard scaler transformation
- Quantizes to int8 range using `INPUT_SCALE = 127.0`
- Clips values to [-128, 127]

#### `fpga_inference_with_overflow()`
- Simulates the FPGA inference pipeline with 32-bit overflow
- For each of 10 classes:
  - Initializes 32-bit accumulator to 0
  - Performs 784 multiply-accumulate operations (one per pixel)
  - Detects overflow events (for debugging)
  - Adds bias with potential overflow
- Returns predicted digit, scores, and overflow flag

### Output
The test provides:
- **Final accuracy** on the test set
- **Overflow event count** and percentage
- **Confusion matrix** showing prediction patterns
- **Sample misclassifications** (first 10)
- **Interpretation** of results:
  - If accuracy < 70% with overflows → overflow is likely the problem
  - If accuracy is good despite overflows → overflow is not the main issue
  - If no overflows detected → issue must be elsewhere in the pipeline

### Usage
```bash
python simulate_fpga_inference.py
```

### Expected Results
This test helps diagnose if the accuracy difference between software (90%) and hardware (59.5%) is caused by 32-bit accumulator overflow.

---

## 2. test_fpga_integration.py

### Purpose
This script performs **physical hardware integration testing** by sending real MNIST images to the FPGA via UART and verifying the predictions returned by the hardware.

### What It Tests
- **End-to-end hardware inference**: Tests the complete FPGA implementation
- **UART communication protocol**: Verifies data transmission and reception
- **Real-time hardware accuracy**: Measures actual FPGA performance on test images
- **Hardware reliability**: Checks for timeouts and communication errors

### Communication Protocol
The script uses the following UART protocol:

1. **Image Upload**:
   - Send `IMG_START_MARKER` (0xBB, 0x66)
   - Send 784 bytes of image data (int8 as uint8)
   - Send `IMG_END_MARKER` (0x66, 0xBB)

2. **Inference Wait**:
   - Wait 50ms for hardware to complete inference

3. **Result Request**:
   - Send `DIGIT_READ_REQUEST` (0xCC)
   - Read 1 byte response containing predicted digit (0-9)

### Configuration
```python
COM_PORT = 'COM3'        # Serial port (change as needed)
BAUD_RATE = 115200       # Must match FPGA UART configuration
TEST_SAMPLES = 1000      # Number of images to test
INPUT_SCALE = 127.0      # Must match training quantization
```

### Key Functions

#### `load_data_and_scaler()`
- Loads scaler parameters from `.npy` files
- Loads MNIST test dataset (last 10,000 images)
- Returns test images, labels, mean, and scale

#### `preprocess_image()`
- Normalizes input images from 0-255 to 0-1
- Applies standard scaler transformation
- Quantizes to int8 range using `INPUT_SCALE = 127.0`
- Converts to uint8 view for UART transmission (preserves 2's complement bits)

#### `run_integration_test()`
Main test loop that:
- Opens serial connection to FPGA
- For each test image:
  - Preprocesses the image
  - Sends via UART protocol
  - Requests and reads prediction
  - Compares with ground truth label
- Calculates final accuracy and confusion matrix

### Output
The test provides:
- **Live status** for each image (PASS/FAIL)
- **Final accuracy** on tested samples
- **Confusion matrix** showing classification patterns
- **Error messages** for timeouts or communication failures

### Usage
```bash
python test_fpga_integration.py
```

**Note**: Make sure the FPGA is properly connected and the correct COM port is configured.

### Expected Results
- Should achieve similar accuracy to `simulate_fpga_inference.py`
- Any significant difference indicates additional hardware issues beyond arithmetic overflow
- Timeouts may indicate UART communication problems or hardware hangs

---

## Comparison Between Tests

| Aspect | simulate_fpga_inference.py | test_fpga_integration.py |
|--------|---------------------------|--------------------------|
| **Environment** | Pure Python simulation | Physical FPGA hardware |
| **Speed** | Fast (software) | Slower (UART + hardware) |
| **Purpose** | Diagnose overflow issues | Verify full system |
| **Dependencies** | None (offline) | Requires FPGA connection |
| **Test Size** | 10,000 images (full test set) | Configurable (default 1,000) |
| **Overflow Detection** | Yes (explicit tracking) | No (hardware only) |
| **Use Case** | Debugging arithmetic | Final validation |

---

## Diagnostic Workflow

1. **Run `simulate_fpga_inference.py` first**:
   - If accuracy matches hardware (~59.5%) → overflow is the problem
   - If accuracy is higher → issue is in hardware implementation

2. **Run `test_fpga_integration.py` second**:
   - Validates the complete hardware system
   - Checks UART communication reliability
   - Confirms real-world performance

3. **Compare results**:
   - If both tests show similar accuracy → arithmetic is correctly implemented
   - If integration test is worse → check UART, timing, or memory issues
   - If integration test is better → check simulation assumptions

---

## Troubleshooting

### simulate_fpga_inference.py
- **Error: Model files not found**: Ensure training has been completed and weights are exported
- **Error: MNIST download fails**: Check internet connection (sklearn downloads MNIST)

### test_fpga_integration.py
- **Serial Error**: Check COM_PORT configuration and cable connection
- **Timeouts**: Increase wait time or check FPGA power/programming
- **Wrong predictions**: Verify weights were loaded correctly on FPGA
- **Communication errors**: Check baud rate matches FPGA UART module

---

## Dependencies

Both tests require:
- `numpy` - Array operations and data handling
- `scikit-learn` - MNIST dataset and metrics

`test_fpga_integration.py` additionally requires:
- `pyserial` - UART communication with FPGA

Install dependencies:
```bash
pip install numpy scikit-learn pyserial
```

---

## Verilog Testbenches

The `regresja/inference/tb/` directory contains Verilog testbenches for hardware simulation and verification.

---

### Test 1: tb_inference.v - Basic Inference Test

#### Purpose
Verifies basic inference module functionality with simple mock data.

#### What It Tests
- Basic state machine operation
- Memory interface connections
- MAC (multiply-accumulate) operations
- Argmax selection

#### Test Setup
- **Weights**: Class 2 has positive weights (+10), others have negative (-10)
- **Inputs**: All pixels set to +10
- **Biases**: All set to 0
- **Expected Result**: Class 2 should be predicted

#### Usage
```bash
# Using Vivado or compatible simulator
xvlog tb_inference.v inference.v
xelab tb_inference
xsim tb_inference -runall
```

#### Expected Output
- Predicted digit: 2
- Test PASSED message

---

### Test 2: tb_pipeline_flush.v - Pipeline Verification Test

#### Purpose
Verifies that exactly 784 MAC operations are performed per class and that pipeline registers are properly reset between classes.

#### What It Tests
- **MAC operation count**: Ensures exactly 784 products computed per class (not 783 or 785)
- **Pipeline register reset**: Verifies `weight_reg`, `pixel_reg`, and `product` are cleared between classes
- **State transitions**: Checks state machine transitions at correct pixel counts
- **Data integrity**: Ensures no carryover between classes

#### Test Strategy
- Uses different weight/pixel patterns for each class to detect carryover
- Monitors internal pipeline registers
- Counts MAC operations per class
- Verifies state transitions occur at correct times

#### Test Patterns
- Class 0: weight = pixel_idx % 10
- Class 1: weight = -(pixel_idx % 10)
- Class 2: weight = 5
- Class 3: weight = -5
- (and so on for all 10 classes)
- Input pixels: (input_addr % 17) + 1

#### Usage
```bash
xvlog tb_pipeline_flush.v inference.v
xelab tb_pipeline_flush
xsim tb_pipeline_flush -runall
```

#### Expected Output
- All 10 classes: exactly 784 MAC operations
- Pipeline registers properly reset
- Accumulator properly reset for each class
- Test PASSED message

---

### Test 3: tb_edge_case_zeros.v - All Zeros Edge Case

#### Purpose
Verifies inference behavior when all weights and inputs are zero, ensuring predictions are based purely on biases.

#### What It Tests
- **Zero handling**: Proper computation when all weights are 0
- **Zero inputs**: Correct behavior with all input pixels at 0
- **Accumulator behavior**: Verifies accumulator stays at 0 throughout computation
- **Bias-only prediction**: Ensures argmax works correctly based solely on bias values
- **Edge case robustness**: No garbage accumulation or unexpected behavior

#### Test Configuration
- **Weights**: All set to 0
- **Input pixels**: All set to 0
- **Biases**: Class N has bias = N × 1000
  - Class 0: bias = 0
  - Class 1: bias = 1000
  - Class 2: bias = 2000
  - ...
  - Class 9: bias = 9000
- **Expected Result**: Class 9 (highest bias)

#### What Gets Verified
1. Accumulator remains at 0 during entire COMPUTE state
2. All 784 multiplications yield 0 (0 × 0 = 0)
3. Final score equals bias value for each class
4. Predicted digit is class 9 (max bias)

#### Usage
```bash
xvlog tb_edge_case_zeros.v inference.v
xelab tb_edge_case_zeros
xsim tb_edge_case_zeros -runall
```

#### Expected Output
```
Predicted Digit: 9
✓ CORRECT: Class 9 has the highest bias (9000)
✓ All weights were zero
✓ All inputs were zero
✓ Accumulator remained at 0 during computation
✓ Prediction based purely on biases
TEST 3 PASSED
```

---

### Test 4: tb_edge_case_max_values.v - Maximum Values Edge Case

#### Purpose
Verifies inference behavior with maximum positive int8 values to ensure no spurious overflow occurs with legitimate large values.

#### What It Tests
- **Maximum value handling**: Proper computation with all weights at +127
- **Maximum inputs**: Correct behavior with all input pixels at +127
- **Accumulator capacity**: Verifies 32-bit accumulator can handle large legitimate values
- **Overflow detection**: Ensures no overflow with values within valid range
- **Arithmetic correctness**: Validates signed multiplication and accumulation

#### Test Configuration
- **Weights**: All set to +127 (max positive int8)
- **Input pixels**: All set to +127 (max positive int8)
- **Product per MAC**: 127 × 127 = 16,129 (fits in 16-bit signed)
- **Expected accumulator**: 16,129 × 784 = 12,645,136
- **32-bit signed range**: -2,147,483,648 to +2,147,483,647
- **Biases**: Class N has bias = N × 10,000
  - Class 0: bias = 0
  - Class 1: bias = 10,000
  - Class 2: bias = 20,000
  - ...
  - Class 9: bias = 90,000
- **Expected Result**: Class 9 (highest final score: 12,645,136 + 90,000 = 12,735,136)

#### What Gets Verified
1. Products correctly calculated: 127 × 127 = 16,129
2. Accumulator reaches expected value: 12,645,136 (no overflow)
3. Value is well within 32-bit signed range (12,645,136 < 2,147,483,647)
4. Accumulator never goes negative during computation
5. Bias addition works correctly with large accumulator values
6. Predicted digit is class 9 (max final score)

#### Mathematical Verification
```
Per MAC:     127 × 127 = 16,129 (16-bit signed ✓)
Accumulator: 16,129 × 784 = 12,645,136 (32-bit signed ✓)
Max score:   12,645,136 + 90,000 = 12,735,136
Check:       12,735,136 < 2^31 - 1 = 2,147,483,647 ✓
Result:      NO OVERFLOW EXPECTED
```

#### Usage
```bash
xvlog tb_edge_case_max_values.v inference.v
xelab tb_edge_case_max_values
xsim tb_edge_case_max_values -runall
```

#### Expected Output
```
Predicted Digit: 9
✓ CORRECT: Class 9 has the highest final score
✓ All weights were +127 (max positive)
✓ All inputs were +127 (max positive)
✓ Products correctly calculated: 127 × 127 = 16,129
✓ Accumulator correctly reached: 12,645,136
✓ No overflow occurred (value within 32-bit signed range)
✓ Correctly predicted class with maximum final score (class 9)
TEST 4 PASSED
```

---

### Test 5: tb_edge_case_min_values.v - Minimum Negative Values Edge Case

#### Purpose
Verifies inference behavior with minimum negative int8 values to validate signed arithmetic, especially that negative × negative = positive.

#### What It Tests
- **Minimum negative value handling**: Proper computation with all weights at -128
- **Minimum negative inputs**: Correct behavior with all input pixels at -128
- **Signed multiplication correctness**: (-128) × (-128) = +16,384 (POSITIVE!)
- **Two's complement arithmetic**: Validates proper signed arithmetic throughout
- **Accumulator capacity**: Verifies 32-bit accumulator handles large positive result from negative operands

#### Test Configuration
- **Weights**: All set to -128 (min negative int8, 0x80)
- **Input pixels**: All set to -128 (min negative int8, 0x80)
- **Product per MAC**: (-128) × (-128) = +16,384 (POSITIVE result!)
- **Expected accumulator**: 16,384 × 784 = 12,845,056 (POSITIVE!)
- **32-bit signed range**: -2,147,483,648 to +2,147,483,647
- **Biases**: Class N has bias = N × 10,000
  - Class 0: bias = 0
  - Class 1: bias = 10,000
  - Class 2: bias = 20,000
  - ...
  - Class 9: bias = 90,000
- **Expected Result**: Class 9 (highest final score: 12,845,056 + 90,000 = 12,935,056)

#### What Gets Verified
1. Products correctly calculated: (-128) × (-128) = +16,384 (positive!)
2. Accumulator reaches expected POSITIVE value: 12,845,056
3. Negative × Negative = Positive (signed arithmetic correctness)
4. Value is well within 32-bit signed range
5. Accumulator never incorrectly goes negative
6. Two's complement representation works properly
7. Predicted digit is class 9 (max final score)

#### Key Insight
This test specifically validates that the inference module correctly implements **signed arithmetic**:
```
Negative × Negative = Positive
(-128) × (-128) = +16,384  ✓ (not -16,384)
```

If this fails, it indicates issues with:
- Sign extension in the multiplication pipeline
- Two's complement interpretation
- Signed vs unsigned arithmetic confusion

#### Mathematical Verification
```
Per MAC:     (-128) × (-128) = +16,384 (16-bit signed ✓)
Accumulator: 16,384 × 784 = 12,845,056 (32-bit signed ✓)
Max score:   12,845,056 + 90,000 = 12,935,056
Check:       12,935,056 < 2^31 - 1 = 2,147,483,647 ✓
Result:      NO OVERFLOW EXPECTED, POSITIVE VALUE
```

#### Usage
```bash
xvlog tb_edge_case_min_values.v inference.v
xelab tb_edge_case_min_values
xsim tb_edge_case_min_values -runall
```

#### Expected Output
```
Predicted Digit: 9
✓ CORRECT: Class 9 has the highest final score
✓ All weights were -128 (min negative)
✓ All inputs were -128 (min negative)
✓ Products correctly calculated: (-128) × (-128) = +16,384
✓ Negative × Negative = Positive (signed arithmetic correct!)
✓ Accumulator correctly reached: 12,845,056 (POSITIVE)
✓ No overflow occurred
✓ Two's complement arithmetic validated
TEST 5 PASSED
```

---

### Test 6: tb_edge_case_mixed_signs.v - Mixed Signs Edge Case

#### Purpose
Verifies inference behavior with mixed positive and negative values to validate negative accumulator handling and argmax with mixed-sign scores.

#### What It Tests
- **Mixed sign multiplication**: Positive × Negative = Negative
- **Negative accumulator handling**: Proper computation with negative intermediate results
- **Argmax with mixed scores**: Correct selection from both negative and positive final scores
- **Bias overcoming negative accumulator**: Large positive bias can make final score positive
- **Signed arithmetic edge cases**: Validates full range of signed operations

#### Test Configuration
- **Weights**: All set to -128 (min negative int8, 0x80)
- **Input pixels**: All set to +127 (max positive int8)
- **Product per MAC**: 127 × (-128) = -16,256 (NEGATIVE!)
- **Expected accumulator**: -16,256 × 784 = -12,744,704 (NEGATIVE!)
- **32-bit signed range**: -2,147,483,648 to +2,147,483,647
- **Biases**: Class N has bias = N × 2,000,000
  - Class 0: bias = 0
  - Class 1: bias = 2,000,000
  - Class 2: bias = 4,000,000
  - ...
  - Class 9: bias = 18,000,000

#### Expected Final Scores
```
Class 0: -12,744,704 + 0         = -12,744,704 (negative)
Class 1: -12,744,704 + 2,000,000 = -10,744,704 (negative)
Class 2: -12,744,704 + 4,000,000 =  -8,744,704 (negative)
Class 3: -12,744,704 + 6,000,000 =  -6,744,704 (negative)
Class 4: -12,744,704 + 8,000,000 =  -4,744,704 (negative)
Class 5: -12,744,704 + 10,000,000 = -2,744,704 (negative)
Class 6: -12,744,704 + 12,000,000 =    -744,704 (negative)
Class 7: -12,744,704 + 14,000,000 =   1,255,296 (POSITIVE!)
Class 8: -12,744,704 + 16,000,000 =   3,255,296 (POSITIVE!)
Class 9: -12,744,704 + 18,000,000 =   5,255,296 (POSITIVE! ← WINNER)
```

**Expected Result**: Class 9 (highest score: 5,255,296)

#### What Gets Verified
1. Products correctly calculated: 127 × (-128) = -16,256 (negative!)
2. Accumulator reaches expected NEGATIVE value: -12,744,704
3. Positive × Negative = Negative (signed arithmetic correctness)
4. Negative accumulator never incorrectly goes positive during computation
5. Bias addition works correctly with negative accumulator
6. Large biases can overcome negative accumulator (classes 7-9 become positive)
7. Argmax correctly handles mix of negative and positive scores
8. Predicted digit is class 9 (max final score)

#### Key Insights
This test validates three critical scenarios:
1. **Signed multiplication with mixed signs**: Ensures positive × negative = negative
2. **Negative intermediate values**: Tests that negative accumulators are handled properly
3. **Argmax with mixed-sign inputs**: Validates that argmax works correctly when some scores are negative and others are positive

#### Mathematical Verification
```
Per MAC:     127 × (-128) = -16,256 (16-bit signed ✓)
Accumulator: -16,256 × 784 = -12,744,704 (32-bit signed ✓)
Max score:   -12,744,704 + 18,000,000 = 5,255,296
Check:       -12,744,704 > -2^31 = -2,147,483,648 ✓
Result:      NO OVERFLOW, NEGATIVE ACCUMULATOR, POSITIVE FINAL SCORE
```

#### Usage
```bash
xvlog tb_edge_case_mixed_signs.v inference.v
xelab tb_edge_case_mixed_signs
xsim tb_edge_case_mixed_signs -runall
```

#### Expected Output
```
Predicted Digit: 9
✓ CORRECT: Class 9 has the highest final score
✓ All weights were -128 (min negative)
✓ All inputs were +127 (max positive)
✓ Products correctly calculated: 127 × (-128) = -16,256
✓ Positive × Negative = Negative (signed arithmetic correct!)
✓ Accumulator correctly reached: -12,744,704 (NEGATIVE)
✓ Negative accumulator handled properly
✓ Bias addition correctly overcame negative accumulator
✓ Argmax correctly selected maximum from mixed pos/neg scores
TEST 6 PASSED
```

---

## Verilog Testbench Comparison

| Test | Purpose | Complexity | What It Catches |
|------|---------|------------|-----------------|
| **Test 1** | Basic functionality | Simple | Major logic errors, basic inference flow |
| **Test 2** | Pipeline correctness | Moderate | Off-by-one errors, pipeline bugs, register leaks |
| **Test 3** | Edge case: zeros | Simple | Zero handling, bias addition, argmax correctness |
| **Test 4** | Edge case: max positive | Simple | Positive overflow issues, large value handling |
| **Test 5** | Edge case: min negative | Simple | Signed arithmetic, two's complement, neg×neg=pos |
| **Test 6** | Edge case: mixed signs | Moderate | Negative accumulator, mixed-sign argmax, pos×neg=neg |

---

## Running All Tests

### Python Tests
```bash
cd regresja/testing
python simulate_fpga_inference.py
python test_fpga_integration.py
```

### Verilog Tests (Vivado)
```bash
cd regresja/inference/tb
vivado -mode batch -source run_all_tests.tcl
```

Or individually:
```bash
# Test 1: Basic functionality
xvlog ../rtl/inference.v tb_inference.v
xelab tb_inference
xsim tb_inference -runall

# Test 2: Pipeline verification
xvlog ../rtl/inference.v tb_pipeline_flush.v
xelab tb_pipeline_flush
xsim tb_pipeline_flush -runall

# Test 3: Edge case - all zeros
xvlog ../rtl/inference.v tb_edge_case_zeros.v
xelab tb_edge_case_zeros
xsim tb_edge_case_zeros -runall

# Test 4: Edge case - maximum positive values
xvlog ../rtl/inference.v tb_edge_case_max_values.v
xelab tb_edge_case_max_values
xsim tb_edge_case_max_values -runall

# Test 5: Edge case - minimum negative values
xvlog ../rtl/inference.v tb_edge_case_min_values.v
xelab tb_edge_case_min_values
xsim tb_edge_case_min_values -runall

# Test 6: Edge case - mixed signs
xvlog ../rtl/inference.v tb_edge_case_mixed_signs.v
xelab tb_edge_case_mixed_signs
xsim tb_edge_case_mixed_signs -runall
```

---

## Future Improvements

### Possible Enhancements
1. **Add timing measurements** to both Python and Verilog tests
2. **Export misclassified images** for visual debugging
3. **Add overflow mitigation tests** (e.g., 64-bit accumulator simulation)
4. **Implement automated COM port detection**
5. **Add batch testing modes** for faster hardware tests
6. **Create visualization tools** for confusion matrices
7. **Add more Verilog edge cases**: 
   - Random value patterns for stress testing
   - Negative biases edge cases
   - Overflow scenarios (intentional overflow testing)
   - Inverse mixed signs (negative input, positive weights)
8. **Create automated regression test suite** for all testbenches
9. **Add waveform capture** for visual debugging in simulation
10. **Add coverage metrics** to measure test completeness

