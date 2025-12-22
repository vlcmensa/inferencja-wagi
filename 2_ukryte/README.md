# Two-Hidden-Layer Neural Network

Multi-layer perceptron with two hidden layers for MNIST classification on FPGA.

## Model Architecture

```
Input (784) → Hidden1 (16) → ReLU → Hidden2 (16) → ReLU → Output (10)
```

- **Input**: 784 pixels (28×28 image, 8-bit signed)
- **Hidden Layer 1**: 16 neurons + ReLU activation (right-shift by 7)
- **Hidden Layer 2**: 16 neurons + ReLU activation (right-shift by 7)
- **Output**: 10 classes (digits 0-9, no activation)

## Parameters

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| L1 (input → hidden1) | 784 × 16 = 12,544 bytes | 16 × 4 = 64 bytes | 12,608 bytes |
| L2 (hidden1 → hidden2) | 16 × 16 = 256 bytes | 16 × 4 = 64 bytes | 320 bytes |
| L3 (hidden2 → output) | 16 × 10 = 160 bytes | 10 × 4 = 40 bytes | 200 bytes |
| **Total** | 12,960 bytes | 168 bytes | 13,128 bytes |

## Current Status

### ✅ Inference Module Verified (100% Accuracy)

The `inference.v` module has been **completely verified** in isolation using `tb_inference.v`:

- **100% accuracy** on 100 test cases
- **Exact score match** with Python simulation (bit-accurate)
- All 10 class scores match Python output exactly
- Testbench confirms: **inference computation is correct**

### ⚠️ Known Issue: Data Transmission

The inference computation itself works perfectly, but there is a **data transmission problem** affecting:
- Weight loading via UART (`uart_router.v`, `load_weights.v`)
- Image loading via UART (`image_loader.v`)

**The problem is NOT in `inference.v`** - the issue lies in the UART communication/data loading modules.

## Folder Structure

| Folder | Description |
|--------|-------------|
| `training/` | Python training script (`siec_2_ukryte.py`) |
| `inference/` | Verilog HDL source files for FPGA implementation |
| `inference/tb/` | Testbench for isolated inference testing |
| `testing/` | Python scripts for testing and validation |
| `utils/` | Utilities for sending weights/images via UART |
| `outputs/` | Exported weights, biases, and test vectors |

## Quantization

Uses PyTorch with manual quantization:

- **Weights**: INT8 (scaled quantization)
- **Biases**: INT32 (accumulated scale)
- **Activations**: INT8 with ReLU
- **Right-shift**: Layers 1-2 use 7-bit right-shift for scaling

## Training

Run the training script:

```bash
cd training
uv run python siec_2_ukryte.py
```

This generates:
- Model checkpoint: `outputs/model.pth`
- Weight files: `outputs/bin/L*_weights.bin`, `outputs/bin/L*_biases.bin`
- Memory files: `outputs/mem/L*_weights.mem`, `outputs/mem/L*_biases.mem`

## Testing

### Generate Test Vectors

```bash
cd testing
python generate_test_vectors.py
```

### Run Isolated Testbench

The testbench (`inference/tb/tb_inference.v`) tests `inference.v` in complete isolation:

```bash
# In Vivado: Set tb_inference as top module and run simulation
```

**Expected Result**: 100% pass rate with exact score matches.

See `inference/tb/README.md` for detailed testbench documentation.
