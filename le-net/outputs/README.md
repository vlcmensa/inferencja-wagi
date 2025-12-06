# Outputs - Exported LeNet-5 Model Parameters

Contains exported weights and biases in different formats.

## Subfolders

| Folder | Description |
|--------|-------------|
| `mem/` | Memory files (`.mem`) - hex text format for Verilog simulation |
| `bin/` | Binary files (`.bin`) - raw bytes for UART transmission |

## Files

| File | Description | Type |
|------|-------------|------|
| `mnist_cnn.pth` | PyTorch model checkpoint | Binary |
| `mnist_cnn.onnx` | ONNX exported model | Binary |

## Layer Files

### Convolutional Layers

| File | Description | Size |
|------|-------------|------|
| `conv1_weights` | Conv1 weights (6 filters, 1×5×5) | 150 values (INT8) |
| `conv1_biases` | Conv1 biases | 6 values (INT32) |
| `conv2_weights` | Conv2 weights (16 filters, 6×5×5) | 2,400 values (INT8) |
| `conv2_biases` | Conv2 biases | 16 values (INT32) |

### Fully Connected Layers

| File | Description | Size |
|------|-------------|------|
| `fc1_weights` | FC1 weights (400 → 120) | 48,000 values (INT8) |
| `fc1_biases` | FC1 biases | 120 values (INT32) |
| `fc2_weights` | FC2 weights (120 → 84) | 10,080 values (INT8) |
| `fc2_biases` | FC2 biases | 84 values (INT32) |
| `fc3_weights` | FC3 weights (84 → 10) | 840 values (INT8) |
| `fc3_biases` | FC3 biases | 10 values (INT32) |

### Quantization Parameters

| File | Description |
|------|-------------|
| `shifts.bin` | Scale/shift factors for each layer |

## Total Size

- Weights: 150 + 2,400 + 48,000 + 10,080 + 840 = **61,470 bytes**
- Biases: (6 + 16 + 120 + 84 + 10) × 4 = **944 bytes**
- **Total: ~62,414 bytes (61 KB)**

## Formats

### MEM Format (`.mem`)

Text file with one hex value per line:
- Weights: 2 hex characters per value (8-bit)
- Biases: 8 hex characters per value (32-bit)

### BIN Format (`.bin`)

Raw binary data for serial transmission:
- Weights: 1 byte per value
- Biases: 4 bytes per value (little-endian)

## Generation

Weights are automatically generated when running `python/le_net.py`.

