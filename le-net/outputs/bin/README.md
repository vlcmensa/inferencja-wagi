# BIN Files - LeNet-5 Binary Format for UART

Raw binary files for uploading to FPGA via serial port.

## Files

| File | Size (bytes) | Description |
|------|--------------|-------------|
| `conv1_weights.bin` | 150 | Conv1 weights, 1 byte each |
| `conv1_biases.bin` | 24 | Conv1 biases (6 × 4 bytes) |
| `conv2_weights.bin` | 2,400 | Conv2 weights, 1 byte each |
| `conv2_biases.bin` | 64 | Conv2 biases (16 × 4 bytes) |
| `fc1_weights.bin` | 48,000 | FC1 weights, 1 byte each |
| `fc1_biases.bin` | 480 | FC1 biases (120 × 4 bytes) |
| `fc2_weights.bin` | 10,080 | FC2 weights, 1 byte each |
| `fc2_biases.bin` | 336 | FC2 biases (84 × 4 bytes) |
| `fc3_weights.bin` | 840 | FC3 weights, 1 byte each |
| `fc3_biases.bin` | 40 | FC3 biases (10 × 4 bytes) |
| `shifts.bin` | Variable | Scale/shift factors for quantization |

## Total Size

- Weights: 150 + 2,400 + 48,000 + 10,080 + 840 = **61,470 bytes**
- Biases: 24 + 64 + 480 + 336 + 40 = **944 bytes**
- **Total: ~62,414 bytes (61 KB)**

## Generation

Created by running:

```bash
uv run ../../shared/convert_to_binary.py ../mem/ ./
```

## Transmission Order

`send_weights.py` sends:

1. Start marker: `0xAA 0x55`
2. Conv1 weights (150 bytes) + biases (24 bytes)
3. Conv2 weights (2,400 bytes) + biases (64 bytes)
4. FC1 weights (48,000 bytes) + biases (480 bytes)
5. FC2 weights (10,080 bytes) + biases (336 bytes)
6. FC3 weights (840 bytes) + biases (40 bytes)
7. Shifts/scales (variable)
8. End marker: `0x55 0xAA`

Total transmission time at 115200 baud: ~5-6 seconds

