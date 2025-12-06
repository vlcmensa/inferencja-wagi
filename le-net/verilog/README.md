# Verilog HDL Files - LeNet-5 CNN

Verilog source files for FPGA implementation of LeNet-5 CNN inference.

## Files

| File | Description |
|------|-------------|
| `inference.v` | Main inference module with CNN layers and state machine |
| `load_weights.v` | UART receiver for loading weights/biases into BRAM |
| `pins.v` | Pin constraints file (XDC format for Xilinx FPGAs) |

## Module Hierarchy

```
lenet5_top
├── weight_loader          (UART RX, stores all layer parameters)
├── image_loader           (UART RX, stores input image)
├── conv1_layer            (First convolutional layer)
├── pool1_layer            (First average pooling layer)
├── conv2_layer            (Second convolutional layer)
├── pool2_layer            (Second average pooling layer)
├── fc1_layer              (First fully connected layer)
├── fc2_layer              (Second fully connected layer)
├── fc3_layer              (Output fully connected layer)
└── seven_segment_display  (Shows predicted digit)
```

## CNN Inference Pipeline

1. **Conv1**: Apply 6 filters (5×5) with tanh activation
2. **Pool1**: 2×2 average pooling
3. **Conv2**: Apply 16 filters (5×5) with tanh activation
4. **Pool2**: 2×2 average pooling, flatten to 400 values
5. **FC1**: Fully connected 400 → 120 with tanh
6. **FC2**: Fully connected 120 → 84 with tanh
7. **FC3**: Fully connected 84 → 10 (output logits)
8. **Argmax**: Find highest score among 10 outputs

## Memory Requirements

- Weight BRAM: ~61 KB
- Bias BRAM: ~1 KB
- Feature maps: ~16 KB (intermediate activations)
- Input buffer: 784 bytes

Total: ~78 KB on-chip memory

## Timing

- Conv1: ~4,200 cycles (6 filters × 28×28 × 25 MACs)
- Pool1: ~2,400 cycles
- Conv2: ~40,000 cycles (16 filters × 10×10 × 25 MACs)
- Pool2: ~800 cycles
- FC layers: ~48,000 cycles
- **Total: ~95,000 cycles**
- At 100 MHz: **~950 µs per image**

## Interface

- UART: 115200 baud (configurable)
- LEDs: Status indicators
- 7-segment display: Predicted digit (0-9)
- Switches: Mode control (weight load / inference)

## Implementation Notes

- Uses fixed-point arithmetic (INT8/INT32)
- Tanh activation approximated with lookup table or piecewise linear
- Average pooling performed with bit shifts for efficiency
- Sequential MAC operations to minimize resource usage

