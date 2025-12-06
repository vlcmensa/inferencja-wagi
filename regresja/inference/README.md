# Inference RTL Implementation

Verilog source files for FPGA implementation of softmax regression inference.

## Folder Structure

| Folder | Description |
|--------|-------------|
| `rtl/` | Verilog RTL source files for inference logic |
| `contraints/` | Pin constraint files (XDC format for Xilinx FPGAs) |

## Files

### RTL Files (`rtl/`)

| File | Description |
|------|-------------|
| `inference.v` | Main inference module with multiply-accumulate logic and state machine |
| `load_weights.v` | UART receiver for loading weights/biases into BRAM |

### Constraints (`contraints/`)

| File | Description |
|------|-------------|
| `pins.xdc` | Pin assignments for Xilinx FPGA (Basys 3 / Nexys A7) |

## Module Hierarchy

```
softmax_regression_top
├── weight_loader      (UART RX, stores weights in BRAM)
├── image_loader       (UART RX, stores image pixels in RAM)
├── image_ram          (784-byte RAM for input image)
├── inference          (Main computation: MAC + argmax)
└── seven_segment_display (Shows predicted digit)
```

## Inference Algorithm

For each class i (0-9):

```
score[i] = Σ(input[j] × weight[i][j]) + bias[i]
           j=0 to 783
```

Final output = argmax(score[0..9])

## Timing

- Sequential processing: one MAC per clock cycle
- Total cycles: 784 × 10 = 7840 (plus overhead)
- At 100 MHz: ~80 µs per image

## Interface

- UART: 115200 baud (configurable)
- LEDs: Status indicators
- 7-segment display: Predicted digit (0-9)

