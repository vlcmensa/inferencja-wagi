# Verilog HDL Files - Two-Hidden-Layer Network

Verilog source files for FPGA implementation of multi-layer inference.

## Files

| File | Description |
|------|-------------|
| `weight_load.v` | UART receiver and BRAM storage for all 3 layers |
| `pins.v` | Pin constraints file (XDC format for Xilinx FPGAs) |

## Memory Layout

| Address Range | Content | Size |
|---------------|---------|------|
| 0 - 12,543 | L1 weights | 12,544 bytes |
| 12,544 - 12,607 | L1 biases | 64 bytes |
| 12,608 - 12,863 | L2 weights | 256 bytes |
| 12,864 - 12,927 | L2 biases | 64 bytes |
| 12,928 - 13,087 | L3 weights | 160 bytes |
| 13,088 - 13,127 | L3 biases | 40 bytes |

## Inference Pipeline

1. **Layer 1**: 784 MACs per neuron × 16 neurons
2. **ReLU**: max(0, x) on 16 values
3. **Layer 2**: 16 MACs per neuron × 16 neurons
4. **ReLU**: max(0, x) on 16 values
5. **Layer 3**: 16 MACs per neuron × 10 neurons
6. **Argmax**: Find highest score among 10 outputs

## Status LEDs

- `led[0]`: Blinks when receiving UART bytes
- `led[1]`: Waiting for start marker
- `led[2]`: Receiving data
- `led[3]`: Transfer complete (success)
- `led[4]`: Error (buffer overflow)
- `led[15:8]`: Current address (progress)

