# MEM Files - LeNet-5 Hex Format Weights

Memory initialization files in Verilog-compatible format.

## Files

| File | Lines | Description |
|------|-------|-------------|
| `conv1_weights.mem` | 150 | Conv1 weights (2 hex chars each) |
| `conv1_biases.mem` | 6 | Conv1 biases (8 hex chars each) |
| `conv2_weights.mem` | 2,400 | Conv2 weights (2 hex chars each) |
| `conv2_biases.mem` | 16 | Conv2 biases (8 hex chars each) |
| `fc1_weights.mem` | 48,000 | FC1 weights (2 hex chars each) |
| `fc1_biases.mem` | 120 | FC1 biases (8 hex chars each) |
| `fc2_weights.mem` | 10,080 | FC2 weights (2 hex chars each) |
| `fc2_biases.mem` | 84 | FC2 biases (8 hex chars each) |
| `fc3_weights.mem` | 840 | FC3 weights (2 hex chars each) |
| `fc3_biases.mem` | 10 | FC3 biases (8 hex chars each) |

## Generation

Run `python/le_net.py` to generate these files from the trained model.

## Format

Each line contains one value in hexadecimal:

- **Weights** (8-bit signed as unsigned hex):
  - Positive: `00` to `7F`
  - Negative: `80` to `FF` (two's complement)

- **Biases** (32-bit signed as unsigned hex):
  - 8 hex characters per value
  - Little-endian format

## Usage in Verilog

```verilog
reg [7:0] conv1_weights [0:149];
reg [31:0] conv1_biases [0:5];

initial begin
    $readmemh("conv1_weights.mem", conv1_weights);
    $readmemh("conv1_biases.mem", conv1_biases);
end
```

