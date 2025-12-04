# MEM Files - Hex Format Weights

Memory initialization files in Verilog-compatible format.

## Files

| File | Size | Description |
|------|------|-------------|
| `W.mem` | 7840 lines | Weights (784 × 10), 2 hex chars per line |
| `B.mem` | 10 lines | Biases, 8 hex chars per line |

## Format

Each line contains one value in hexadecimal:

- **Weights** (8-bit signed as unsigned hex):
  - Positive: `00` to `7F`
  - Negative: `80` to `FF` (two's complement)

- **Biases** (32-bit signed as unsigned hex):
  - 8 hex characters per value

## Usage in Verilog

```verilog
initial $readmemh("W.mem", weight_memory);
initial $readmemh("B.mem", bias_memory);
```

