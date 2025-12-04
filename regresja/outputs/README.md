# Outputs - Exported Model Parameters

Contains exported weights and biases in different formats.

## Subfolders

| Folder | Description |
|--------|-------------|
| `mem/` | Memory files (`.mem`) - hex text format for Verilog simulation |
| `bin/` | Binary files (`.bin`) - raw bytes for UART transmission |

## File Naming

- `W.mem` / `W.bin` - Weights (784 × 10 = 7840 values, INT8)
- `B.mem` / `B.bin` - Biases (10 values, INT32)

## Formats

### MEM Format (`.mem`)

Text file with one hex value per line:

- Weights: 2 hex characters per value (8-bit)
- Biases: 8 hex characters per value (32-bit)

### BIN Format (`.bin`)

Raw binary data for serial transmission:

- Weights: 1 byte per value
- Biases: 4 bytes per value (little-endian)

