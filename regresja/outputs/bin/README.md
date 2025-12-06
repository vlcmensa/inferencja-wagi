# BIN Files - Binary Format for UART

Raw binary files for uploading to FPGA via serial port.

## Files

| File | Size (bytes) | Description |
|------|--------------|-------------|
| `W.bin` | 7840 | Weights, 1 byte each |
| `B.bin` | 40 | Biases, 4 bytes each (little-endian) |

## Generation

Created by running:

```bash
uv run ../../python/convert_to_binary.py ../mem/ ./
```

## Transmission Order

`send_weights.py` sends:

1. Start marker: `0xAA 0x55`
2. Weights: 7840 bytes
3. Biases: 40 bytes
4. End marker: `0x55 0xAA`

Total: 7884 bytes




