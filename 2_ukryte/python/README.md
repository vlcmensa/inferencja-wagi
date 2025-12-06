# Python Scripts - Two-Hidden-Layer Network

Scripts for training, exporting, and uploading the multi-layer model.

## Files

| Script | Description |
|--------|-------------|
| `siec_2_ukryte.py` | Train two-hidden-layer neural network model |
| `siec_int8.py` | Alternative training script with INT8 quantization |
| `export_weights.py` | Extract INT8 weights from trained model |
| `send_weights.py` | Upload all layer weights to FPGA via UART |

## Weight Export

`export_weights.py` loads the quantized `.pth` model and extracts:

- L1 weights and biases (input → hidden1)
- L2 weights and biases (hidden1 → hidden2)  
- L3 weights and biases (hidden2 → output)

Outputs `.mem` files with proper quantization scaling.

## UART Protocol

Similar to softmax regression, but with more data:

- Start marker: `0xAA 0x55`
- L1 weights (12,544 bytes) + L1 biases (64 bytes)
- L2 weights (256 bytes) + L2 biases (64 bytes)
- L3 weights (160 bytes) + L3 biases (40 bytes)
- End marker: `0x55 0xAA`

Total: ~13,128 bytes




