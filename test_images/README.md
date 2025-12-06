# Test Images

Collection of handwritten digit images for testing FPGA inference.

## Format

- **Resolution**: 28×28 pixels (MNIST format)
- **Color**: Grayscale (white digit on black background)
- **Format**: PNG

## File Naming

Files are named with 5-digit numbers (e.g., `00001.png`, `00087.png`).

The number typically corresponds to the MNIST test set index.

## Usage

### Send Single Image to Softmax Regression

```bash
uv run regresja/python/send_image.py test_images/00001.png COM3 115200
```

### Send to Two-Hidden-Layer Network

```bash
uv run 2_ukryte/python/send_image.py test_images/00001.png COM3 115200
```

### Send to LeNet-5 CNN

```bash
uv run le-net/python/send_image.py test_images/00001.png COM3 115200
```

### Test All Images

```bash
uv run regresja/python/test_all_images.py
```

## Adding Custom Images

When adding your own handwritten digit images:

1. Draw black digit on white background (or vice versa)
2. Save as PNG (any resolution)
3. The `send_image.py` script will:
   - Resize to 20×20 pixels
   - Center in 28×28 canvas
   - Invert colors if needed (use `--no-invert` flag if already white-on-black)
   - Apply preprocessing (StandardScaler + quantization)




