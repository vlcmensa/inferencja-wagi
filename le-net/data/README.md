# Data - Normalization Parameters

Saved normalization parameters from training, required for inference.

## Files

| File | Description |
|------|-------------|
| `norm_mean.npy` | Mean value for MNIST normalization (0.1307) |
| `norm_std.npy` | Standard deviation for MNIST normalization (0.3081) |
| `scale_info.txt` | Quantization scale factors and model information |

## Usage

These files are used by `send_image.py` to preprocess images before sending to FPGA.

The preprocessing formula:

```
x_normalized = (x / 255.0 - mean) / std
x_int8 = round(x_normalized × 127)
```

## Standard MNIST Normalization

LeNet-5 uses the standard MNIST normalization values:
- Mean: 0.1307 (computed from entire MNIST training set)
- Std: 0.3081 (computed from entire MNIST training set)

These values normalize pixel intensities to approximately zero mean and unit variance.

## Important

Run `python/le_net.py` first to generate these files before testing inference.

