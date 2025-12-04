# Data - Preprocessing Parameters

Saved StandardScaler parameters from training, required for inference.

## Files

| File | Description |
|------|-------------|
| `scaler_mean.npy` | Mean values for each of 784 pixels |
| `scaler_scale.npy` | Scale (std dev) values for each pixel |
| `scale_info.txt` | Quantization scale factors information |

## Usage

These files are used by `send_image.py` to preprocess images before sending to FPGA.

The preprocessing formula:

```
x_scaled = (x / 255.0 - mean) / scale
x_int8 = round(x_scaled × 127)
```

## Important

Run `soft_reg_lepsza_kwant.py` first to generate these files before testing inference.

