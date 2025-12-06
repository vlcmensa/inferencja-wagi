# Softmax Regression Model

Single-layer softmax (logistic) regression classifier for MNIST digits.

## Model Architecture

- **Input**: 784 pixels (28×28 grayscale image)
- **Output**: 10 classes (digits 0-9)
- **Parameters**: 
  - Weights: 784 × 10 = 7840 values (INT8)
  - Biases: 10 values (INT32)

## Folder Structure

| Folder | Description |
|--------|-------------|
| `python/` | Python scripts for training, export, and UART communication |
| `inference/` | Verilog HDL source files and constraints for FPGA implementation |
| `data/` | Saved preprocessing parameters (StandardScaler) |
| `outputs/` | Exported model weights and biases |

## How to Use

### 1. Train the Model

```bash
uv run python/soft_reg_lepsza_kwant.py
```

This generates:
- `W_improved.mem` - Quantized weights (INT8)
- `B_improved.mem` - Quantized biases (INT32)
- `scaler_mean.npy`, `scaler_scale.npy` - Preprocessing parameters

### 2. Convert to Binary

```bash
uv run ../shared/convert_to_binary.py outputs/mem outputs/bin
```

### 3. Upload Weights to FPGA

```bash
uv run python/send_weights.py COM3 115200
```

### 4. Send Test Image

```bash
uv run python/send_image.py ../test_images/00001.png COM3 115200
```

## Expected Accuracy

- Float model: ~92%
- Quantized model: ~91-92% (minimal accuracy loss)




