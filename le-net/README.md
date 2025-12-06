# LeNet-5 Convolutional Neural Network

Classic CNN architecture (LeNet-5) for MNIST digit classification, featuring convolutional layers for automatic feature extraction.

## Model Architecture

```
Input (28×28)
   ↓
Conv1: 6 filters, 5×5 kernel → 28×28×6
   ↓ tanh activation
AvgPool: 2×2 → 14×14×6
   ↓
Conv2: 16 filters, 5×5 kernel → 10×10×16
   ↓ tanh activation
AvgPool: 2×2 → 5×5×16
   ↓ flatten
FC1: 400 → 120
   ↓ tanh activation
FC2: 120 → 84
   ↓ tanh activation
FC3: 84 → 10
   ↓
Output (10 classes)
```

## Parameters

| Layer | Weights | Biases | Total Parameters |
|-------|---------|--------|------------------|
| Conv1 | 6 × 1 × 5 × 5 = 150 | 6 | 156 |
| Conv2 | 16 × 6 × 5 × 5 = 2,400 | 16 | 2,416 |
| FC1 | 400 × 120 = 48,000 | 120 | 48,120 |
| FC2 | 120 × 84 = 10,080 | 84 | 10,164 |
| FC3 | 84 × 10 = 840 | 10 | 850 |
| **Total** | | | **61,706 parameters** |

## Folder Structure

| Folder | Description |
|--------|-------------|
| `python/` | Python scripts for training, export, and UART communication |
| `verilog/` | Verilog HDL source files for FPGA implementation |
| `data/` | Normalization parameters for preprocessing |
| `outputs/` | Exported model weights and biases |

## Quantization

The model uses INT8 quantization:
- Weights: INT8 (per-channel quantization)
- Biases: INT32 (accumulated scale)
- Activations: INT8 with tanh approximation
- Pooling: Average pooling with proper scaling

## How to Use

### 1. Train the Model

```bash
uv run python/le_net.py
```

This generates:
- `outputs/mnist_cnn.pth` - PyTorch model checkpoint
- `outputs/mnist_cnn.onnx` - ONNX export
- `outputs/mem/*.mem` - Quantized weights and biases (hex format)

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

### 5. Test All Images

```bash
uv run python/test_all_images.py
```

## Expected Accuracy

- Float model: ~98-99%
- Quantized model: ~97-98% (minimal accuracy loss)

LeNet-5 significantly outperforms simpler models due to convolutional feature extraction.

## Preprocessing

Images are normalized using the standard MNIST normalization:
- Mean: 0.1307
- Std: 0.3081

The normalization parameters are saved in `data/norm_mean.npy` and `data/norm_std.npy`.

