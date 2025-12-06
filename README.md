# FPGA-Based Neural Network Inference for MNIST Digit Classification

This project implements neural network inference on FPGA hardware for classifying handwritten digits (MNIST dataset). Multiple model architectures are supported, from simple logistic regression to convolutional neural networks.

## Project Goal

- Train neural network models in Python (PyTorch/scikit-learn)
- Export quantized weights and biases (INT8/INT32)
- Implement inference logic in Verilog (Vivado)
- Load model parameters to FPGA via UART
- Send images to FPGA for real-time classification

## Folder Structure

| Folder | Description |
|--------|-------------|
| `regresja/` | Softmax (logistic) regression model - simple single-layer classifier |
| `2_ukryte/` | Two-hidden-layer neural network (784 → 16 → 16 → 10) |
| `le-net/` | LeNet-5 convolutional neural network (CNN) architecture |
| `shared/` | Shared Python utilities used by all models |
| `test_images/` | PNG test images for FPGA inference testing |
| `data/` | MNIST dataset and preprocessing parameters |

## Workflow

1. **Train model** - Run Python script (e.g., `python/soft_reg_lepsza_kwant.py`)
2. **Export weights** - Save as `.mem` files (hex format for Verilog)
3. **Convert to binary** - Use `regresja/python/convert_to_binary.py` for UART transmission
4. **Upload to FPGA** - Use `send_weights.py` via serial port
5. **Run inference** - Send test images with `send_image.py`
6. **Check result** - Read predicted digit from 7-segment display or LEDs

## Hardware Requirements

- FPGA board with UART interface (tested on Basys 3 / Nexys A7)
- USB-to-Serial connection (115200 baud default)

## Dependencies

This project uses `uv` for Python package management. Install dependencies:

```bash
uv sync
```

Main dependencies: `torch`, `torchvision`, `scikit-learn`, `numpy`, `pillow`, `pyserial`




