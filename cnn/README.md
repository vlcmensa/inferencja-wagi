# Simple CNN FPGA Implementation for MNIST

This project implements a simple Convolutional Neural Network (CNN) on the Basys3 FPGA for MNIST digit recognition. This is an evolution of a previous MLP project, incorporating critical bug fixes learned from previous implementations.

## Architecture

### Model
- **Input**: 28x28 Grayscale (Int8)
- **Layer 1 (Conv)**: 4 Filters, 3x3 Kernel, Stride 1, No Padding
  - Output: 26×26×4 feature map (2,704 values)
  - Activation: ReLU
  - Quantization: Right-shift by 7 after accumulation
- **Layer 2 (Dense)**: Flatten (2,704 inputs) → 10 Outputs
  - Activation: None (Raw Scores)

### Memory Requirements
- **Conv Weights**: 4 × 1 × 3 × 3 = 36 bytes
- **Conv Biases**: 4 × 4 = 16 bytes
- **Dense Weights**: 2,704 × 10 = 27,040 bytes
- **Dense Biases**: 10 × 4 = 40 bytes
- **Total Payload**: ~27 KB

## Critical Bug Fixes

This project incorporates three critical fixes learned from previous MLP implementations:

1. **Binary Safety**: `uart_router.v` counts bytes and ignores protocol markers until the payload is full, preventing random data that resembles markers from terminating transfers early.

2. **Flow Control**: Python scripts use chunking + `time.sleep()` to prevent UART RX buffer overflow during transmission.

3. **Alignment**: `image_loader.v` drops the first byte (protocol artifact) to align pixels correctly, ensuring Pixel 0 lands in Address 0.

## Project Structure

```
cnn/
├── training/
│   └── train_cnn.py          # PyTorch training and weight export
├── inference/
│   ├── rtl/                   # Verilog RTL modules
│   │   ├── top.v             # Top-level module
│   │   ├── inference.v       # CNN inference engine
│   │   ├── uart_router.v     # UART protocol router (binary safe)
│   │   ├── load_weights.v    # Weight loader
│   │   ├── image_loader.v    # Image loader (alignment fix)
│   │   ├── ram_cnn.v         # Feature map and dense weights RAM
│   │   ├── conv_ram.v        # Conv weights and biases RAM
│   │   ├── image_ram.v       # Input image RAM
│   │   └── ... (supporting modules)
│   └── constraints/
│       └── pins.xdc          # Basys3 pin constraints
├── utils/
│   ├── send_weights.py       # Send weights to FPGA
│   └── send_image.py         # Send image to FPGA (with flow control)
├── testing/
│   └── compare_fpga.py       # Test FPGA inference
└── outputs/
    ├── bin/                  # Binary weight files
    └── npy/                  # Normalization parameters
```

## Usage

### 1. Train and Export Weights

```bash
cd training
python train_cnn.py
```

This will:
- Train the CNN model on MNIST
- Export quantized weights to `../outputs/bin/`
- Export normalization parameters to `../outputs/npy/`

### 2. Synthesize and Program FPGA

1. Open Vivado and create a new project
2. Add all RTL files from `inference/rtl/`
3. Add constraints from `inference/constraints/pins.xdc`
4. Synthesize, implement, and generate bitstream
5. Program the Basys3 board

### 3. Send Weights to FPGA

```bash
cd utils
python send_weights.py --port COM7
```

This sends all weights/biases to the FPGA. Wait for completion before sending images.

### 4. Send Image and Test

```bash
# Send a single image
python send_image.py 0 --port COM7

# Test inference
cd ../testing
python compare_fpga.py --port COM7 --index 0
```

The FPGA will:
- Receive the image
- Automatically start inference
- Display the predicted digit on the 7-segment display
- Store results for UART readback

### 5. Read Results via UART

Send command bytes:
- `0xCC`: Request predicted digit (1 byte response)
- `0xCD`: Request all class scores (40 bytes response)

## Protocol

### Weight Transfer
- Start: `0xAA 0x55`
- Data: Conv weights (36) + Conv biases (16) + Dense weights (27,040) + Dense biases (40)
- End: `0x55 0xAA`

### Image Transfer
- Start: `0xBB 0x66`
- Data: 784 pixels (preprocessed, quantized)
- End: `0x66 0xBB`

### Commands
- `0xCC`: Read predicted digit
- `0xCD`: Read all class scores

## Notes

- All Python scripts use chunked transmission with flow control delays to prevent UART buffer overflow
- The inference module automatically starts when an image is loaded
- Results are displayed on the 7-segment display and can be read via UART
- The system is binary-safe: protocol markers are only checked after receiving the expected number of bytes

## Dependencies

- PyTorch
- torchvision
- numpy
- pyserial


