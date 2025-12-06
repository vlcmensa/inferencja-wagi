# Python Scripts - LeNet-5 CNN

Python scripts for training, testing, and FPGA communication.

## Files

| Script | Description |
|--------|-------------|
| `le_net.py` | Train LeNet-5 CNN with quantization and export weights |
| `send_weights.py` | Upload model weights to FPGA via UART |
| `send_image.py` | Send test image to FPGA for inference |
| `test_model.py` | Test quantized model locally in Python |
| `test_all_images.py` | Batch test all images from test_images folder |

## UART Protocol

### Weights Upload (`send_weights.py`)

- Start marker: `0xAA 0x55`
- Data structure:
  - Conv1 weights + biases
  - Conv2 weights + biases
  - FC1 weights + biases
  - FC2 weights + biases
  - FC3 weights + biases
  - Scale/shift factors
- End marker: `0x55 0xAA`

Total size: ~61KB of parameters

### Image Upload (`send_image.py`)

- Start marker: `0xBB 0x66`
- Data: 784 bytes (28×28 pixels, preprocessed INT8)
- End marker: `0x66 0xBB`

## Image Preprocessing

Before sending to FPGA, images are:

1. Resized to 20×20 pixels
2. Centered in 28×28 canvas (4px padding)
3. Inverted if needed (white digit on black background)
4. Normalized with standard MNIST normalization (mean=0.1307, std=0.3081)
5. Quantized to INT8 (-128 to 127)

## Training

The `le_net.py` script:
1. Loads MNIST dataset via torchvision
2. Trains LeNet-5 with tanh activations
3. Performs post-training quantization
4. Exports weights to `.mem` and `.bin` formats
5. Saves model checkpoint and ONNX export

Training time: ~5-10 minutes on CPU, ~1-2 minutes on GPU

