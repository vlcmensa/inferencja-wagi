import serial
import time
import os
import struct
import numpy as np
import argparse
import torch
from torchvision import datasets, transforms

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200
BIN_DIR = "../outputs/bin"
NPY_DIR = "../outputs/npy"
OUTPUT_FILE = "../outputs/txt/comparison.txt"

# Protocol Markers
IMG_START = bytes([0xBB, 0x66])
IMG_END   = bytes([0x66, 0xBB])
CMD_READ_SCORES = bytes([0xCD])

# ==========================================
# BIT-EXACT PYTHON SIMULATION
# ==========================================
def simulate_cnn_inference(image, conv_w, conv_b, dense_w, dense_b):
    """
    Simulates the FPGA CNN pipeline exactly using integer arithmetic.
    """
    # 1. Reshape Input Image (784 -> 28x28)
    img_2d = image.reshape(28, 28).astype(np.int32)
    
    # 2. Reshape Conv Weights (4 filters, 1ch, 3x3) -> (4, 3, 3)
    conv_w_reshaped = conv_w.reshape(4, 3, 3).astype(np.int32)
    
    # 3. Layer 1: Convolution
    # Buffer for feature maps: 4 filters * 26 * 26
    feature_maps = np.zeros((4, 26, 26), dtype=np.int32)

    for f in range(4): # Loop over 4 filters
        bias_val = conv_b[f]
        for r in range(26):
            for c in range(26):
                acc = 0
                # 3x3 Convolution
                for kr in range(3):
                    for kc in range(3):
                        pixel = img_2d[r+kr, c+kc]
                        weight = conv_w_reshaped[f, kr, kc]
                        acc += pixel * weight
                
                # Add Bias
                acc += bias_val
                
                # FPGA Pipeline Steps:
                acc = acc >> 7          # 1. Arithmetic Right Shift by 7
                if acc < 0: acc = 0     # 2. ReLU
                if acc > 127: acc = 127 # 3. Saturation (clamp to 8-bit)
                
                feature_maps[f, r, c] = acc

    # 4. Layer 2: Dense
    # Flatten order must match FPGA memory write order:
    # [Filter0 (all rows), Filter1 (all rows)...]
    flattened_fm = feature_maps.flatten().astype(np.int32)
    
    # Reshape Dense Weights (10 classes, 2704 inputs)
    dense_w_reshaped = dense_w.reshape(10, 2704).astype(np.int32)
    
    scores = np.zeros(10, dtype=np.int32)
    
    for c in range(10):
        # Accumulate Dot Product
        # We use int64 here to strictly avoid python overflow, 
        # though FPGA wraps at 32-bit. 
        dot_prod = np.dot(flattened_fm, dense_w_reshaped[c])
        acc = dot_prod + dense_b[c]
        
        # Cast back to int32 to simulate FPGA register width
        scores[c] = np.int32(acc)
        
    return scores

# ==========================================
# UTILITIES
# ==========================================
def load_files():
    """Load weights and normalization parameters."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(script_dir, BIN_DIR)
    npy_path = os.path.join(script_dir, NPY_DIR)

    # Weights
    cw = np.fromfile(os.path.join(bin_path, "conv_weights.bin"), dtype=np.int8)
    cb = np.fromfile(os.path.join(bin_path, "conv_biases.bin"), dtype=np.int32)
    dw = np.fromfile(os.path.join(bin_path, "dense_weights.bin"), dtype=np.int8)
    db = np.fromfile(os.path.join(bin_path, "dense_biases.bin"), dtype=np.int32)

    # Norm params
    mean = np.load(os.path.join(npy_path, "norm_mean.npy"))
    std = np.load(os.path.join(npy_path, "norm_std.npy"))

    return (cw, cb, dw, db), (mean, std)

def get_mnist_data(num_images=10):
    """Load first N images from MNIST test set."""
    transform = transforms.Compose([transforms.ToTensor()])
    # Adjust root path as needed based on where you run the script from
    dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    
    images = []
    labels = []
    for i in range(num_images):
        img, label = dataset[i]
        # Convert 28x28 tensor to flattened numpy array (0-255)
        img_np = (img.squeeze().numpy() * 255).astype(np.uint8).flatten()
        images.append(img_np)
        labels.append(label)
    return images, labels

def preprocess(image, mean, std):
    """Normalize and Quantize Image (matches FPGA input format)."""
    # Normalize to [0,1]
    x = image.astype(np.float32) / 255.0
    # Standardization
    x = (x - mean) / std
    # Quantize to int8 (scale 127)
    x = np.clip(np.round(x * 127.0), -128, 127).astype(np.int8)
    return x

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = parser.parse_args()

    # Determine absolute path for output file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, OUTPUT_FILE)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Open output file for writing
    output_file = open(log_path, "w", encoding='utf-8')

    # 1. Load Data
    # print("Loading weights and data...")  # Commented out - silent execution
    (cw, cb, dw, db), (mean, std) = load_files()
    images, labels = get_mnist_data(10)

    # 2. Open UART
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2) # Wait for FPGA reset
    except Exception as e:
        # print(f"Error opening {args.port}: {e}")  # Commented out - silent execution
        output_file.close()
        return

    # Write header
    output_file.write("Python Simulation vs FPGA Hardware Comparison\n")
    output_file.write("Simple CNN (Conv-ReLU-Dense)\n")
    output_file.write("=" * 80 + "\n")
    output_file.write(f"Total Images: 10\n")
    output_file.write(f"Hardware Shifts: Conv=7, Dense=0\n")
    output_file.write("=" * 80 + "\n")
    output_file.write("\n")

    correct_matches = 0

    for idx in range(10):
        img_raw = images[idx]
        label = labels[idx]
        
        # Preprocess
        img_input = preprocess(img_raw, mean, std)
        
        # --- A. PYTHON SIMULATION ---
        expected_scores = simulate_cnn_inference(img_input, cw, cb, dw, db)
        py_pred = np.argmax(expected_scores)

        # --- B. FPGA INFERENCE ---
        # 1. Send Image (with Flow Control)
        ser.reset_input_buffer()
        ser.write(IMG_START)
        
        # Chunking
        img_bytes = img_input.tobytes()
        for i in range(0, len(img_bytes), 64):
            ser.write(img_bytes[i:i+64])
            time.sleep(0.005) # 5ms delay
            
        ser.write(IMG_END)
        
        # Wait for inference
        time.sleep(0.1)
        
        # 2. Read Scores
        ser.write(CMD_READ_SCORES)
        response = ser.read(40)
        
        if len(response) != 40:
            # print(f"Error: Received {len(response)} bytes from FPGA (expected 40)")  # Commented out
            continue
            
        # Unpack Little Endian 32-bit Signed Integers
        fpga_scores = np.array(struct.unpack('<10i', response), dtype=np.int32)
        fpga_pred = np.argmax(fpga_scores)

        # --- C. COMPARE ---
        is_match = np.array_equal(expected_scores, fpga_scores)
        if is_match: correct_matches += 1
        
        # Format scores as list string (Use tolist() to fix np.int32 format)
        py_scores_str = str(expected_scores.tolist())
        fpga_scores_str = str(fpga_scores.tolist())
        
        # Determine status
        if is_match and py_pred == fpga_pred:
            status = f"✓ MATCH (both predictions: {py_pred})"
        else:
            max_diff = np.max(np.abs(fpga_scores - expected_scores))
            status = f"✗ MISMATCH (Python: {py_pred}, FPGA: {fpga_pred}) - Max Diff: {max_diff}"
        
        # Write formatted output
        output_file.write(f"Image {idx:3d} | Label: {label} | Python Pred: {py_pred} | Python: {py_scores_str}\n")
        output_file.write(f"          |           | FPGA Pred: {fpga_pred:2d}    | FPGA:   {fpga_scores_str}\n")
        output_file.write(f"          |           | Status: {status}\n")
        output_file.write("\n")

    # Write summary
    output_file.write("=" * 80 + "\n")
    output_file.write(f"Final Summary: {correct_matches}/10 Exact Logic Matches\n")
    output_file.write("=" * 80 + "\n")
    
    output_file.close()
    ser.close()

if __name__ == "__main__":
    main()