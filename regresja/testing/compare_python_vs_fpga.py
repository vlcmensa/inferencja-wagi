"""
Compare Python Simulation vs FPGA Hardware Output

This script runs the same images through both:
1. Python simulation (with 32-bit overflow)
2. Actual FPGA hardware via UART

Results are written to a text file showing logits from both sources for comparison.
"""

import serial
import time
import sys
import os
import struct
import numpy as np
from sklearn.datasets import fetch_openml

# Import simulation functions
from simulate_fpga_inference import (
    fpga_inference_with_overflow,
    load_model_and_scaler,
    preprocess_image_fpga
)

# Configuration
COM_PORT = 'COM3'     # Change this to your FPGA's COM port
BAUD_RATE = 115200
TEST_SAMPLES = 100    # Number of images to test
INPUT_SCALE = 127.0   # Must match training
OUTPUT_FILE = '../outputs/txt/comparison_output.txt'

# Protocol Constants
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])
SCORES_READ_REQUEST = bytes([0xCD])


def load_mnist_test_data():
    """Load MNIST test dataset."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Use the last 10000 as test set (standard split)
    X_test = X[60000:]
    y_test = y[60000:].astype(int)
    
    return X_test, y_test


def read_scores_from_fpga(ser):
    """Read all 10 class scores from FPGA via 0xCD protocol.
    
    Returns:
        numpy array of 10 int32 scores, or None on error
    """
    ser.reset_input_buffer()
    ser.write(SCORES_READ_REQUEST)
    
    # Read 40 bytes (10 scores × 4 bytes each)
    resp = ser.read(40)
    
    if len(resp) != 40:
        print(f"Error: Expected 40 bytes, got {len(resp)}")
        return None
    
    # Unpack as 10 signed 32-bit integers (little-endian)
    scores = struct.unpack('<10i', resp)
    return np.array(scores, dtype=np.int32)


def preprocess_for_uart(image_f32, mean, scale):
    """
    Preprocess image for UART transmission (returns uint8 view).
    
    Args:
        image_f32: Raw pixel values (0-255)
        mean: Scaler mean
        scale: Scaler scale
    
    Returns:
        Preprocessed image as uint8 array (for UART transmission)
    """
    # 1. Normalize to [0, 1]
    img_norm = image_f32 / 255.0
    
    # 2. Standard scaler
    img_scaled = (img_norm - mean) / scale
    
    # 3. Quantize to int8 range
    img_quant = np.round(img_scaled * INPUT_SCALE)
    img_int8 = np.clip(img_quant, -128, 127).astype(np.int8)
    
    # 4. View as uint8 for UART transmission (Preserves 2's complement bits)
    return img_int8.view(np.uint8)


def run_comparison():
    """Main comparison function."""
    
    # Load Python simulation model
    print("Loading Python simulation model...")
    W, B, mean, scale = load_model_and_scaler()
    print(f"  Weights shape: {W.shape}, dtype: {W.dtype}")
    print(f"  Biases shape: {B.shape}, dtype: {B.dtype}")
    print()
    
    # Load MNIST test data
    X_test, y_test = load_mnist_test_data()
    print(f"Test data loaded: {len(X_test)} images")
    print()
    
    # Open output file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Opening output file: {output_path}")
    print()
    
    # Try to open serial connection
    print(f"Connecting to FPGA on {COM_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(1)  # Allow FPGA to reset
        ser.reset_input_buffer()
        print("✓ Connected to FPGA")
        print()
    except serial.SerialException as e:
        print(f"✗ Serial Error: {e}")
        print("Cannot proceed without FPGA connection.")
        sys.exit(1)
    
    # Start comparison
    print(f"Starting comparison on {TEST_SAMPLES} images...")
    print("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("Python Simulation vs FPGA Hardware Comparison\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Images: {TEST_SAMPLES}\n")
        f.write("=" * 80 + "\n\n")
        
        match_count = 0
        mismatch_count = 0
        error_count = 0
        
        for i in range(TEST_SAMPLES):
            label = y_test[i]
            
            # Preprocess image once
            img_int8 = preprocess_image_fpga(X_test[i], mean, scale)
            img_uart = preprocess_for_uart(X_test[i], mean, scale)
            
            # 1. Run Python simulation
            python_pred, python_scores, _ = fpga_inference_with_overflow(img_int8, W, B)
            
            # 2. Send to FPGA and get hardware scores
            ser.write(IMG_START_MARKER)
            ser.write(img_uart.tobytes())
            ser.write(IMG_END_MARKER)
            ser.flush()
            
            time.sleep(0.05)  # Wait for inference
            
            fpga_scores = read_scores_from_fpga(ser)
            
            # Format and write results
            if fpga_scores is None:
                f.write(f"Image {i:3d} | Label: {label} | Python Pred: {python_pred} | Python: {python_scores.tolist()}\n")
                f.write(f"                     | FPGA Pred: ERROR | FPGA:   ERROR - Failed to read scores\n\n")
                error_count += 1
            else:
                fpga_pred = np.argmax(fpga_scores)
                f.write(f"Image {i:3d} | Label: {label} | Python Pred: {python_pred} | Python: {python_scores.tolist()}\n")
                f.write(f"                     | FPGA Pred: {fpga_pred}   | FPGA:   {fpga_scores.tolist()}\n")
                
                # Check if they match
                if np.array_equal(python_scores, fpga_scores):
                    f.write(f"                     | Status: ✓ MATCH (both predictions: {python_pred})\n\n")
                    match_count += 1
                else:
                    max_diff = np.max(np.abs(python_scores - fpga_scores))
                    f.write(f"                     | Status: ✗ MISMATCH (max diff: {max_diff}, Python pred: {python_pred}, FPGA pred: {fpga_pred})\n\n")
                    mismatch_count += 1
            
            # Progress indicator (every 10 images)
            if (i + 1) % 10 == 0 or (i + 1) == TEST_SAMPLES:
                print(f"  Processed: {i+1:3d}/{TEST_SAMPLES} images | "
                      f"Matches: {match_count} | Mismatches: {mismatch_count} | Errors: {error_count}")
        
        # Write summary
        f.write("=" * 80 + "\n")
        f.write("END OF COMPARISON\n")
        f.write("=" * 80 + "\n")
    
    ser.close()
    
    print("=" * 80)
    print(f"Comparison complete!")
    print(f"  Total images:  {TEST_SAMPLES}")
    print(f"  Matches:       {match_count}")
    print(f"  Mismatches:    {mismatch_count}")
    print(f"  Errors:        {error_count}")
    print(f"  Output saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()

