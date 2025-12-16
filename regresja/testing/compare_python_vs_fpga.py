"""
Compare Python Simulation vs FPGA Hardware Output

This script runs the same images through both:
1. Python simulation (with 32-bit overflow)
2. Actual FPGA hardware via UART

Results can be written to a text file and/or displayed on console.
"""

import serial
import time
import sys
import os
import struct
import numpy as np
from sklearn.datasets import fetch_openml
import argparse

# Import simulation functions
from simulate_fpga_inference import (
    fpga_inference_with_overflow,
    load_model_and_scaler,
    preprocess_image_fpga
)

# Default Configuration
DEFAULT_COM_PORT = 'COM3'
DEFAULT_BAUD_RATE = 115200
DEFAULT_TEST_SAMPLES = 100
INPUT_SCALE = 127.0   # Must match training
DEFAULT_OUTPUT_FILE = '../outputs/txt/comparison_output.txt'

# Protocol Constants
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])
SCORES_READ_REQUEST = bytes([0xCD])


class OutputWriter:
    """Handles output to file and/or console."""
    
    def __init__(self, output_mode='file', file_path=None):
        """
        Initialize output writer.
        
        Args:
            output_mode: 'file', 'console', or 'both'
            file_path: Path to output file (required if mode is 'file' or 'both')
        """
        self.mode = output_mode
        self.file_handle = None
        
        if output_mode in ['file', 'both']:
            if file_path is None:
                raise ValueError("file_path required for file output mode")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            self.file_handle = open(file_path, 'w', encoding='utf-8')
    
    def write(self, text):
        """Write text to configured output(s)."""
        if self.mode in ['console', 'both']:
            print(text, end='')
        
        if self.mode in ['file', 'both'] and self.file_handle:
            self.file_handle.write(text)
    
    def close(self):
        """Close file handle if open."""
        if self.file_handle:
            self.file_handle.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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


def run_comparison(com_port, baud_rate, test_samples, output_mode, output_file):
    """
    Main comparison function.
    
    Args:
        com_port: Serial port for FPGA connection
        baud_rate: Serial baud rate
        test_samples: Number of images to test
        output_mode: 'file', 'console', or 'both'
        output_file: Path to output file (used if mode is 'file' or 'both')
    """
    
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
    
    # Prepare output file path
    if output_mode in ['file', 'both']:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        print(f"Output will be saved to: {output_path}")
    else:
        output_path = None
        print("Output mode: Console only")
    print()
    
    # Try to open serial connection
    print(f"Connecting to FPGA on {com_port} at {baud_rate} baud...")
    try:
        ser = serial.Serial(com_port, baud_rate, timeout=2)
        time.sleep(1)  # Allow FPGA to reset
        ser.reset_input_buffer()
        print("✓ Connected to FPGA")
        print()
    except serial.SerialException as e:
        print(f"✗ Serial Error: {e}")
        print("Cannot proceed without FPGA connection.")
        sys.exit(1)
    
    # Start comparison
    print(f"Starting comparison on {test_samples} images...")
    print("=" * 80)
    print()
    
    with OutputWriter(output_mode, output_path) as writer:
        # Write header
        writer.write("Python Simulation vs FPGA Hardware Comparison\n")
        writer.write("=" * 80 + "\n")
        writer.write(f"Total Images: {test_samples}\n")
        writer.write("=" * 80 + "\n\n")
        
        match_count = 0
        mismatch_count = 0
        error_count = 0
        
        for i in range(test_samples):
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
                writer.write(f"Image {i:3d} | Label: {label} | Python Pred: {python_pred} | Python: {python_scores.tolist()}\n")
                writer.write(f"                     | FPGA Pred: ERROR | FPGA:   ERROR - Failed to read scores\n\n")
                error_count += 1
            else:
                fpga_pred = np.argmax(fpga_scores)
                writer.write(f"Image {i:3d} | Label: {label} | Python Pred: {python_pred} | Python: {python_scores.tolist()}\n")
                writer.write(f"                     | FPGA Pred: {fpga_pred}   | FPGA:   {fpga_scores.tolist()}\n")
                
                # Check if they match
                if np.array_equal(python_scores, fpga_scores):
                    writer.write(f"                     | Status: ✓ MATCH (both predictions: {python_pred})\n\n")
                    match_count += 1
                else:
                    max_diff = np.max(np.abs(python_scores - fpga_scores))
                    writer.write(f"                     | Status: ✗ MISMATCH (max diff: {max_diff}, Python pred: {python_pred}, FPGA pred: {fpga_pred})\n\n")
                    mismatch_count += 1
            
            # Progress indicator (every 10 images) - always to console
            if (i + 1) % 10 == 0 or (i + 1) == test_samples:
                progress_msg = (f"  Processed: {i+1:3d}/{test_samples} images | "
                              f"Matches: {match_count} | Mismatches: {mismatch_count} | Errors: {error_count}")
                # Only print to console if not already in console/both mode (to avoid duplication)
                if output_mode == 'file':
                    print(progress_msg)
        
        # Write summary
        writer.write("=" * 80 + "\n")
        writer.write("END OF COMPARISON\n")
        writer.write("=" * 80 + "\n")
    
    ser.close()
    
    # Final summary - always to console
    if output_mode != 'console':
        print()
        print("=" * 80)
        print(f"Comparison complete!")
        print(f"  Total images:  {test_samples}")
        print(f"  Matches:       {match_count}")
        print(f"  Mismatches:    {mismatch_count}")
        print(f"  Errors:        {error_count}")
        if output_mode in ['file', 'both']:
            print(f"  Output saved to: {output_path}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare Python simulation vs FPGA hardware inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save to default file (backward compatible)
  python compare_python_vs_fpga.py
  
  # Print to console only
  python compare_python_vs_fpga.py --output console
  
  # Save to file and print to console
  python compare_python_vs_fpga.py --output both
  
  # Custom file path
  python compare_python_vs_fpga.py --output file --file results.txt
  
  # Test more samples
  python compare_python_vs_fpga.py --samples 200 --output both
        """
    )
    
    parser.add_argument('--port', type=str, default=DEFAULT_COM_PORT,
                        help=f'Serial COM port (default: {DEFAULT_COM_PORT})')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD_RATE,
                        help=f'Baud rate (default: {DEFAULT_BAUD_RATE})')
    parser.add_argument('--samples', type=int, default=DEFAULT_TEST_SAMPLES,
                        help=f'Number of test samples (default: {DEFAULT_TEST_SAMPLES})')
    parser.add_argument('--output', type=str, 
                        choices=['file', 'console', 'both'], 
                        default='file',
                        help='Output mode: file (save to file), console (print), or both (default: file)')
    parser.add_argument('--file', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Output file path (default: {DEFAULT_OUTPUT_FILE})')
    
    args = parser.parse_args()
    
    # Run comparison with specified arguments
    run_comparison(
        com_port=args.port,
        baud_rate=args.baud,
        test_samples=args.samples,
        output_mode=args.output,
        output_file=args.file
    )

