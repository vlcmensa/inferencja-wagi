import serial
import time
import sys
import os
import struct
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix

# Import FPGA simulation functions for validation
from simulate_fpga_inference import (
    fpga_inference_with_overflow,
    load_model_and_scaler as load_sim_model
)

# Configuration
COM_PORT = 'COM3'     # Change this to your port
BAUD_RATE = 115200
TEST_SAMPLES = 1000    # Number of images to test
INPUT_SCALE = 127.0   # Must match training

# Protocol Constants
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])
DIGIT_READ_REQUEST = bytes([0xCC])
SCORES_READ_REQUEST = bytes([0xCD])


def load_data_and_scaler():
    """Load scaler params and MNIST test data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npy_dir = os.path.join(script_dir, "..", "outputs", "npy")
    
    # Load Scaler
    try:
        mean = np.load(os.path.join(npy_dir, "scaler_mean.npy"))
        scale = np.load(os.path.join(npy_dir, "scaler_scale.npy"))
    except FileNotFoundError:
        print("Error: Scaler files not found. Run training script first.")
        sys.exit(1)

    # Load MNIST (Cached by sklearn)
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Use the last 10000 as test set (standard split)
    X_test = X[60000:]
    y_test = y[60000:].astype(int)
    
    return X_test, y_test, mean, scale


def preprocess_image(image_f32, mean, scale):
    """Normalize, Scale, and Quantize exactly like the training script."""
    # 1. Normalize 0-255 to 0-1
    img_norm = image_f32 / 255.0
    
    # 2. Standard Scaler
    img_scaled = (img_norm - mean) / scale
    
    # 3. Quantize to int8 range
    img_quant = np.round(img_scaled * INPUT_SCALE)
    img_int8 = np.clip(img_quant, -128, 127).astype(np.int8)
    
    # 4. View as uint8 for UART transmission (Preserves 2's complement bits)
    return img_int8.view(np.uint8)


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


def run_integration_test():
    X_test, y_test, mean, scale = load_data_and_scaler()
    
    # Pick random samples or first N samples
    indices = range(TEST_SAMPLES)
    
    print(f"\nStarting Integration Test on {TEST_SAMPLES} images...")
    print(f"Port: {COM_PORT}, Baud: {BAUD_RATE}")
    print("-" * 60)
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(1) # Allow reset
        ser.reset_input_buffer()
        
        y_pred = []
        y_true = []
        
        for i, idx in enumerate(indices):
            target_label = y_test[idx]
            y_true.append(target_label)
            
            # Prepare data
            img_data = preprocess_image(X_test[idx], mean, scale)
            
            # 1. Send Image
            ser.write(IMG_START_MARKER)
            ser.write(img_data.tobytes())
            ser.write(IMG_END_MARKER)
            ser.flush()
            
            # 2. Wait for Inference (80us is fast, but UART is slow. 0.05s is safe)
            time.sleep(0.05)
            
            # 3. Request Result
            ser.reset_input_buffer()
            ser.write(DIGIT_READ_REQUEST)
            
            # 4. Read Result
            resp = ser.read(1)
            
            if len(resp) != 1:
                print(f"Error: Timeout on image {i}")
                y_pred.append(-1)
                scores_str = "ERROR"
            else:
                pred = int.from_bytes(resp, byteorder='little') & 0x0F
                y_pred.append(pred)
                
                # 5. Read Class Scores
                time.sleep(0.01)
                scores = read_scores_from_fpga(ser)
                
                if scores is None:
                    scores_str = "ERROR"
                else:
                    scores_str = str(scores.tolist())
                
            # Display with scores
            print(f"Img {i:3d} | Label: {target_label} | Pred: {y_pred[-1]:2d} | Scores: {scores_str}")

        ser.close()
        
        # Final Stats
        acc = accuracy_score(y_true, y_pred)
        print("-" * 60)
        print(f"Final Accuracy: {acc * 100:.2f}%")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
    except serial.SerialException as e:
        print(f"Serial Error: {e}")


def run_detailed_test_with_scores(num_samples=10):
    """Run a detailed test that reads and displays class scores for analysis.
    
    Args:
        num_samples: Number of images to test with detailed score output
    """
    X_test, y_test, mean, scale = load_data_and_scaler()
    
    indices = range(num_samples)
    
    print(f"\nStarting Detailed Test with Scores on {num_samples} images...")
    print(f"Port: {COM_PORT}, Baud: {BAUD_RATE}")
    print("=" * 80)
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(1)
        ser.reset_input_buffer()
        
        for i, idx in enumerate(indices):
            target_label = y_test[idx]
            img_data = preprocess_image(X_test[idx], mean, scale)
            
            # Send Image
            ser.write(IMG_START_MARKER)
            ser.write(img_data.tobytes())
            ser.write(IMG_END_MARKER)
            ser.flush()
            
            time.sleep(0.05)
            
            # Read predicted digit
            ser.reset_input_buffer()
            ser.write(DIGIT_READ_REQUEST)
            resp = ser.read(1)
            
            if len(resp) != 1:
                print(f"Error: Timeout on image {i}")
                continue
                
            pred = int.from_bytes(resp, byteorder='little') & 0x0F
            
            # Read all class scores
            time.sleep(0.01)  # Small delay between requests
            scores = read_scores_from_fpga(ser)
            
            if scores is None:
                print(f"Error: Failed to read scores for image {i}")
                continue
            
            # Display results
            print(f"\nImage {i} (Index {idx}):")
            print(f"  True Label:      {target_label}")
            print(f"  Predicted:       {pred}")
            print(f"  Status:          {'✓ CORRECT' if pred == target_label else '✗ WRONG'}")
            print(f"  Class Scores:")
            
            # Show scores with highlighting
            for class_idx in range(10):
                score = scores[class_idx]
                marker = " <-- PREDICTED" if class_idx == pred else ""
                marker += " (TRUE)" if class_idx == target_label else ""
                print(f"    Class {class_idx}: {score:12d}{marker}")
            
            # Show score differences
            max_score = np.max(scores)
            second_max = np.partition(scores, -2)[-2]
            confidence = max_score - second_max
            print(f"  Confidence Margin: {confidence} (max - 2nd_max)")
            print("-" * 80)
        
        ser.close()
        
    except serial.SerialException as e:
        print(f"Serial Error: {e}")


def run_validated_test_with_scores(num_samples=10):
    """Run hardware validation test comparing FPGA scores with software simulation.
    
    This mode verifies that the FPGA computes the exact same scores as the
    software simulation (including 32-bit overflow behavior).
    
    Args:
        num_samples: Number of images to test with score validation
    """
    # Load MNIST data and scaler
    X_test, y_test, mean, scale = load_data_and_scaler()
    
    # Load weights and biases for simulation
    print("Loading model for simulation...")
    W, B, sim_mean, sim_scale = load_sim_model()
    
    # Verify scalers match
    if not np.allclose(mean, sim_mean):
        print("WARNING: Scaler mean mismatch between test and simulation!")
    if not np.allclose(scale, sim_scale):
        print("WARNING: Scaler scale mismatch between test and simulation!")
    
    print(f"\nStarting Validated Test on {num_samples} images...")
    print(f"Port: {COM_PORT}, Baud: {BAUD_RATE}")
    print("=" * 80)
    print("Comparing FPGA hardware scores with software simulation...")
    print("=" * 80)
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(1)
        ser.reset_input_buffer()
        
        pass_count = 0
        fail_count = 0
        overflow_count = 0
        
        for i in range(num_samples):
            target_label = y_test[i]
            
            # Preprocess image
            img_data = preprocess_image(X_test[i], mean, scale)
            
            # Send to FPGA
            ser.write(IMG_START_MARKER)
            ser.write(img_data.tobytes())
            ser.write(IMG_END_MARKER)
            ser.flush()
            time.sleep(0.05)
            
            # Read FPGA predicted digit
            ser.reset_input_buffer()
            ser.write(DIGIT_READ_REQUEST)
            resp = ser.read(1)
            if len(resp) != 1:
                print(f"Error: Timeout on image {i}")
                fail_count += 1
                continue
            fpga_pred = int.from_bytes(resp, byteorder='little') & 0x0F
            
            # Read FPGA scores
            time.sleep(0.01)
            fpga_scores = read_scores_from_fpga(ser)
            if fpga_scores is None:
                print(f"Error: Failed to read FPGA scores for image {i}")
                fail_count += 1
                continue
            
            # Compute reference scores using simulation
            # Convert uint8 view back to int8 for simulation
            img_int8 = img_data.view(np.int8)
            ref_pred, ref_scores, had_overflow = fpga_inference_with_overflow(
                img_int8, W, B
            )
            
            if had_overflow:
                overflow_count += 1
            
            # VALIDATION: Compare scores
            scores_match = np.array_equal(fpga_scores, ref_scores)
            max_error = np.max(np.abs(fpga_scores - ref_scores))
            pred_match = (fpga_pred == ref_pred)
            
            if scores_match and pred_match:
                pass_count += 1
                status = "✓ PASS"
            else:
                fail_count += 1
                status = "✗ FAIL"
            
            # Display results
            print(f"\nImage {i}:")
            print(f"  True Label:       {target_label}")
            print(f"  FPGA Predicted:   {fpga_pred} {'✓' if fpga_pred == target_label else '✗'}")
            print(f"  Sim Predicted:    {ref_pred} {'✓' if ref_pred == target_label else '✗'}")
            print(f"  Pred Match:       {'✓ YES' if pred_match else '✗ NO'}")
            print(f"  Scores Match:     {'✓ YES' if scores_match else '✗ NO'}")
            print(f"  Max Score Error:  {max_error}")
            print(f"  Overflow Flag:    {'YES' if had_overflow else 'NO'}")
            print(f"  Overall:          {status}")
            
            # If mismatch, show detailed comparison
            if not scores_match or not pred_match:
                print(f"\n  ⚠️  MISMATCH DETECTED - Detailed Score Comparison:")
                print(f"  Class | FPGA Score  | Sim Score   | Difference  | Notes")
                print(f"  ------|-------------|-------------|-------------|------------------")
                for c in range(10):
                    diff = fpga_scores[c] - ref_scores[c]
                    notes = []
                    if c == fpga_pred:
                        notes.append("FPGA pred")
                    if c == ref_pred:
                        notes.append("SIM pred")
                    if c == target_label:
                        notes.append("TRUE")
                    if diff != 0:
                        notes.append("⚠️ ERROR")
                    notes_str = ", ".join(notes) if notes else ""
                    print(f"    {c}   | {fpga_scores[c]:11d} | {ref_scores[c]:11d} | "
                          f"{diff:11d} | {notes_str}")
            
            print("-" * 80)
        
        ser.close()
        
        # Summary
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Images:     {num_samples}")
        print(f"Passed:           {pass_count} ({pass_count/num_samples*100:.1f}%)")
        print(f"Failed:           {fail_count} ({fail_count/num_samples*100:.1f}%)")
        print(f"Overflow Events:  {overflow_count} ({overflow_count/num_samples*100:.1f}%)")
        print(f"{'='*80}")
        
        if fail_count == 0:
            print("\n✓ SUCCESS: All scores match perfectly!")
            print("  Hardware is computing the exact same values as the simulation.")
            print("  FPGA implementation is correct.")
        else:
            print(f"\n✗ FAILURE: {fail_count} mismatches detected!")
            print("  Possible causes:")
            print("  - Pipeline bug in inference.v")
            print("  - Sign extension error")
            print("  - Accumulator overflow handling")
            print("  - Bias addition error")
            print("  - Memory addressing issue")
            print("  Review the detailed comparisons above.")
        
        if overflow_count > 0:
            print(f"\nℹ️  Note: {overflow_count} images triggered 32-bit overflow.")
            print("   This is expected behavior and handled correctly if scores match.")
        
        print(f"{'='*80}")
        
    except serial.SerialException as e:
        print(f"Serial Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FPGA inference with MNIST')
    parser.add_argument('--mode', 
                        choices=['quick', 'detailed', 'validate'], 
                        default='quick',
                        help='Test mode: quick (1000 images, accuracy only), '
                             'detailed (view scores), validate (verify scores vs simulation)')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples for detailed/validate modes (default: 10)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_integration_test()
    elif args.mode == 'detailed':
        run_detailed_test_with_scores(num_samples=args.samples)
    else:  # validate
        run_validated_test_with_scores(num_samples=args.samples)
