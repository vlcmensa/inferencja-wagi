"""
Exact FPGA Hardware Simulation with 32-bit Overflow

This script replicates the exact arithmetic operations performed by the FPGA hardware
INCLUDING 32-bit signed overflow (wrapping behavior).

This tests whether overflow in the 32-bit accumulator is causing the accuracy drop
from 90% (software) to 59.5% (hardware).

Hardware Implementation (from inference.v):
- Input pixels: 8-bit signed (-128 to 127)
- Weights: 8-bit signed (-128 to 127)
- Products: 16-bit signed (8-bit × 8-bit)
- Accumulator: 32-bit signed WITH OVERFLOW (wraps at ±2,147,483,648)
- Bias: 32-bit signed
- Final score: accumulator + bias (with overflow)
- Prediction: argmax(scores)
"""

import numpy as np
import os
import sys
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix

# Constants (must match training and hardware)
INPUT_SCALE = 127.0

def load_model_and_scaler():
    """Load trained weights, biases, and scaler parameters."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load weights (8-bit signed)
    W_raw = np.loadtxt(os.path.join(script_dir, '../outputs/mem/W.mem'), dtype=str)
    W = np.array([int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in W_raw])
    W = W.reshape(10, 784).astype(np.int8)  # 10 classes × 784 pixels
    
    # Load biases (32-bit signed)
    B_raw = np.loadtxt(os.path.join(script_dir, '../outputs/mem/B.mem'), dtype=str)
    B = np.array([int(x, 16) if int(x, 16) < 2**31 else int(x, 16) - 2**32 for x in B_raw])
    B = B.astype(np.int32)
    
    # Load scaler parameters
    mean = np.load(os.path.join(script_dir, '../outputs/npy/scaler_mean.npy'))
    scale = np.load(os.path.join(script_dir, '../outputs/npy/scaler_scale.npy'))
    
    return W, B, mean, scale


def preprocess_image_fpga(image_f32, mean, scale):
    """
    Preprocess image exactly like test_fpga_integration.py does.
    
    Steps:
    1. Normalize 0-255 to 0-1
    2. Apply standard scaler
    3. Quantize to int8 range with INPUT_SCALE
    4. Clip to [-128, 127]
    
    Args:
        image_f32: Raw pixel values (0-255)
        mean: Scaler mean
        scale: Scaler scale
    
    Returns:
        Preprocessed image as int8 array
    """
    # 1. Normalize to [0, 1]
    img_norm = image_f32 / 255.0
    
    # 2. Standard scaler
    img_scaled = (img_norm - mean) / scale
    
    # 3. Quantize to int8 range
    img_quant = np.round(img_scaled * INPUT_SCALE)
    img_int8 = np.clip(img_quant, -128, 127).astype(np.int8)
    
    return img_int8


def fpga_inference_with_overflow(input_pixels, weights, biases):
    """
    Simulate exact FPGA inference arithmetic WITH 32-bit overflow.
    
    Replicates the hardware pipeline from inference.v with overflow:
    - For each class (0-9):
      - accumulator = 0 (32-bit signed, WRAPS on overflow)
      - For each pixel (0-783):
        - product = weight[class][pixel] × input[pixel] (both int8)
        - product is 16-bit signed result
        - accumulator += sign_extend(product) to 32-bit WITH WRAPPING
      - score[class] = accumulator + bias[class] WITH WRAPPING
    - prediction = argmax(scores)
    
    Args:
        input_pixels: int8 array of shape (784,)
        weights: int8 array of shape (10, 784)
        biases: int32 array of shape (10,)
    
    Returns:
        predicted_digit (0-9), scores array
    """
    NUM_CLASSES = 10
    NUM_PIXELS = 784
    
    # Use int32 for scores - this will wrap on overflow
    scores = np.zeros(NUM_CLASSES, dtype=np.int32)
    
    # Track overflow occurrences
    overflow_detected = False
    
    # Process each class
    for class_idx in range(NUM_CLASSES):
        # Initialize accumulator (32-bit signed, will wrap on overflow)
        accumulator = np.int32(0)
        
        # Multiply-accumulate loop
        for pixel_idx in range(NUM_PIXELS):
            # Get weight and input (both 8-bit signed)
            weight = np.int16(weights[class_idx, pixel_idx])  # Cast to int16
            pixel = np.int16(input_pixels[pixel_idx])         # Cast to int16
            
            # Multiply: 8-bit × 8-bit → 16-bit signed
            product = weight * pixel  # This is 16-bit signed result
            
            # Sign-extend to 32-bit and add WITH OVERFLOW
            # np.int32() forces wrapping at 32-bit boundaries
            product_32 = np.int32(product)
            new_accumulator = np.int32(accumulator + product_32)
            
            # Check for overflow (for debugging purposes)
            # Overflow occurs if signs match but result sign differs
            if ((accumulator > 0 and product_32 > 0 and new_accumulator < 0) or
                (accumulator < 0 and product_32 < 0 and new_accumulator > 0)):
                overflow_detected = True
            
            accumulator = new_accumulator
        
        # Add bias (32-bit signed) WITH OVERFLOW
        final_score = np.int32(accumulator + biases[class_idx])
        scores[class_idx] = final_score
    
    # Argmax
    predicted_digit = np.argmax(scores)
    
    return predicted_digit, scores, overflow_detected


def load_mnist_test_data():
    """Load MNIST test dataset."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Use the last 10000 as test set (standard split)
    X_test = X[60000:]
    y_test = y[60000:].astype(int)
    
    return X_test, y_test


def main():
    """Main testing function."""
    print("=" * 70)
    print("FPGA Hardware Simulation - WITH 32-BIT OVERFLOW")
    print("=" * 70)
    print()
    print("This simulation uses 32-bit signed integers that wrap on overflow,")
    print("exactly matching the hardware's 2's complement arithmetic.")
    print()
    
    # Load model
    print("Loading model weights and scaler...")
    W, B, mean, scale = load_model_and_scaler()
    print(f"  Weights shape: {W.shape}, dtype: {W.dtype}")
    print(f"  Biases shape: {B.shape}, dtype: {B.dtype}")
    print(f"  Scaler mean shape: {mean.shape}")
    print(f"  Scaler scale shape: {scale.shape}")
    print()
    
    # Load test data
    X_test, y_test = load_mnist_test_data()
    print(f"Test data loaded: {len(X_test)} images")
    print()
    
    # Run inference on first 1000 test images
    print("Running FPGA simulation (with overflow) on first 1000 test images...")
    print("-" * 70)
    
    y_pred = []
    y_true = []
    overflow_count = 0
    
    num_test = 1000  # Test first 1000 images only
    
    for i in range(num_test):
        # Preprocess image
        img_preprocessed = preprocess_image_fpga(X_test[i], mean, scale)
        
        # Run FPGA inference with overflow
        pred, scores, had_overflow = fpga_inference_with_overflow(img_preprocessed, W, B)
        
        if had_overflow:
            overflow_count += 1
        
        y_pred.append(pred)
        y_true.append(y_test[i])
        
        # Display result immediately (matching test_fpga_integration.py format)
        print(f"Img {i:3d} | Label: {y_test[i]} | Pred: {pred:2d} | Scores: {scores.tolist()}")
    
    print("-" * 70)
    print()
    
    # Calculate final metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Overflow Events: {overflow_count}/{num_test} images ({overflow_count/num_test*100:.1f}%)")
    print()
    print("Confusion Matrix:")
    print("(Rows = True Labels, Columns = Predicted Labels)")
    print()
    print("     ", end="")
    for i in range(10):
        print(f"{i:5d}", end=" ")
    print()
    print("    " + "-" * 66)
    for i in range(10):
        print(f"{i:2d} | ", end="")
        for j in range(10):
            print(f"{conf_matrix[i, j]:5d}", end=" ")
        print()
    print()
    
    # Show some misclassifications
    print("Sample Misclassifications (first 10):")
    print("-" * 70)
    misclass_count = 0
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            misclass_count += 1
            if misclass_count <= 10:
                print(f"  Image {i:5d}: True={y_true[i]}, Predicted={y_pred[i]}")
            elif misclass_count > 10:
                break
    
    if misclass_count == 0:
        print("  No misclassifications found!")
    
    print()
    print("=" * 70)
    print(f"Total misclassifications: {misclass_count}/{len(y_true)}")
    print()
    print("INTERPRETATION:")
    if accuracy < 70:
        print("  → Low accuracy with overflow suggests overflow is the problem!")
        print("  → Hardware needs wider accumulators or better weight scaling.")
    elif overflow_count > 0:
        print(f"  → {overflow_count} overflow events occurred but accuracy is still good.")
        print("  → Overflow may not be the main issue.")
    else:
        print("  → No overflow detected - issue must be elsewhere.")
    print("=" * 70)


if __name__ == "__main__":
    main()

