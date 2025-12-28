import os
import numpy as np
import sys
import torch
from torchvision import datasets, transforms

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "bin")
NPY_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "npy")

# Model Constants
INPUT_SCALE = 127.0
SHIFT_CONV = 8     # Must match the value used in your export script
SHIFT_DENSE = 8    # Must match the value used in your export script

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def relu(x):
    return np.maximum(0, x)

def max_pool_2x2(input_vol):
    """
    Simulates 2x2 Max Pooling with Stride 2.
    Input: (Channels, Height, Width)
    Output: (Channels, Height/2, Width/2)
    """
    c, h, w = input_vol.shape
    new_h, new_w = h // 2, w // 2
    output_vol = np.zeros((c, new_h, new_w), dtype=np.int32)
    
    for ch in range(c):
        for r in range(new_h):
            for col in range(new_w):
                # Extract 2x2 window
                # Rows: 2*r to 2*r+2, Cols: 2*col to 2*col+2
                window = input_vol[ch, r*2 : r*2+2, col*2 : col*2+2]
                output_vol[ch, r, col] = np.max(window)
                
    return output_vol

def convolve_layer(input_vol, weights, biases, shift):
    """
    Standard Multi-Channel Convolution.
    Input: (In_Channels, H, W)
    Weights: (Out_Channels, In_Channels, 3, 3)
    Output: (Out_Channels, H-2, W-2)
    """
    in_ch, h, w = input_vol.shape
    out_ch, _, k_h, k_w = weights.shape
    out_h, out_w = h - 2, w - 2
    
    output_vol = np.zeros((out_ch, out_h, out_w), dtype=np.int32)

    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(out_h):
            for c in range(out_w):
                acc = 0
                # Sum over all input channels
                for ch in range(in_ch):
                    window = input_vol[ch, r:r+3, c:c+3]
                    w_kernel = weights[f, ch]
                    acc += np.sum(window * w_kernel)
                
                # Add Bias
                acc += bias_val
                
                # FPGA Pipeline Steps:
                acc = acc >> shift            # 1. Arithmetic Right Shift
                if acc < 0: acc = 0           # 2. ReLU
                if acc > 127: acc = 127       # 3. Saturation
                
                output_vol[f, r, c] = acc
                
    return output_vol

# ==========================================
# 3. BIT-EXACT INFERENCE ENGINE
# ==========================================
def simulate_quantized_inference(image_bytes, weights_dict):
    # Unpack weights
    c1_w, c1_b = weights_dict['c1']
    c2_w, c2_b = weights_dict['c2']
    dw, db     = weights_dict['dense']
    
    # 0. Load Image
    img = np.frombuffer(image_bytes, dtype=np.int8)
    # Shape: (1, 28, 28) for consistent 3D processing
    img_3d = img.reshape(1, 28, 28).astype(np.int32)
    
    # --- LAYER 1: Conv 16x3x3 ---
    # Reshape weights: (16 filters, 1 channel, 3, 3)
    c1_w_reshaped = c1_w.reshape(16, 1, 3, 3).astype(np.int32)
    # Conv -> ReLU -> Saturation
    x = convolve_layer(img_3d, c1_w_reshaped, c1_b, SHIFT_CONV)
    # Pooling: 26x26 -> 13x13
    x = max_pool_2x2(x)

    # --- LAYER 2: Conv 32x3x3 ---
    # Reshape weights: (32 filters, 16 channels, 3, 3)
    c2_w_reshaped = c2_w.reshape(32, 16, 3, 3).astype(np.int32)
    # Conv -> ReLU -> Saturation
    x = convolve_layer(x, c2_w_reshaped, c2_b, SHIFT_CONV)
    # Pooling: 11x11 -> 5x5
    x = max_pool_2x2(x)
    
    # --- LAYER 3: Dense ---
    # Flatten: (32, 5, 5) -> 800
    flattened = x.flatten().astype(np.int32)
    
    # Reshape Dense: (10, 800)
    # Note: 800 comes from 32 channels * 5 * 5 pixels
    dw_reshaped = dw.reshape(10, 800).astype(np.int32)
    
    scores = np.zeros(10, dtype=np.int32)
    for c in range(10):
        dot_prod = np.dot(flattened, dw_reshaped[c])
        scores[c] = dot_prod + db[c]
        
    return np.argmax(scores)

# ==========================================
# 4. UTILITIES (LOADERS)
# ==========================================
def load_all_weights():
    try:
        # Load Layer 1
        c1_w = np.fromfile(os.path.join(BIN_DIR, "conv1_weights.bin"), dtype=np.int8)
        c1_b = np.fromfile(os.path.join(BIN_DIR, "conv1_biases.bin"), dtype=np.int32)
        
        # Load Layer 2
        c2_w = np.fromfile(os.path.join(BIN_DIR, "conv2_weights.bin"), dtype=np.int8)
        c2_b = np.fromfile(os.path.join(BIN_DIR, "conv2_biases.bin"), dtype=np.int32)
        
        # Load Dense
        dw = np.fromfile(os.path.join(BIN_DIR, "dense_weights.bin"), dtype=np.int8)
        db = np.fromfile(os.path.join(BIN_DIR, "dense_biases.bin"), dtype=np.int32)
        
        mean = np.load(os.path.join(NPY_DIR, "norm_mean.npy"))
        std  = np.load(os.path.join(NPY_DIR, "norm_std.npy"))
        
        return {
            'c1': (c1_w, c1_b),
            'c2': (c2_w, c2_b),
            'dense': (dw, db)
        }, (mean, std)
        
    except FileNotFoundError as e:
        sys.exit(f"Error: Missing binary file. {e}\nDid you run the updated training script?")

def preprocess(image_tensor, mean, std):
    x = image_tensor.numpy().squeeze()
    x = (x - mean) / std
    x = np.clip(np.round(x * INPUT_SCALE), -128, 127).astype(np.int8)
    return x.tobytes()

def get_data():
    import logging
    logging.getLogger("torchvision").setLevel(logging.CRITICAL)
    transform = transforms.Compose([transforms.ToTensor()])
    data_root = os.path.join(SCRIPT_DIR, "..", "..", "data")
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    finally:
        sys.stdout = old_stdout
    return dataset

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    print("Loading weights...")
    weights, (norm_mean, norm_std) = load_all_weights()
    dataset = get_data()
    
    total_images = 100  # Set lower for speed (pure Python loops are slow)
    correct = 0

    print(f"Running inference on first {total_images} images...")
    
    for i in range(total_images):
        img_tensor, label = dataset[i]
        img_bytes = preprocess(img_tensor, norm_mean, norm_std)
        
        prediction = simulate_quantized_inference(img_bytes, weights)
        
        if prediction == label:
            correct += 1
            
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} images...")

    accuracy = (correct / total_images) * 100
    print(f"Accuracy of quantized model: {accuracy:.2f}%")

if __name__ == "__main__":
    main()