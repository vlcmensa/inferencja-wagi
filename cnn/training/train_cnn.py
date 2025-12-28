import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 5           # Increased slightly for better convergence
LR = 0.001
INPUT_SCALE = 127.0  # Input mapped to range 0-127
SHIFT_CONV = 8       # Bit shift for Convolution output divisions
SHIFT_DENSE = 8      # Bit shift for Dense output divisions

# --- 1. Define Improved Model ---
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        # Block 1: Input (1x28x28) -> Conv (16x26x26) -> MaxPool (16x13x13)
        # Using 16 filters allows capturing more primitive shapes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        
        # Block 2: Input (16x13x13) -> Conv (32x11x11) -> MaxPool (32x5x5)
        # Using 32 filters allows combining primitives into complex digit parts
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten: 32 channels * 5 * 5 pixels = 800 features
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1) 
        
        # Dense
        x = self.fc(x)
        return x

# --- 2. Train ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Standard MNIST Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ImprovedCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Training ImprovedCNN on {device}...")
    model.train()
    
    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate rough accuracy for print
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} complete. Accuracy: {acc:.2f}%")
    
    return model

# --- 3. Helpers for .MEM export (Two's Complement Hex) ---
def save_mem_weights(filename, weights_int8):
    with open(filename, "w") as f:
        # Flatten ensures we write line by line
        for val in weights_int8.flatten():
            val = int(val)
            if val < 0: val += 256
            f.write(f"{val:02x}\n")

def save_mem_biases(filename, biases_int32):
    with open(filename, "w") as f:
        for val in biases_int32:
            val = int(val)
            if val < 0: val += (1 << 32)
            f.write(f"{val:08x}\n")

# --- 4. Export Quantized Weights ---
def export_weights(model):
    print("\n--- Starting Export ---")
    
    # Setup Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "..", "outputs")
    
    dirs = {
        "bin": os.path.join(outputs_dir, "bin"),
        "mem": os.path.join(outputs_dir, "mem"),
        "npy": os.path.join(outputs_dir, "npy")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # Save Normalization Params
    np.save(os.path.join(dirs["npy"], "norm_mean.npy"), np.array([0.1307]))
    np.save(os.path.join(dirs["npy"], "norm_std.npy"), np.array([0.3081]))

    # --- Extract Layers ---
    # Move to CPU and numpy
    c1_w = model.conv1.weight.data.cpu().numpy()
    c1_b = model.conv1.bias.data.cpu().numpy()
    
    c2_w = model.conv2.weight.data.cpu().numpy()
    c2_b = model.conv2.bias.data.cpu().numpy()
    
    fc_w = model.fc.weight.data.cpu().numpy()
    fc_b = model.fc.bias.data.cpu().numpy()

    print("Quantizing Layer 1 (Conv)...")
    # 1. Quantize Conv1
    # Scale weights so max value fits in 127
    w_scale_c1 = 127.0 / np.max(np.abs(c1_w))
    c1_w_int8 = np.clip(np.round(c1_w * w_scale_c1), -128, 127).astype(np.int8)
    
    # Bias scale = Input_Scale * Weight_Scale
    b_scale_c1 = INPUT_SCALE * w_scale_c1
    c1_b_int32 = np.round(c1_b * b_scale_c1).astype(np.int32)
    
    # Calculate Output Scale for Layer 1 (simulating the bit shift)
    # This scale represents the "real world value" of 1 unit in the activation map
    scale_out_c1 = (INPUT_SCALE * w_scale_c1) / (2**SHIFT_CONV)

    print("Quantizing Layer 2 (Conv)...")
    # 2. Quantize Conv2
    w_scale_c2 = 127.0 / np.max(np.abs(c2_w))
    c2_w_int8 = np.clip(np.round(c2_w * w_scale_c2), -128, 127).astype(np.int8)
    
    # Bias scale = Prev_Layer_Scale * Current_Weight_Scale
    b_scale_c2 = scale_out_c1 * w_scale_c2
    c2_b_int32 = np.round(c2_b * b_scale_c2).astype(np.int32)
    
    # Calculate Output Scale for Layer 2
    scale_out_c2 = (scale_out_c1 * w_scale_c2) / (2**SHIFT_CONV)

    print("Quantizing Layer 3 (Dense)...")
    # 3. Quantize FC
    w_scale_fc = 127.0 / np.max(np.abs(fc_w))
    fc_w_int8 = np.clip(np.round(fc_w * w_scale_fc), -128, 127).astype(np.int8)
    
    # Bias scale
    b_scale_fc = scale_out_c2 * w_scale_fc
    fc_b_int32 = np.round(fc_b * b_scale_fc).astype(np.int32)

    # --- SAVE FILES ---
    
    # Dictionary to iterate and save easily
    layers = [
        ("conv1", c1_w_int8, c1_b_int32),
        ("conv2", c2_w_int8, c2_b_int32),
        ("dense", fc_w_int8, fc_b_int32)
    ]

    for name, w, b in layers:
        # Binaries
        w.tofile(os.path.join(dirs["bin"], f"{name}_weights.bin"))
        b.tofile(os.path.join(dirs["bin"], f"{name}_biases.bin"))
        
        # Memory Files (Hex)
        save_mem_weights(os.path.join(dirs["mem"], f"{name}_weights.mem"), w)
        save_mem_biases(os.path.join(dirs["mem"], f"{name}_biases.mem"), b)
        
        print(f"Saved {name}: Weights {w.shape}, Biases {b.shape}")

    print("\nExport Complete.")
    print(f"Outputs saved to: {outputs_dir}")

if __name__ == "__main__":
    model = train()
    export_weights(model)