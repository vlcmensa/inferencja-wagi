import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values to [0, 1]
y = y.astype(int)

# Split into train and test
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

# Scale data with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 2. TRAIN SOFTMAX REGRESSION
# ============================================================
print("Training softmax regression...")
model = LogisticRegression(max_iter=200, solver='lbfgs')
model.fit(X_train_scaled, y_train)

# Test accuracy with float model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Float model accuracy: {accuracy * 100:.2f}%")

# Extract weights and biases
W = model.coef_  # Shape: (10, 784)
b = model.intercept_  # Shape: (10,)

# ============================================================
# 3. QUANTIZATION WITH DYNAMIC SCALING
# ============================================================
print("\nQuantizing weights and biases...")

# Input scale (inputs will be quantized to int8 range)
INPUT_SCALE = 127.0

# Quantize weights with dynamic scale to use full int8 range
max_abs_W = np.max(np.abs(W))
W_SCALE = 127.0 / max_abs_W
W_int8 = np.clip(np.round(W * W_SCALE), -127, 127).astype(np.int8)

# Quantize biases: scale = weight_scale * input_scale
BIAS_SCALE = W_SCALE * INPUT_SCALE
b_int32 = np.round(b * BIAS_SCALE).astype(np.int32)

print(f"Weight range: [{W.min():.4f}, {W.max():.4f}]")
print(f"Weight scale factor: {W_SCALE:.4f}")
print(f"Quantized weight range: [{W_int8.min()}, {W_int8.max()}]")
print(f"Bias scale factor: {BIAS_SCALE:.4f}")
print(f"Quantized bias range: [{b_int32.min()}, {b_int32.max()}]")

# ============================================================
# 4. VERIFY QUANTIZED MODEL ACCURACY
# ============================================================
def predict_quantized(X, W_q, b_q):
    """Simulate FPGA inference with quantized weights."""
    X_int = np.round(X * INPUT_SCALE).astype(np.int32)
    logits = X_int @ W_q.T.astype(np.int32) + b_q
    return np.argmax(logits, axis=1)

y_pred_quant = predict_quantized(X_test_scaled, W_int8, b_int32)
acc_quant = accuracy_score(y_test, y_pred_quant)
print(f"Quantized model accuracy: {acc_quant * 100:.2f}%")
print(f"Accuracy loss from quantization: {(accuracy - acc_quant) * 100:.2f}%")

# ============================================================
# 5. SAVE TO .MEM FILES (PROPER TWO'S COMPLEMENT HEX)
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "outputs", "mem")
data_dir = os.path.join(script_dir, "..", "data")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print("\nSaving weights and biases to .mem files...")

# Save weights (int8 as 2-digit hex, two's complement)
with open(os.path.join(output_dir, "W.mem"), "w") as f:
    for i in range(10):  # 10 classes
        for j in range(784):  # 784 pixels
            val = int(W_int8[i, j])
            # Convert negative to two's complement (add 256)
            if val < 0:
                val = val + 256
            f.write(f"{val:02x}\n")

# Save biases (int32 as 8-digit hex, two's complement)
with open(os.path.join(output_dir, "B.mem"), "w") as f:
    for i in range(10):  # 10 biases
        val = int(b_int32[i])
        # Convert negative to two's complement (add 2^32)
        if val < 0:
            val = val + (1 << 32)
        f.write(f"{val:08x}\n")

# Save scale info for reference
with open(os.path.join(data_dir, "scale_info.txt"), "w") as f:
    f.write(f"Weight scale factor: {W_SCALE}\n")
    f.write(f"Input scale factor: {INPUT_SCALE}\n")
    f.write(f"Bias scale factor: {BIAS_SCALE}\n")
    f.write(f"\nOriginal weight range: [{W.min()}, {W.max()}]\n")
    f.write(f"Original bias range: [{b.min()}, {b.max()}]\n")
    f.write(f"\nQuantized model accuracy: {acc_quant * 100:.2f}%\n")

# Save StandardScaler parameters for send_image.py
np.save(os.path.join(data_dir, "scaler_mean.npy"), scaler.mean_)
np.save(os.path.join(data_dir, "scaler_scale.npy"), scaler.scale_)

print(f"Generated: {output_dir}/W.mem, B.mem")
print(f"Generated: {data_dir}/scale_info.txt, scaler_mean.npy, scaler_scale.npy")
print("\nDone!")