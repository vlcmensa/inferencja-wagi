import serial
import time
import sys
import os
import argparse
import numpy as np

# Configuration
DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200
BIN_DIR = "../outputs/bin"

# Protocol Markers (Must match uart_router.v)
# Defined as 1D arrays (lists) to prevent concatenation errors
START_MARKER = np.array([0xAA, 0x55], dtype=np.uint8)
END_MARKER   = np.array([0x55, 0xAA], dtype=np.uint8)

def load_files(base_path):
    """Load weights from binary files into numpy arrays."""
    files = {
        "conv_w": "conv_weights.bin",
        "conv_b": "conv_biases.bin",
        "dense_w": "dense_weights.bin",
        "dense_b": "dense_biases.bin"
    }
    
    data_chunks = []
    total_bytes = 0
    
    print("Loading weight files...")
    
    # 1. Conv Weights
    path = os.path.join(base_path, files["conv_w"])
    if not os.path.exists(path): raise FileNotFoundError(f"{path} missing")
    cw = np.fromfile(path, dtype=np.uint8)
    data_chunks.append(cw)
    print(f"  Conv Weights: {cw.size} bytes")
    
    # 2. Conv Biases
    path = os.path.join(base_path, files["conv_b"])
    cb = np.fromfile(path, dtype=np.uint8)
    data_chunks.append(cb)
    print(f"  Conv Biases: {cb.size} bytes")
    
    # 3. Dense Weights
    path = os.path.join(base_path, files["dense_w"])
    dw = np.fromfile(path, dtype=np.uint8)
    data_chunks.append(dw)
    print(f"  Dense Weights: {dw.size} bytes")
    
    # 4. Dense Biases
    path = os.path.join(base_path, files["dense_b"])
    db = np.fromfile(path, dtype=np.uint8)
    data_chunks.append(db)
    print(f"  Dense Biases: {db.size} bytes")
    
    # Check total size logic (excluding markers)
    payload_size = sum(c.size for c in data_chunks)
    print(f"  Total Payload: {payload_size} bytes")
    
    return data_chunks, payload_size

def send_weights(port, baud):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(script_dir, BIN_DIR)
    
    # Load data chunks
    chunks, payload_size = load_files(bin_path)
    
    # Concatenate everything: Start Marker + Data + End Marker
    # This is where the fix is: START_MARKER is now strictly 1D
    all_data = np.concatenate([START_MARKER] + chunks + [END_MARKER])
    
    total_packet_size = all_data.size
    print(f"  Total Transmission (w/ markers): {total_packet_size} bytes")

    # Connect to Serial
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(1) # Allow FPGA reset
    except serial.SerialException as e:
        print(f"Error opening port {port}: {e}")
        return 1

    print(f"Sending to {port}...")
    start_time = time.time()
    
    # Send in chunks with flow control
    CHUNK_SIZE = 64
    bytes_sent = 0
    
    # Convert numpy array to bytes for pyserial
    raw_bytes = all_data.tobytes()
    
    for i in range(0, len(raw_bytes), CHUNK_SIZE):
        chunk = raw_bytes[i : i + CHUNK_SIZE]
        ser.write(chunk)
        bytes_sent += len(chunk)
        
        # Progress Bar
        progress = (bytes_sent / total_packet_size) * 100
        sys.stdout.write(f"\rProgress: {progress:.1f}% ({bytes_sent}/{total_packet_size})")
        sys.stdout.flush()
        
        # Flow Control: Give FPGA time to process BRAM writes
        time.sleep(0.005) 

    print("\nDone.")
    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.2f}s")
    
    ser.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial Port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud Rate")
    args = parser.parse_args()
    
    sys.exit(send_weights(args.port, args.baud))