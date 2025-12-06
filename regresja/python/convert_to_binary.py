"""
Universal converter: .mem files (hex text) -> binary files for UART transmission.

Auto-detects file type based on hex character count per line:
  - 2 hex chars -> 8-bit values (weights) -> 1 byte each
  - 8 hex chars -> 32-bit values (biases) -> 4 bytes each (little-endian)

Usage:
  python convert_to_binary.py [input_folder] [output_folder]
  
  If no paths given, defaults to:
    - Input:  regresja/outputs/mem/
    - Output: regresja/outputs/bin/
  
  Converts all .mem files found in the input folder.
"""

import os
import sys
import glob


def detect_bytes_per_value(mem_path):
    """
    Auto-detect if file contains 8-bit or 32-bit values.
    Returns: 1 for 8-bit, 4 for 32-bit, None if can't detect
    """
    with open(mem_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            hex_len = len(line)
            if hex_len == 2:
                return 1   # 8-bit weights
            elif hex_len == 8:
                return 4   # 32-bit biases
            else:
                print(f"  WARNING: {mem_path} has {hex_len} hex chars - unexpected format")
                return None
    
    return None


def convert_mem_to_bin(mem_path, bin_path, bytes_per_value):
    """
    Convert a .mem file to binary.
    
    Args:
        mem_path: Path to input .mem file
        bin_path: Path to output .bin file
        bytes_per_value: 1 for 8-bit weights, 4 for 32-bit biases
    
    Returns:
        (value_count, byte_count)
    """
    with open(mem_path, 'r') as f:
        lines = f.readlines()
    
    data = bytearray()
    value_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse hex value
        value = int(line, 16)
        value_count += 1
        
        # Convert to bytes (little-endian)
        if bytes_per_value == 1:
            # 8-bit value - handle as signed byte (two's complement already in file)
            data.append(value & 0xFF)
        else:
            # 32-bit value - little endian (LSB first)
            for i in range(bytes_per_value):
                data.append((value >> (i * 8)) & 0xFF)
    
    with open(bin_path, 'wb') as f:
        f.write(data)
    
    return value_count, len(data)


def main():
    # Default paths relative to this script's location (regresja/python/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "..", "outputs", "mem")
    default_output = os.path.join(script_dir, "..", "outputs", "bin")
    
    # Get directories from arguments or use defaults
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = default_input
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output
    
    # Make paths absolute
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .mem files
    mem_files = glob.glob(os.path.join(input_dir, "*.mem"))
    
    if not mem_files:
        print(f"No .mem files found in: {input_dir}")
        sys.exit(0)
    
    print(f"Input folder:  {input_dir}")
    print(f"Output folder: {output_dir}\n")
    
    total_bytes = 0
    converted = 0
    
    for mem_path in sorted(mem_files):
        mem_name = os.path.basename(mem_path)
        bin_name = mem_name.replace(".mem", ".bin")
        bin_path = os.path.join(output_dir, bin_name)
        
        # Auto-detect byte size
        bytes_per_value = detect_bytes_per_value(mem_path)
        
        if bytes_per_value is None:
            print(f"  SKIP: {mem_name} - could not detect format")
            continue
        
        # Convert
        values, size = convert_mem_to_bin(mem_path, bin_path, bytes_per_value)
        total_bytes += size
        converted += 1
        
        bit_size = bytes_per_value * 8
        print(f"  {mem_name:25} -> {bin_name:25} ({values:6} x {bit_size}-bit, {size:6} bytes)")
    
    print(f"\n{'='*60}")
    print(f"Converted: {converted} files")
    print(f"Total binary data: {total_bytes} bytes ({total_bytes / 1024:.2f} KB)")
    print(f"Transmission time at 9600 baud:   ~{total_bytes * 10 / 9600:.1f} seconds")
    print(f"Transmission time at 115200 baud: ~{total_bytes * 10 / 115200:.1f} seconds")


if __name__ == "__main__":
    main()

