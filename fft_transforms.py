#!/usr/bin/env python3
"""
Simplified FFT Image Processing using numpy real FFT functions
Usage: python fft_transforms.py <input_path> <output_dir> [--truncate-factor 0.3] [--split train]
"""

import os
import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

def load_mnist(split="train", max_images=None):
    if not DATASETS_AVAILABLE:
        raise ImportError("pip install datasets pillow")
    
    dataset = load_dataset("ylecun/mnist", split=split)
    if max_images:
        dataset = dataset.select(range(min(max_images, len(dataset))))
    
    return np.array([np.array(item['image'], dtype=np.float32) / 255.0 
                     for item in dataset])

def truncate_fft(fft_data, factor=0.3):
    h, w = fft_data.shape
    copied_fft = fft_data.copy()
    
    center_h = math.ceil(h * factor)
    center_w = math.ceil(w * factor)
    
    if center_h % 2 == 0:
        center_h += 1
    if center_w % 2 == 0:
        center_w += 1
    
    top = math.ceil((h - center_h) / 2)
    left = math.ceil((w - center_w) / 2)
    bottom = top + center_h
    right = left + center_w

    return copied_fft[top:bottom, left:right]

def untruncate_fft(truncated, original_shape):
    """Untruncate FFT data back to original shape"""
    h, w = original_shape
    padded = np.zeros(original_shape, dtype=np.complex64)
    
    th, tw = truncated.shape
    top = math.ceil((h - th) / 2)
    left = math.ceil((w - tw) / 2)
    
    padded[top:top + th, left:left + tw] = truncated
    return padded

def remove_redundancies(redundant):
    """Keep just the fild vertical half of the 2d array including axis if odd"""
    if redundant.ndim != 2:
        raise ValueError("Input must be a 2D array")
    _, w = redundant.shape
    if w % 2 == 0:
        return redundant[:, :w // 2]
    else:
        return redundant[:, :w // 2 + 1]
    
def restore_hermitian(truncated, original_shape):
    h, w = original_shape
    restored = np.zeros((h, w), dtype=truncated.dtype)
    restored[:, :truncated.shape[1]] = truncated
    
    truncated_w = truncated.shape[1]
    offset = 0 if w % 2 == 0 else -1 
    start_j = truncated_w + offset
    for j in range(1, start_j+1):
        restored[:, w - j] = np.conj(restored[::-1, j-1])
    
    return restored

def pad_rfft(truncated, original_shape, fill_value=0):
    """Pad truncated rfft back to original shape"""
    h, w = original_shape
    padded = np.full((h, w), fill_value, dtype=truncated.dtype)
    th, tw = truncated.shape
    padded[:th, :tw] = truncated
    return padded

def serialize_complex(data, scale, orig_shape, trunc_factor, filename, precision= 10):
    """Serialize complex array to JSON"""
    serialized = {
        "data": [
            [
                [
                    round(float(np.real(val)), precision),
                    round(float(np.imag(val)), precision),
                ] for val in row
            ] for row in data
        ],
        "scale": float(scale),
        "original_shape": orig_shape,
        "truncate_factor": trunc_factor
    }
    with open(filename, 'w') as f:
        json.dump(serialized, f, separators=(',', ':'))

def load_complex(filename):
    """Load complex array from JSON"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    complex_array = np.array([[complex(real, imag) for real, imag in row] 
                             for row in data["data"]])
    return complex_array, data["scale"], data["original_shape"], data["truncate_factor"]

def process_image(image, output_dir, idx, truncate_factor=0.3):
    name = f"image_{idx:04d}"
    
    # Forward transform
    fft_data = np.fft.fftshift(np.fft.fftn(image, norm='forward'))
    rfft_data = np.fft.rfftn(image)
    
    # Truncate
    truncated = truncate_fft(fft_data, truncate_factor)

    # Verify untruncation
    untruncated = untruncate_fft(truncated, fft_data.shape)
    pre_reconstructured = np.fft.ifftn(np.fft.ifftshift(untruncated), norm='forward')

    non_redundant = remove_redundancies(truncated)
    
    # Normalize
    max_mag = np.abs(non_redundant).max()
    scale = max_mag if max_mag > 0 else 1.0
    normalized = non_redundant / scale
    serializable = normalized
    
    # Serialize
    json_path = os.path.join(output_dir, f"{name}.json")
    serialize_complex(serializable, scale, fft_data.shape, truncate_factor, json_path, precision=1)
    
    # Load and Reconstruct
    loaded_norm, loaded_scale, loaded_shape, _ = load_complex(json_path)
    loaded_norm = loaded_norm * loaded_scale
    restored_hermitian = restore_hermitian(loaded_norm, truncated.shape)
    restored_untruncated = untruncate_fft(restored_hermitian, loaded_shape)
    restored_unfft = np.fft.ifftn(np.fft.ifftshift(restored_untruncated), norm='forward')
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    
    # Original image and its transform
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(np.real(fft_data), cmap='RdBu')
    axes[1, 0].set_title('Original FFT Real')
    ny, nx = np.real(fft_data).shape
    axes[1, 0].axhline(y=ny / 2, color='k', linestyle='--', linewidth=1)
    axes[1, 0].axvline(x=nx / 2, color='k', linestyle='--', linewidth=1) 
    
    axes[2, 0].imshow(np.imag(fft_data), cmap='RdBu')
    axes[2, 0].set_title('Original FFT Imag')
    ny, nx = np.real(fft_data).shape
    axes[2, 0].axhline(y=ny / 2, color='k', linestyle='--', linewidth=1)
    axes[2, 0].axvline(x=nx / 2, color='k', linestyle='--', linewidth=1) 

    axes[0, 1].imshow(np.real(pre_reconstructured), cmap='gray')
    axes[0, 1].set_title('Truncated Image')
    axes[0, 1].axis('off')

    axes[1, 1].imshow(np.real(truncated), cmap='RdBu')
    axes[1, 1].set_title('Truncated Real')
    ny, nx = np.real(truncated).shape
    axes[1, 1].axhline(y=ny / 2, color='k', linestyle='--', linewidth=1)
    axes[1, 1].axvline(x=nx / 2, color='k', linestyle='--', linewidth=1) 
    
    axes[2, 1].imshow(np.imag(truncated), cmap='RdBu')
    axes[2, 1].set_title('Truncated Imag')
    ny, nx = np.real(truncated).shape
    axes[2, 1].axhline(y=ny / 2, color='k', linestyle='--', linewidth=1)
    axes[2, 1].axvline(x=nx / 2, color='k', linestyle='--', linewidth=1) 

    axes[0, 2].imshow(image, cmap='gray')
    axes[0, 2].set_title('Original Image')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(np.real(non_redundant), cmap='RdBu')
    axes[1, 2].set_title('Non Redundant Real')
    
    axes[2, 2].imshow(np.imag(non_redundant), cmap='RdBu')
    axes[2, 2].set_title('Non Redundant Imag')
    
    axes[0, 3].imshow(np.real(restored_unfft), cmap='gray')
    axes[0, 3].set_title('Json Reconstructed Image')
    axes[0, 3].axis('off')

    axes[1, 3].imshow(np.real(rfft_data), cmap='RdBu')
    axes[1, 3].set_title('Json RFFT Real')
    
    axes[2, 3].imshow(np.imag(rfft_data), cmap='RdBu')
    axes[2, 3].set_title('Json RFFT Imag')



    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="FFT Image Processing with rfftn/irfftn")
    parser.add_argument("input_path", help="Image index or 'all'")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--truncate-factor", type=float, default=0.3, help="Fraction to keep (0-1)")
    parser.add_argument("--max-images", type=int, default=10, help="Max images when using 'all'")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split")
    
    args = parser.parse_args()
    
    if not DATASETS_AVAILABLE:
        print("Error: pip install datasets pillow")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        max_load = 1000 if args.input_path.lower() == "all" else 100
        images = load_mnist(args.split, max_load)
        print(f"Loaded {len(images)} images")
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        return
    
    if args.input_path.lower() == "all":
        num_process = min(args.max_images, len(images))
        for i in range(num_process):
            process_image(images[i], args.output_dir, i, args.truncate_factor)
    else:
        try:
            idx = int(args.input_path)
            if idx >= len(images):
                print(f"Index {idx} out of range")
                return
            process_image(images[idx], args.output_dir, idx, args.truncate_factor)
        except ValueError:
            print("Invalid input path")
            return
    
    print(f"Results in {args.output_dir}")

if __name__ == "__main__":
    main()