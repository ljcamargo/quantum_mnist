#!/usr/bin/env python3
"""
Simplified FFT Image Processing using numpy real FFT functions
Usage: python fft_transforms.py <input_path> <output_dir> [--truncate-factor 0.3] [--split train]
"""

import os
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

    data = [
        [np.array(item['image'], dtype=np.float32) / 255.0, item['label']]
        for item in dataset
    ]
    return data

def in_circle(x: int, y: int, shape: tuple[int, int]) -> bool:
    """
    Check if a pixel (x,y) lies inside or on the rim of a circle
    whose diameter equals the side length of the square array.

    Args:
        x, y : int
            Pixel coordinates (0-indexed).
        shape : tuple[int,int]
            Shape of the array (must be square, e.g. (n,n)).

    Returns:
        bool : True if inside or on rim, False otherwise.
    """
    n, m = shape
    assert n == m, "Shape must be square"

    # circle center (middle of grid)
    cx, cy = (n - 1) / 2, (m - 1) / 2
    r = n / 2  # radius = half of side length

    # squared distance from center
    d2 = (x - cx)**2 + (y - cy)**2

    return d2 <= r**2 + 1e-9  # include rim

def circular_mask(shape: tuple[int, int]) -> np.ndarray:
    """
    Create a boolean mask for pixels inside or on the rim of a circle
    inscribed in a square array of given shape.
    """
    n, m = shape
    assert n == m, "Shape must be square"

    # grid of coordinates
    y, x = np.ogrid[:n, :m]  # y = rows, x = cols
    cx, cy = (n - 1) / 2, (m - 1) / 2
    r = n / 2

    # squared distance
    d2 = (x - cx)**2 + (y - cy)**2
    return d2 <= r**2

def erase_outer_circle(fft_data):
    mask = circular_mask(fft_data.shape)
    masked = fft_data.copy()
    masked[~mask] = None
    return masked

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

def normalize(array):
    valid_mask = ~np.isnan(array)  # True where data is valid
    if np.any(valid_mask):
        max_mag = np.abs(array[valid_mask]).max()
    else:
        max_mag = 1.0
    scale = max_mag if max_mag > 0 else 1.0
    normalized = array.copy()
    normalized[valid_mask] = array[valid_mask] / scale
    normalized[~valid_mask] = np.nan + 1j*np.nan
    return normalized, scale

def replace_nan_pixels(arr: np.ndarray, fill_value: complex = 0+0j) -> np.ndarray:
    """
    Replace NaN+NaN pixels in a complex array with a specified fill value.
    
    Args:
        arr : np.ndarray
            Complex array with NaNs for excluded pixels.
        fill_value : complex
            Value to replace NaN pixels with.
    
    Returns:
        np.ndarray : new array with NaNs replaced.
    """
    out = arr.copy()
    mask = np.isnan(arr.real) & np.isnan(arr.imag)
    out[mask] = fill_value
    return out

def serialize_complex(data, id, label, scale, orig_shape, trunc_factor, filename, precision, jsonl):
    """Serialize complex array to JSON"""
    serialized = {
        "id": id,
        "label": label,
        "data": [
            [
                [
                    None if (
                        math.isnan(val.real) or math.isnan(val.imag)
                    ) else (
                        round(float(val.real), precision) if precision else float(val.real)
                    ),
                    None if (
                        math.isnan(val.real) or math.isnan(val.imag)
                    ) else (
                        round(float(val.imag), precision) if precision else float(val.imag)
                    ),
                ] for val in row
            ] for row in data
        ],
        "scale": float(scale),
        "original_shape": orig_shape,
        "truncate_factor": trunc_factor
    }
    if jsonl:
        with open(filename, 'a') as f:
            json.dump(serialized, f, separators=(',', ':'))
            f.write('\n')
    else:
        with open(filename, 'w') as f:
            json.dump(serialized, f, separators=(',', ':'))

def load_complex(filename):
    """Load complex array from JSON"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    complex_array = np.array([
        [
            complex(r if r is not None else np.nan, i if i is not None else np.nan)
            for r, i in row
        ]
        for row in data["data"]
    ])
    return complex_array, data["scale"], data["original_shape"], data["truncate_factor"]

def process_image(item, output_dir, idx, truncate_factor, precision, create_plots, jsonl):
    [image, label] = item
    id = f"image_{idx:04d}"
    name = f"{id}_{label}"
    
    # Forward transform
    fft_data = np.fft.fftshift(np.fft.fftn(image, norm='forward'))
    
    # Truncate
    truncated = truncate_fft(fft_data, truncate_factor)

    # Verify untruncation
    if create_plots:
        untruncated = untruncate_fft(truncated, fft_data.shape)
        pre_reconstructured = np.fft.ifftn(np.fft.ifftshift(untruncated), norm='forward')
    
    # Experimental Circular Mask
    truncated = erase_outer_circle(truncated)

    non_redundant = remove_redundancies(truncated)
    serializable, scale = normalize(non_redundant)
    
    # Serialize
    if jsonl:
        json_path = os.path.join(output_dir, f"dataset.jsonl")
    else:
        json_path = os.path.join(output_dir, f"{name}.json")
    serialize_complex(serializable, id, label, scale, fft_data.shape, truncate_factor, json_path, precision, jsonl)
    
    if not create_plots:
        return
    
    # Load and Reconstruct
    loaded_norm, loaded_scale, loaded_shape, _ = load_complex(json_path)

    # Restore NaNs into defaults:
    loaded_norm = replace_nan_pixels(loaded_norm, fill_value = 0.0+0.0j)
    loaded_norm = loaded_norm * loaded_scale
    restored_hermitian = restore_hermitian(loaded_norm, truncated.shape)
    restored_untruncated = untruncate_fft(restored_hermitian, loaded_shape)
    restored_unfft = np.fft.ifftn(np.fft.ifftshift(restored_untruncated), norm='forward')
    
    # Plotting 
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
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

    axes[1, 3].imshow(np.real(restored_untruncated), cmap='RdBu')
    axes[1, 3].set_title('Json RFFT Real')
    
    axes[2, 3].imshow(np.imag(restored_untruncated), cmap='RdBu')
    axes[2, 3].set_title('Json RFFT Imag')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="FFT Image Processing with rfftn/irfftn")
    parser.add_argument("input_path", help="Image index or 'all'")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--truncate-factor", type=float, default=0.3, help="Fraction to keep (0-1)")
    parser.add_argument("--precision", type=int, help="Precision for serializing floats")
    parser.add_argument("--max-images", type=int, default=10, help="Max images when using 'all'")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split")
    parser.add_argument("--create-plots", action="store_true", help="Create example images")
    parser.add_argument("--jsonl", action="store_true", help="Stores JSONL instead of JSON")
    
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
            process_image(images[i], args.output_dir, i, args.truncate_factor, args.precision, args.create_plots, args.jsonl)
    else:
        try:
            idx = int(args.input_path)
            if idx >= len(images):
                print(f"Index {idx} out of range")
                return
            process_image(images[idx], args.output_dir, idx, args.truncate_factor, args.precision, args.create_plots, args.jsonl)
        except ValueError:
            print("Invalid input path")
            return
    
    print(f"Results in {args.output_dir}")

if __name__ == "__main__":
    main()