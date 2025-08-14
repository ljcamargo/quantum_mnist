#!/usr/bin/env python3
"""
Fixed FFT Reconstruction Script
Usage: python inverse_transform.py <input_json> <output_dir> [--ignore-scale]
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import Tuple

def load_complex_array(json_path: str) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Load the compact JSON format"""
    with open(json_path) as f:
        data = json.load(f)
    
    complex_array = np.array([
        [complex(val[0], val[1]) for val in row] 
        for row in data["data"]
    ], dtype=np.complex128)
    
    return complex_array, data.get("scale", 1.0), tuple(data["shape"])

def reconstruct_full_fft(
    cropped_data: np.ndarray,
    target_shape: Tuple[int, int],
    use_scaling: bool = True,
    scaling_factor: float = 1.0
) -> np.ndarray:
    """Properly reconstruct full FFT from cropped data"""
    h, w = target_shape
    non_redundant_h = h // 2
    non_redundant_w = w // 2
    
    # 1. Pad the cropped data to non-redundant quadrant size
    padded_quadrant = np.zeros((non_redundant_h, non_redundant_w), dtype=np.complex128)
    
    # Place cropped data in bottom-right of the quadrant
    start_h = non_redundant_h - cropped_data.shape[0]
    start_w = non_redundant_w - cropped_data.shape[1]
    padded_quadrant[start_h:, start_w:] = cropped_data * (scaling_factor if use_scaling else 1.0)
    
    # 2. Build full FFT
    full_fft = np.zeros(target_shape, dtype=np.complex128)
    
    # Place the padded quadrant
    full_fft[non_redundant_h:, non_redundant_w:] = padded_quadrant
    
    # Reconstruct symmetric parts
    # Top-right quadrant (excluding first row)
    if non_redundant_h > 1:
        full_fft[:non_redundant_h-1, non_redundant_w:] = np.conj(np.flipud(full_fft[non_redundant_h+1:, non_redundant_w:]))
    
    # Left half (excluding first column)
    if non_redundant_w > 1:
        full_fft[:, :non_redundant_w-1] = np.conj(np.fliplr(full_fft[:, non_redundant_w+1:]))
    
    # Handle DC component (must be real)
    full_fft[0, 0] = np.real(full_fft[0, 0])
    
    return full_fft

def reconstruct_fft_from_non_redundant(
    normalized_coeffs: np.ndarray, 
    scaling_factor: float, 
    mask_factor: float = 0.3
) -> np.ndarray:
    """
    Reconstructs the original FFT data from normalized non-redundant coefficients.
    
    Args:
        normalized_coeffs: Normalized non-redundant FFT coefficients (complex)
        scaling_factor: Original max magnitude used for normalization
        mask_factor: Same mask factor used in extraction (default 0.3)
    
    Returns:
        Reconstructed FFT data with original dimensions
    """
    # Calculate original dimensions from the non-redundant array and mask_factor
    non_red_h, non_red_w = normalized_coeffs.shape
    
    # Reverse the calculation: non_red dimensions = (center_h // 2, center_w // 2)
    # where center_h = int(h * mask_factor) (adjusted for even)
    # So: h * mask_factor / 2 ≈ non_red_h, therefore h ≈ non_red_h * 2 / mask_factor
    h = int(non_red_h * 2 / mask_factor)
    w = int(non_red_w * 2 / mask_factor)
    
    # Denormalize the coefficients
    denormalized = normalized_coeffs * scaling_factor
    
    # Calculate the kept region dimensions based on mask_factor
    center_h = int(h * mask_factor)
    center_w = int(w * mask_factor)
    
    # Ensure even dimensions for symmetry (same logic as original)
    center_h = center_h - 1 if center_h % 2 == 1 else center_h
    center_w = center_w - 1 if center_w % 2 == 1 else center_w
    
    # Initialize the reconstructed FFT with zeros
    reconstructed_fft = np.zeros((h, w), dtype=complex)
    
    # Calculate the placement bounds for the non-redundant portion
    start_h = h // 2
    start_w = w // 2
    end_h = start_h + (center_h // 2)
    end_w = start_w + (center_w // 2)
    
    # Place the denormalized coefficients in the bottom-right quadrant
    reconstructed_fft[start_h:end_h, start_w:end_w] = denormalized
    
    # Reconstruct the full FFT using Hermitian symmetry
    # For a real-valued input, FFT has conjugate symmetry: F(k) = F*(-k)
    
    # Top-left quadrant (DC and low frequencies)
    tl_h = center_h // 2
    tl_w = center_w // 2
    reconstructed_fft[:tl_h, :tl_w] = np.conj(np.flipud(np.fliplr(denormalized)))
    
    # Top-right quadrant
    reconstructed_fft[:tl_h, start_w:end_w] = np.conj(np.flipud(denormalized))
    
    # Bottom-left quadrant  
    reconstructed_fft[start_h:end_h, :tl_w] = np.conj(np.fliplr(denormalized))
    
    # Handle the DC component and edges for perfect symmetry
    # DC component should be real
    reconstructed_fft[0, 0] = np.real(reconstructed_fft[h//2, w//2])
    
    # Handle Nyquist frequencies if they exist
    if h % 2 == 0:
        reconstructed_fft[h//2, 0] = np.real(reconstructed_fft[h//2, 0])
        if w % 2 == 0:
            reconstructed_fft[h//2, w//2] = np.real(reconstructed_fft[h//2, w//2])
    
    if w % 2 == 0:
        reconstructed_fft[0, w//2] = np.real(reconstructed_fft[0, w//2])
    
    return reconstructed_fft

def save_reconstruction(
    reconstructed: np.ndarray,
    output_path: str,
    original_json: str,
    ignore_scale: bool
):
    """Save clean reconstruction without artifacts"""
    fig = plt.figure(figsize=(6, 6), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Display with proper grayscale
    img = ax.imshow(reconstructed, cmap='gray', vmin=0, vmax=1, interpolation='none')
    ax.set_title(f'Reconstructed (Scale {"ignored" if ignore_scale else "applied"})')
    ax.axis('off')
    
    # Add metadata
    plt.figtext(0.5, 0.01, 
               f"Source: {os.path.basename(original_json)}\n"
               f"Shape: {reconstructed.shape}", 
               ha="center")
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=100, facecolor='white')
    plt.close(fig)

def main():
    parser = ArgumentParser(description="Reconstruct MNIST from FFT JSON")
    parser.add_argument("input_json", help="Input JSON file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--ignore-scale", action="store_true",
                      help="Ignore scaling factor")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    cropped_data, scaling_factor, original_shape = load_complex_array(args.input_json)
    print(f"Loaded {cropped_data.shape} array from {args.input_json}")
    print(f"Original shape: {original_shape}, Scaling factor: {scaling_factor}")
    
    # Reconstruct
    # full_fft = reconstruct_full_fft(
    #     cropped_data,
    #     original_shape,
    #     not args.ignore_scale,
    #     scaling_factor
    # )

    full_fft = reconstruct_fft_from_non_redundant(
        cropped_data,
        scaling_factor,
        mask_factor=0.25  # Assuming same mask factor as used in extraction
    )
    
    # Inverse FFT
    reconstructed = np.real(np.fft.ifft2(np.fft.ifftshift(full_fft)))
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Save
    base_name = os.path.splitext(os.path.basename(args.input_json))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}_reconstructed.png")
    save_reconstruction(reconstructed, output_path, args.input_json, args.ignore_scale)
    
    print(f"Reconstruction saved to {output_path}")

if __name__ == "__main__":
    main()