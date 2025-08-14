#!/usr/bin/env python3
"""
FFT Image Processing Script for MNIST Dataset
Usage: python fft_transforms.py <input_path> <output_dir> [--mask-factor 0.3] [--split train]

This script loads MNIST data from Hugging Face and performs FFT analysis with:
1. Load MNIST dataset from Hugging Face (ylecun/mnist)
2. Apply 2D FFT to images
3. Apply dark square mask to simulate compression
4. Apply inverse FFT to see masking effects
5. Extract non-redundant FFT data (remove symmetric components)
6. Save visualizations and serialized complex arrays

Requirements: pip install datasets pillow matplotlib numpy
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, Optional

try:
    from datasets import load_dataset
    from PIL import Image
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

def load_mnist_from_huggingface(split: str = "train", max_images: Optional[int] = None) -> np.ndarray:
    """
    Load MNIST images from Hugging Face dataset.
    
    Args:
        split: 'train' or 'test'
        max_images: Maximum number of images to load
    
    Returns:
        numpy array of images normalized to [0, 1]
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library not found. Please install it with:\n"
            "pip install datasets pillow"
        )
    
    print(f"Loading MNIST {split} dataset from Hugging Face...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("ylecun/mnist", split=split)
    
    if max_images is not None:
        dataset = dataset.select(range(min(max_images, len(dataset))))
    
    print(f"Converting {len(dataset)} images to numpy arrays...")
    
    # Convert PIL images to numpy arrays
    images = []
    for i, item in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} images...")
        
        # Convert PIL Image to numpy array
        pil_image = item['image']
        np_image = np.array(pil_image, dtype=np.float32) / 255.0
        images.append(np_image)
    
    return np.array(images)

def load_mnist_images(split: str = "train", max_images: Optional[int] = None) -> np.ndarray:
    """
    Load MNIST images using the best available method.
    
    Args:
        split: 'train' or 'test' 
        max_images: Maximum number of images to load
    
    Returns:
        numpy array of images normalized to [0, 1]
    """
    return load_mnist_from_huggingface(split, max_images)

def apply_center_mask_inplace(fft_data: np.ndarray, mask_factor: float = 0.3) -> np.ndarray:
    """
    Apply masking by zeroing out outer regions while preserving center values.
    """
    h, w = fft_data.shape
    masked_fft = fft_data.copy()  # Start with original data
    
    # Calculate center region to keep
    center_h = int(h * mask_factor)
    center_w = int(w * mask_factor)
    
    # Make dimensions even for perfect centering
    if center_h % 2 == 1:
        center_h -= 1
    if center_w % 2 == 1:
        center_w -= 1
    
    # Calculate starting positions for perfectly centered region
    start_h = (h - center_h) // 2
    start_w = (w - center_w) // 2
    
    # Zero out everything EXCEPT the center region
    masked_fft[:start_h, :] = 0.0 + 0.0j  # Top
    masked_fft[start_h + center_h:, :] = 0.0 + 0.0j  # Bottom
    masked_fft[start_h:start_h + center_h, :start_w] = 0.0 + 0.0j  # Left of center
    masked_fft[start_h:start_h + center_h, start_w + center_w:] = 0.0 + 0.0j  # Right of center
    
    return masked_fft

def apply_fft_2d(image: np.ndarray) -> np.ndarray:
    """Apply 2D FFT to image with shifting."""
    return np.fft.fftshift(np.fft.fft2(image))

def apply_inverse_fft_2d(fft_data: np.ndarray) -> np.ndarray:
    """Apply inverse 2D FFT with shifting."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft_data)))

def extract_non_redundant_fft(fft_data: np.ndarray, mask_factor: float = 0.3) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Returns:
        - Normalized non-redundant FFT coefficients (complex)
        - Original max magnitude (for reconstruction)
        - Original dimensions (h, w) for reconstruction
    """
    h, w = fft_data.shape
    
    # Calculate the kept region dimensions based on mask_factor
    center_h = int(h * mask_factor)
    center_w = int(w * mask_factor)
    
    # Ensure even dimensions for symmetry
    center_h = center_h - 1 if center_h % 2 == 1 else center_h
    center_w = center_w - 1 if center_w % 2 == 1 else center_w
    
    # Calculate the bottom-right quadrant bounds (non-redundant portion)
    start_h = h // 2
    start_w = w // 2
    end_h = start_h + (center_h // 2)
    end_w = start_w + (center_w // 2)
    
    # Extract and crop the non-redundant portion
    non_redundant = fft_data[start_h:end_h, start_w:end_w].copy()
    
    # Normalization
    max_mag = np.abs(non_redundant).max()
    scaling_factor = max_mag if max_mag > 0 else 1.0
    normalized = non_redundant / scaling_factor
    
    return normalized, scaling_factor, (h, w)

def visualize_complex_array(data: np.ndarray, title: str = "") -> plt.Figure:
    """Create visualization of complex array (magnitude and phase)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Magnitude (log scale)
    magnitude = np.abs(data)
    im_mag = axes[0].imshow(np.log(magnitude + 1e-8), cmap='gray')
    axes[0].set_title(f'{title} - Log Magnitude')
    plt.colorbar(im_mag, ax=axes[0])
    
    # Phase (grayscale for better analysis)
    phase = np.angle(data)
    # Normalize phase from [-π, π] to [0, 1] for grayscale
    phase_normalized = (phase + np.pi) / (2 * np.pi)
    im_phase = axes[1].imshow(phase_normalized, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'{title} - Phase (grayscale)')
    plt.colorbar(im_phase, ax=axes[1])
    
    plt.tight_layout()
    return fig

def serialize_complex_array(
        data: np.ndarray, 
        scaling_factor: float, 
        original_shape: Tuple[int, int], 
        filename: str,
        precision: int = 10
    ):
    """Serializes with shape metadata for reconstruction"""
    serializable = {
        "data": [
            [
                [
                    round(float(np.real(val)), precision),
                    round(float(np.imag(val)), precision)
                ] for val in row
            ] for row in data
        ],
        "shape": original_shape,
        "scale": round(float(scaling_factor), 2),
    }
    
    with open(filename, 'w') as f:
        json.dump(serializable, f, separators=(',', ':'), indent=None)

def process_image(image: np.ndarray, output_dir: str, image_idx: int, mask_factor: float = 0.3):
    """Process a single image through the complete pipeline."""
    image_name = f"image_{image_idx:04d}"
    
    # Step 0: Save original image
    print(f"Processing {image_name} - Step 0: Original Image")
    #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    #ax.set_title(f'Original MNIST Image - {image_name}')
    #ax.axis('off')
    #plt.tight_layout()
    #fig.savefig(os.path.join(output_dir, f"{image_name}_00_original_image.png"), 
    #            bbox_inches='tight', facecolor='white')
    #plt.close(fig)
    
    # Step 1: Apply FFT
    print(f"Processing {image_name} - Step 1: FFT")
    fft_data = apply_fft_2d(image)
    
    # Visualize original FFT
    #fig = visualize_complex_array(fft_data, f"Original FFT - {image_name}")
    #fig.savefig(os.path.join(output_dir, f"{image_name}_01_original_fft.png"))
    #plt.close(fig)
    
    # Step 2: Apply mask to FFT (direct region copying, not multiplication)
    print(f"Processing {image_name} - Step 2: Masking")
    masked_fft = apply_center_mask_inplace(fft_data, mask_factor)
    
    # Create mask visualization for reference
    h, w = fft_data.shape
    center_h = int(h * mask_factor)
    center_w = int(w * mask_factor)
    if center_h % 2 == 1: center_h -= 1
    if center_w % 2 == 1: center_w -= 1
    start_h = (h - center_h) // 2
    start_w = (w - center_w) // 2
    
    mask_vis = np.zeros((h, w))
    mask_vis[start_h:start_h + center_h, start_w:start_w + center_w] = 1.0
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'FFT Mask (factor={mask_factor}) - {image_name}')
    ax.axis('off')
    #plt.tight_layout()
    #fig.savefig(os.path.join(output_dir, f"{image_name}_02a_mask.png"), 
    #            bbox_inches='tight', facecolor='white')
    #plt.close(fig)
    
    # Verify masking worked correctly
    print(f"Original FFT non-zero count: {np.count_nonzero(fft_data)}")
    print(f"Masked FFT non-zero count: {np.count_nonzero(masked_fft)} (should be ~{center_h*center_w})")

    # Check that exactly the center region is preserved and all else is zero
    center_original = fft_data[start_h:start_h + center_h, start_w:start_w + center_w]
    center_masked = masked_fft[start_h:start_h + center_h, start_w:start_w + center_w]
    center_diff = np.abs(center_original - center_masked).max()
    print(f"Max difference in center region: {center_diff:.2e} (should be 0.0)")

    # Verify all non-center values are exactly zero
    non_center_mask = np.ones_like(masked_fft, dtype=bool)
    non_center_mask[start_h:start_h + center_h, start_w:start_w + center_w] = False
    non_center_values = masked_fft[non_center_mask]
    print(f"Max non-center value: {np.abs(non_center_values).max():.2e} (should be 0.0)")
    
    # Visualize masked FFT
    #fig = visualize_complex_array(masked_fft, f"Masked FFT - {image_name}")
    #fig.savefig(os.path.join(output_dir, f"{image_name}_02b_masked_fft.png"))
    #plt.close(fig)
    
    # Step 3: Apply inverse FFT to see masking effect
    print(f"Processing {image_name} - Step 3: Inverse FFT")
    reconstructed = apply_inverse_fft_2d(masked_fft)
    
    # Ensure reconstructed image is properly bounded and real
    reconstructed = np.real(reconstructed)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Visualize original vs reconstructed
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Reconstructed (After Masking)')
    axes[1].axis('off')
    
    difference = np.abs(image - reconstructed)
    axes[2].imshow(difference, cmap='hot', vmin=0, vmax=np.max(difference))
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{image_name}_03_reconstruction_comparison.png"),
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Step 4: Extract non-redundant FFT data
    print(f"Processing {image_name} - Step 4: Non-redundant extraction")
    non_redundant, scaling_factor, original_shape = extract_non_redundant_fft(masked_fft, mask_factor)

    # Visualize non-redundant data (pass just the array, not the tuple)
    fig = visualize_complex_array(non_redundant, f"Non-redundant FFT - {image_name}")
    fig.savefig(os.path.join(output_dir, f"{image_name}_04_non_redundant_fft.png"))
    plt.close(fig)

    # Step 5: Serialize complex array (pass both array and scaling factor)
    print(f"Processing {image_name} - Step 5: Serialization")
    output_path = os.path.join(output_dir, f"{image_name}_05.json")
    serialize_complex_array(non_redundant, scaling_factor, original_shape, output_path)

    # Save processing summary
    # summary = {
    #     "image_index": image_idx,
    #     "original_shape": image.shape,
    #     "fft_shape": fft_data.shape,
    #     "non_redundant_shape": non_redundant.shape,
    #     "mask_factor": mask_factor,
    #     "compression_ratio": (np.prod(fft_data.shape) / np.prod(non_redundant.shape)),
    #     "reconstruction_mse": float(np.mean((image - reconstructed) ** 2))
    # }
    
    # with open(os.path.join(output_dir, f"{image_name}_summary.json"), 'w') as f:
    #     json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="FFT Image Processing for MNIST")
    parser.add_argument("input_path", help="Input path (image index or 'all' for batch processing)")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--mask-factor", type=float, default=0.3, 
                       help="Mask factor (fraction of center to keep, 0-1)")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to process when using 'all'")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                       help="Dataset split to use (train or test)")
    
    args = parser.parse_args()
    
    # Check if required libraries are available
    if not DATASETS_AVAILABLE:
        print("Error: Required libraries not found.")
        print("Please install them with:")
        print("pip install datasets pillow")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MNIST images from Hugging Face
    print("Loading MNIST images from Hugging Face...")
    try:
        # Load more images for the pool, then limit processing
        max_load = 1000 if args.input_path.lower() == "all" else 100
        images = load_mnist_images(args.split, max_images=max_load)
        print(f"Loaded {len(images)} images from {args.split} split")
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        print("Make sure you have internet connection and the datasets library installed:")
        print("pip install datasets pillow")
        return
    
    # Process images
    if args.input_path.lower() == "all":
        # Process multiple images
        num_to_process = min(args.max_images, len(images))
        print(f"Processing {num_to_process} images...")
        
        for i in range(num_to_process):
            print(f"\n--- Processing image {i+1}/{num_to_process} ---")
            process_image(images[i], args.output_dir, i, args.mask_factor)
    else:
        # Process single image
        try:
            image_idx = int(args.input_path)
            if image_idx >= len(images):
                print(f"Image index {image_idx} out of range (0-{len(images)-1})")
                return
            
            print(f"Processing single image at index {image_idx}...")
            process_image(images[image_idx], args.output_dir, image_idx, args.mask_factor)
            
        except ValueError:
            print(f"Invalid input path: {args.input_path}. Use integer index or 'all'")
            return
    
    print(f"\nProcessing complete! Results saved to {args.output_dir}")
    print(f"Dataset used: MNIST {args.split} split from Hugging Face")
    print("\nGenerated files per image:")
    print("  *_00_original_image.png - Original MNIST image")
    print("  *_01_original_fft.png - Original FFT visualization")
    print("  *_02a_mask.png - FFT mask visualization")
    print("  *_02b_masked_fft.png - Masked FFT visualization") 
    print("  *_03_reconstruction_comparison.png - Original vs reconstructed image")
    print("  *_04_non_redundant_fft.png - Non-redundant FFT data")
    print("  *_05_complex_array.json - JSON serialized complex array (human-readable)")
    print("  *_summary.json - Processing summary and metrics")
    print("\nThe JSON format allows easy parsing and reconstruction:")
    print("  data[i][j] = [real_part, imaginary_part]")
    print("\nRequirements:")
    print("  pip install datasets pillow matplotlib numpy")

if __name__ == "__main__":
    main()