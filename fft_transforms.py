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
7. Reconstruct from JSON data and compare

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
        mask_factor: float,  # Added mask_factor parameter
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
        "mask_factor": mask_factor  # Added mask_factor to JSON
    }
    
    with open(filename, 'w') as f:
        json.dump(serializable, f, separators=(',', ':'), indent=None)

def load_complex_array_from_json(filename: str) -> Tuple[np.ndarray, float, Tuple[int, int], float]:
    """Load complex array from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Reconstruct complex array
    complex_array = np.array([
        [complex(real, imag) for real, imag in row]
        for row in data["data"]
    ])
    
    return complex_array, data["scale"], tuple(data["shape"]), data["mask_factor"]

def reconstruct_fft_from_non_redundant(
    normalized_coeffs: np.ndarray, 
    scaling_factor: float, 
    original_shape: Tuple[int, int],
    mask_factor: float = 0.3
) -> np.ndarray:
    """
    Reconstructs the original FFT data from normalized non-redundant coefficients.
    Simply puts the data back where it was extracted from.
    
    Args:
        normalized_coeffs: Normalized non-redundant FFT coefficients (complex)
        scaling_factor: Original max magnitude used for normalization
        original_shape: Original dimensions (h, w) from JSON metadata
        mask_factor: Same mask factor used in extraction (default 0.3)
    
    Returns:
        Reconstructed FFT data with original dimensions
    """
    # Use the original dimensions from JSON metadata
    h, w = original_shape
    
    # Denormalize the coefficients
    denormalized = normalized_coeffs * scaling_factor
    
    # Calculate the kept region dimensions (same as masking logic)
    center_h = int(h * mask_factor)
    center_w = int(w * mask_factor)
    
    # Ensure even dimensions for symmetry (same logic as original)
    center_h = center_h - 1 if center_h % 2 == 1 else center_h
    center_w = center_w - 1 if center_w % 2 == 1 else center_w
    
    # Initialize the reconstructed FFT with zeros
    reconstructed_fft = np.zeros((h, w), dtype=complex)
    
    # The extract_non_redundant_fft function extracted from these exact coordinates:
    start_h = h // 2
    start_w = w // 2
    end_h = start_h + (center_h // 2)
    end_w = start_w + (center_w // 2)
    
    # Put the data back in the exact same location it was extracted from
    reconstructed_fft[start_h:end_h, start_w:end_w] = denormalized
    
    return reconstructed_fft

def process_image(image: np.ndarray, output_dir: str, image_idx: int, mask_factor: float = 0.3):
    """Process a single image through the complete pipeline."""
    image_name = f"image_{image_idx:04d}"
    
    # Step 0: Save original image
    print(f"Processing {image_name} - Step 0: Original Image")
    
    # Step 1: Apply FFT
    print(f"Processing {image_name} - Step 1: FFT")
    fft_data = apply_fft_2d(image)
    
    # Visualize original FFT
    fig = visualize_complex_array(fft_data, f"Original FFT - {image_name}")
    fig.savefig(os.path.join(output_dir, f"{image_name}_01_original_fft.png"))
    plt.close(fig)
    
    # Step 2: Apply mask to FFT (direct region copying, not multiplication)
    print(f"Processing {image_name} - Step 2: Masking")
    masked_fft = apply_center_mask_inplace(fft_data, mask_factor)

    # Visualize masked FFT
    fig = visualize_complex_array(fft_data, f"Original FFT - {image_name}")
    fig.savefig(os.path.join(output_dir, f"{image_name}_02_masked_fft.png"))
    plt.close(fig)
    
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
    serialize_complex_array(non_redundant, scaling_factor, original_shape, output_path, mask_factor)

    # Step 6: Load from JSON and reconstruct
    print(f"Processing {image_name} - Step 6: JSON Reconstruction Test")
    loaded_coeffs, loaded_scale, loaded_shape, loaded_mask_factor = load_complex_array_from_json(output_path)
    
    # Verify loaded data matches original
    print(f"Data integrity check:")
    print(f"  Coefficients match: {np.allclose(loaded_coeffs, non_redundant)}")
    print(f"  Scale match: {loaded_scale == scaling_factor}")
    print(f"  Shape match: {loaded_shape == original_shape}")
    print(f"  Mask factor match: {loaded_mask_factor == mask_factor}")
    
    # Reconstruct FFT from loaded JSON data
    reconstructed_fft_from_json = reconstruct_fft_from_non_redundant(
        loaded_coeffs, loaded_scale, loaded_shape, loaded_mask_factor
    )
    
    # Compare reconstructed FFT with original masked FFT
    fft_diff = np.abs(reconstructed_fft_from_json - masked_fft).max()
    print(f"  Max FFT reconstruction error: {fft_diff:.2e}")
    
    # Apply inverse FFT to get final reconstructed image
    final_reconstructed = apply_inverse_fft_2d(reconstructed_fft_from_json)
    final_reconstructed = np.real(final_reconstructed)  # Ensure real values only
    final_reconstructed = np.clip(final_reconstructed, 0, 1)  # Ensure valid range
    
    # Check if reconstructed image has any complex components (shouldn't happen)
    if np.iscomplexobj(final_reconstructed):
        print(f"  Warning: Reconstructed image has complex components!")
        max_imag = np.abs(np.imag(final_reconstructed)).max()
        print(f"  Max imaginary component: {max_imag:.2e}")
        final_reconstructed = np.real(final_reconstructed)
    
    # Compare final images
    image_diff = np.abs(final_reconstructed - reconstructed).max()
    print(f"  Max image reconstruction error: {image_diff:.2e}")
    
    # Step 7: Visualize reconstruction comparison
    print(f"Processing {image_name} - Step 7: Full Pipeline Comparison")
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    
    # Top row: Original pipeline
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reconstructed, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Direct Reconstruction\n(FFT → Mask → iFFT)')
    axes[0, 1].axis('off')

    magnitude = np.abs(masked_fft)
    axes[0, 2].imshow(np.log(magnitude + 1e-8), cmap='gray',)
    axes[0, 2].set_title('Masked FFT Magnitude')
    axes[0, 2].axis('off')

    phase = np.angle(masked_fft)
    phase_normalized = (phase + np.pi) / (2 * np.pi)
    axes[0, 3].imshow(phase_normalized, cmap='gray',)
    axes[0, 3].set_title('Masked FFT Phase')
    axes[0, 3].axis('off')

    print(f"Masked FFT {masked_fft}")
    
    # Bottom row: JSON pipeline
    axes[1, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(final_reconstructed, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('JSON Reconstruction\n(JSON → FFT → iFFT)')
    axes[1, 1].axis('off')

    magnitude = np.abs(reconstructed_fft_from_json)
    axes[0, 2].imshow(np.log(magnitude + 1e-8), cmap='gray',)
    axes[0, 2].set_title('JSON FFT Magnitude')
    axes[0, 2].axis('off')

    phase = np.angle(reconstructed_fft_from_json)
    phase_normalized = (phase + np.pi) / (2 * np.pi)
    axes[0, 3].imshow(phase_normalized, cmap='gray',)
    axes[0, 3].set_title('JSON FFT Phase')
    axes[0, 3].axis('off')

    print(f"Reconstructed FFT {reconstructed_fft_from_json}")
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{image_name}_06_full_pipeline_comparison.png"),
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Step 8: Compression ratio and statistics
    print(f"Processing {image_name} - Step 8: Statistics")
    original_size = image.size * 8  # float64 bytes
    json_size = os.path.getsize(output_path)  # JSON file size
    non_redundant_size = non_redundant.size * 16  # complex128 bytes
    
    compression_ratio = original_size / non_redundant_size
    
    stats = {
        "original_image_shape": image.shape,
        "original_fft_shape": fft_data.shape,
        "non_redundant_shape": non_redundant.shape,
        "mask_factor": mask_factor,
        "scaling_factor": float(scaling_factor),
        "compression_ratio": float(compression_ratio),
        "original_size_bytes": int(original_size),
        "non_redundant_size_bytes": int(non_redundant_size),
        "json_file_size_bytes": int(json_size),
        "max_fft_reconstruction_error": float(fft_diff),
        "max_image_reconstruction_error": float(image_diff),
        "mean_squared_error": float(np.mean((image - final_reconstructed)**2)),
        "peak_signal_noise_ratio": float(20 * np.log10(1.0 / np.sqrt(np.mean((image - final_reconstructed)**2))))
    }
    
    stats_path = os.path.join(output_dir, f"{image_name}_07_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  PSNR: {stats['peak_signal_noise_ratio']:.2f} dB")
    print(f"  JSON file size: {json_size} bytes")

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
    print("  *_03_reconstruction_comparison.png - Original vs direct reconstructed image")
    print("  *_04_non_redundant_fft.png - Non-redundant FFT data visualization")
    print("  *_05.json - JSON serialized complex array with metadata")
    print("  *_06_full_pipeline_comparison.png - Direct vs JSON reconstruction comparison")
    print("  *_07_stats.json - Processing statistics and metrics")
    print("\nThe JSON format includes all reconstruction metadata:")
    print("  data[i][j] = [real_part, imaginary_part]")
    print("  scale = normalization factor")
    print("  shape = original FFT dimensions")  
    print("  mask_factor = compression parameter")
    print("\nRequirements:")
    print("  pip install datasets pillow matplotlib numpy")

if __name__ == "__main__":
    main()