import os
import torch
import numpy as np
import nibabel as nib
from FullRC import RCContrastAugmentationWithNonLinearity
import matplotlib.pyplot as plt
from pathlib import Path


def create_directories():
    """Create input and output directories if they don't exist."""
    Path("RC_data/input").mkdir(parents=True, exist_ok=True)
    Path("RC_data/output").mkdir(parents=True, exist_ok=True)


def load_nifti(file_path):
    """Load a NIfTI file and return the data as a numpy array."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()


def save_nifti(data, file_path, reference_nifti):
    """Save data as a NIfTI file using the reference image's header."""
    nifti_img = nib.Nifti1Image(
        data, reference_nifti.affine, reference_nifti.header)
    nib.save(nifti_img, file_path)


def visualize_slice(input_data, output_data, slice_idx, output_path):
    """Visualize and save a comparison of input and output slices."""
    plt.figure(figsize=(12, 4))

    # Plot input
    plt.subplot(131)
    plt.imshow(input_data[slice_idx], cmap='gray')
    plt.title('Input')
    plt.axis('off')

    # Plot output
    plt.subplot(132)
    plt.imshow(output_data[slice_idx], cmap='gray')
    plt.title('Output')
    plt.axis('off')

    # Plot difference
    plt.subplot(133)
    plt.imshow(output_data[slice_idx] - input_data[slice_idx], cmap='gray')
    plt.title('Difference')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_image(input_path, output_path, rc_augmentation, device):
    """Process a single NIfTI image through the RC augmentation pipeline."""
    # Load the NIfTI image
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()

    # Convert to torch tensor and add batch and channel dimensions
    data_tensor = torch.from_numpy(data).unsqueeze(
        0).unsqueeze(0).float().to(device)

    # Apply RC augmentation
    with torch.no_grad():
        augmented_data = rc_augmentation(data_tensor)

    # Convert back to numpy and remove the extra dimensions
    augmented_data = augmented_data.squeeze().cpu().numpy()

    # Save the augmented data
    save_nifti(augmented_data, output_path, nifti_img)

    # Visualize middle slice
    middle_slice = data.shape[0] // 2
    visualize_slice(
        data,
        augmented_data,
        middle_slice,
        output_path.replace('.nii.gz', '_comparison.png')
    )

    return data, augmented_data


def main():
    # Create directories
    create_directories()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize RC augmentation
    rc_augmentation = RCContrastAugmentationWithNonLinearity(
        num_layers=4,
        kernel_size=1,
        negative_slope=0.2
    ).to(device)

    # Process all NIfTI files in the input directory
    input_dir = "RC_data/input"
    output_dir = "RC_data/output"

    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            print(f"Processing {filename}...")

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                data, augmented_data = process_image(
                    input_path,
                    output_path,
                    rc_augmentation,
                    device
                )

                # Print some statistics
                print(f"Input range: [{data.min():.3f}, {data.max():.3f}]")
                print(
                    f"Output range: [{augmented_data.min():.3f}, {augmented_data.max():.3f}]")
                print(
                    f"Mean difference: {np.mean(np.abs(augmented_data - data)):.3f}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print("Processing complete!")


if __name__ == "__main__":
    main()
