import os
import torch
import numpy as np
import nibabel as nib
from FullRC import RCContrastAugmentationWithNonLinearity
import matplotlib.pyplot as plt
from pathlib import Path



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
    
    # Store original min and max for denormalization if needed
    original_min = data.min()
    original_max = data.max()
    
    # Normalize data to [0,1] range
    data_normalized = (data - original_min) / (original_max - original_min)

    # Convert to torch tensor and add batch and channel dimensions
    data_tensor = torch.from_numpy(data_normalized).unsqueeze(
        0).unsqueeze(0).float().to(device)

    # Apply RC augmentation
    with torch.no_grad():
        augmented_data = rc_augmentation(data_tensor)

    # Convert back to numpy and remove the extra dimensions
    augmented_data = augmented_data.squeeze().cpu().numpy()
    
    # Clip values to ensure they stay in [0,1] range
    augmented_data = np.clip(augmented_data, 0, 1)
    
    # Optional: Denormalize back to original range
    # Uncomment the following line if you want to restore original intensity range
    # augmented_data = augmented_data * (original_max - original_min) + original_min

    # Save the augmented data
    save_nifti(augmented_data, output_path, nifti_img)

    # Visualize middle slice
    middle_slice = data_normalized.shape[0] // 2
    visualize_slice(
        data_normalized,
        augmented_data,
        middle_slice,
        output_path.replace('.nii.gz', '_comparison.png')
    )

    return data_normalized, augmented_data


def main():
    # Create output directory if it doesn't exist
    os.makedirs("RC_data/output", exist_ok=True)

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
            
            # Add "_rc" to the output filename
            base_name = os.path.splitext(os.path.splitext(filename)[0])[0]  # Remove both .nii and .gz extensions
            output_filename = f"{base_name}_rc.nii.gz"
            output_path = os.path.join(output_dir, output_filename)

            try:
                data_normalized, augmented_data = process_image(
                    input_path,
                    output_path,
                    rc_augmentation,
                    device
                )

                # Print some statistics
                print(f"Input range (normalized): [{data_normalized.min():.3f}, {data_normalized.max():.3f}]")
                print(
                    f"Output range: [{augmented_data.min():.3f}, {augmented_data.max():.3f}]")
                print(
                    f"Mean difference: {np.mean(np.abs(augmented_data - data_normalized)):.3f}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print("Processing complete!")


if __name__ == "__main__":
    main()
