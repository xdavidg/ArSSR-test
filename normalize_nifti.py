import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def normalize_nifti(input_path, output_path):
    """
    Load a NIfTI image, normalize its intensity values to [0,1], and save it.
    
    Args:
        input_path: Path to the input NIfTI file
        output_path: Path to save the normalized NIfTI file
    """
    # Load the NIfTI image
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()
    
    # Get min and max values
    min_val = data.min()
    max_val = data.max()
    
    # Normalize to [0,1]
    normalized_data = (data - min_val) / (max_val - min_val)
    
    # Create a new NIfTI image with the same header and affine
    normalized_img = nib.Nifti1Image(normalized_data, nifti_img.affine, nifti_img.header)
    
    # Save the normalized image
    nib.save(normalized_img, output_path)
    
    return min_val, max_val

def main():
    # Define input and output directories
    input_dir = "hr_test_images"
    output_dir = "normalized_images"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Get list of NIfTI files in the input directory
    nifti_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]
    
    if not nifti_files:
        print(f"No NIfTI files found in {input_dir}")
        return
    
    print(f"Found {len(nifti_files)} NIfTI files to process")
    
    # Process each file
    for filename in tqdm(nifti_files, desc="Normalizing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            min_val, max_val = normalize_nifti(input_path, output_path)
            tqdm.write(f"Processed {filename}: Original range [{min_val:.4f}, {max_val:.4f}] → [0, 1]")
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {str(e)}")
    
    print(f"Normalization complete! {len(nifti_files)} files processed.")

if __name__ == "__main__":
    main() 