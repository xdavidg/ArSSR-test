# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: data_build_custom.py
#   Purpose: Create patches from custom dataset with dimensions 178x178x96
# -----------------------------------------
import os
import utils
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import random


def create_patches(input_path, output_path, patch_size=40, patches_per_image=6, margin=15):
    """
    Create patches from high-resolution images

    Args:
        input_path: Path to high-resolution images
        output_path: Path to save patches
        patch_size: Size of each patch (default: 40)
        patches_per_image: Number of patches per image (default: 6)
        margin: Margin to avoid black background (default: 15)
    """
    os.makedirs(output_path, exist_ok=True)

    # Get list of images
    filenames = [f for f in os.listdir(input_path) if f.endswith('.nii.gz')]

    for f in tqdm(filenames):
        name = f.split('.')[0]

        # Read image
        img_path = os.path.join(input_path, f)
        img = sitk.ReadImage(img_path)
        img_vol = sitk.GetArrayFromImage(img)

        h, w, d = img_vol.shape

        # Create patches
        for i in range(patches_per_image):
            # Random starting point avoiding black background
            # Note: Adjusted for smaller dimensions
            x0 = np.random.randint(margin, h-patch_size-margin)
            y0 = np.random.randint(margin, w-patch_size-margin)
            z0 = np.random.randint(margin, d-patch_size-margin)

            # Extract patch
            patch = img_vol[x0:x0+patch_size,
                            y0:y0+patch_size,
                            z0:z0+patch_size]

            # Save patch
            output_file = os.path.join(output_path, f'{name}_{i}.nii.gz')
            utils.write_img(vol=patch,
                            ref_path=img_path,
                            out_path=output_file)


if __name__ == '__main__':
    # Configuration
    config = {
        'raw_data_path': 'marmo_dataset/raw',
        'processed_data_path': 'marmo_dataset/processed',
        'train_ratio': 0.7,    # 70% for training
        'val_ratio': 0.15,     # 15% for validation
        'test_ratio': 0.15,    # 15% for testing
        'patch_size': 40,      # Keeping same patch size as original
        'patches_per_image': 6,
        'margin': 15           # Adjusted for smaller dimensions
    }

    # Create train/val/test split
    filenames = [f for f in os.listdir(
        config['raw_data_path']) if f.endswith('.nii.gz')]
    random.shuffle(filenames)

    total_files = len(filenames)
    train_idx = int(total_files * config['train_ratio'])
    val_idx = int(total_files * (config['train_ratio'] + config['val_ratio']))

    train_files = filenames[:train_idx]
    val_files = filenames[train_idx:val_idx]
    test_files = filenames[val_idx:]

    # Create patches for training set
    train_path = os.path.join(config['processed_data_path'], 'train')
    os.makedirs(train_path, exist_ok=True)
    for f in train_files:
        # Process the file
        create_patches(
            # Pass the directory containing the file
            input_path=config['raw_data_path'],
            output_path=train_path,
            patch_size=config['patch_size'],
            patches_per_image=config['patches_per_image'],
            margin=config['margin']
        )

    # Create patches for validation set
    val_path = os.path.join(config['processed_data_path'], 'val')
    os.makedirs(val_path, exist_ok=True)
    for f in val_files:
        # Process the file
        create_patches(
            # Pass the directory containing the file
            input_path=config['raw_data_path'],
            output_path=val_path,
            patch_size=config['patch_size'],
            patches_per_image=config['patches_per_image'],
            margin=config['margin']
        )

    # Create patches for test set
    test_path = os.path.join(config['processed_data_path'], 'test')
    os.makedirs(test_path, exist_ok=True)
    for f in test_files:
        # Process the file
        create_patches(
            # Pass the directory containing the file
            input_path=config['raw_data_path'],
            output_path=test_path,
            patch_size=config['patch_size'],
            patches_per_image=config['patches_per_image'],
            margin=config['margin']
        )

    print(f"Created patches with:")
    print(
        f"- Patch size: {config['patch_size']}x{config['patch_size']}x{config['patch_size']}")
    print(f"- Margin: {config['margin']} voxels")
    print(f"- Patches per image: {config['patches_per_image']}")
    print(
        f"- Training/Validation/Test split: {config['train_ratio']*100}%/{config['val_ratio']*100}%/{config['test_ratio']*100}%")
    print(f"- Number of files in each set:")
    print(f"  Training: {len(train_files)}")
    print(f"  Validation: {len(val_files)}")
    print(f"  Test: {len(test_files)}")
