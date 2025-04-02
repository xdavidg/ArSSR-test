# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: create_lr.py
#   Date    : 2024/3/31
#   Purpose: Using the same LR pipeline as train_loader to artificially create LR images to be used for testing (test.py)
# -----------------------------------------
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import argparse
from tqdm import tqdm
import utils
import random

if __name__ == "__main__":
    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input_path",
        type=str,
        default="hr_test_images",
        help="the file path of HR input images",
    )
    parser.add_argument(
        "-output_path",
        type=str,
        default="test/input",
        help="the file path to save LR images",
    )
    parser.add_argument(
        "-scale",
        type=float,
        default=None,
        help="the down-sampling scale (if not specified, will use random scale between 2 and 4)",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Get list of input files
    filenames = os.listdir(args.input_path)

    for f in tqdm(filenames):
        # Read the HR image
        hr_path = os.path.join(args.input_path, f)
        hr_img = sitk.ReadImage(hr_path)
        hr_vol = sitk.GetArrayFromImage(hr_img)

        # Get original spacing
        original_spacing = hr_img.GetSpacing()

        # Determine scale factor
        if args.scale is None:
            # Randomly get an up-sampling scale from [2, 4]
            scale = np.round(random.uniform(2, 4 + 0.04), 1)
        else:
            scale = args.scale

        # Create LR image by downsampling
        lr_vol = nd.interpolation.zoom(hr_vol, 1 / scale, order=3)

        # Adjust spacing for the LR image
        new_spacing = tuple(s * scale for s in original_spacing)

        # Create new image with adjusted spacing
        lr_img = sitk.GetImageFromArray(lr_vol)
        lr_img.SetSpacing(new_spacing)

        # Save the LR image with scale factor in filename
        output_filename = f"LR_{scale}x_{f}"
        output_path = os.path.join(args.output_path, output_filename)
        sitk.WriteImage(lr_img, output_path)

        print(f"Created low-resolution version: {output_filename} (scale: {scale})")

    print("Finished creating low-resolution images!")
