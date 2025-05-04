import os
import pydicom
import numpy as np
import argparse
import yaml
from utils import load_config, load_dicom_series, parse_arguments, alpha_fusion, visualize_MIP_per_plane
import matplotlib.pyplot as plt

VISUALIZE = False

def min_max_normalization(vol: np.ndarray) -> np.ndarray:
    """Applies min-max normalization to the input volume."""
    min_val = np.min(vol)
    max_val = np.max(vol)
    normalized_image = (vol - min_val) / (max_val - min_val)
    return normalized_image

def display_planes(volume: np.ndarray, pixel_len_mm: list):
    """Displays the axial, coronal, and sagittal planes of the volume."""
    depth, height, width = volume.shape

    # Calculate central indices
    axial_idx = depth // 2
    coronal_idx = height // 2
    sagittal_idx = width // 2

    plt.figure(figsize=(10, 6))

    plt.subplot(131)
    plt.imshow(volume[axial_idx,:,:], cmap='bone', aspect=pixel_len_mm[1]/pixel_len_mm[2])
    plt.title(f"Axial Plane (Slice {axial_idx})")
    plt.xlabel("Width")
    plt.ylabel("Height")

    plt.subplot(132)
    plt.imshow(volume[:,coronal_idx,:], cmap='bone', aspect=pixel_len_mm[0]/pixel_len_mm[2])
    plt.title(f"Coronal Plane (Slice {coronal_idx})")
    plt.xlabel("Width")
    plt.ylabel("Depth")

    plt.subplot(133)
    plt.imshow(volume[:,:,sagittal_idx], cmap='bone', aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.title(f"Sagittal Plane (Slice {sagittal_idx})")
    plt.xlabel("Height")
    plt.ylabel("Depth")

    plt.tight_layout()
    plt.show()

def main():
    
    args = parse_arguments()
    config = load_config(args.config)

    dataset_dir = args.dataset_dir or config.get("dataset_dir")
    input_dir = args.input_dir or config.get("input_dir")
    output_dir = args.output_dir or config.get(
        "output_dir", "results/animation"
    )  
    
    visualize_flag = args.visualize or config.get("visualize", VISUALIZE)

    if not dataset_dir or not input_dir:
        raise ValueError(
            "Reference directory and input directory path must be provided via command line or config file."
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    try: 
        # Step 1: Load the reference and input DICOM series
        ref_slices, ref_pixel_len_mm = load_dicom_series(dataset_dir, key_info=True)
        input_slices, input_pixel_len_mm = load_dicom_series(input_dir, key_info=True)
        
        # Check if pixel spacing is consistent between reference and input volumes
        if ref_pixel_len_mm != input_pixel_len_mm:
            print(f"Pixel spacing mismatch: Reference={ref_pixel_len_mm}mm, Input={input_pixel_len_mm}mm")
        else:
            print(f"Pixel spacing is consistent in both images: {ref_pixel_len_mm}mm")
        
        ref_volume = np.array([f.pixel_array for f in ref_slices])
        input_volume = np.array([f.pixel_array for f in input_slices])
        
        print(f"Ref volume shape: {ref_volume.shape}")
        print(f"Input volume shape: {input_volume.shape}")
        
        if visualize_flag:
            visualize_MIP_per_plane(ref_volume, ref_pixel_len_mm)
            visualize_MIP_per_plane(input_volume, input_pixel_len_mm)
        
        # Step 2: Normalize both volumes
        ref_volume = min_max_normalization(ref_volume)
        input_volume = min_max_normalization(input_volume)

        display_planes(ref_volume, ref_pixel_len_mm)
        
        # scan_3d = rigid_transformation(scan_3d, parameters, img_phantom)
        # scan_3d_pspace = img_phantom
        # img_atlas_pspace = img_atlas
        
            
            
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # Object names and color configuration
    
    
if __name__ == "__main__":
    main()