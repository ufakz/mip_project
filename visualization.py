import os
import pydicom
import numpy as np
import argparse
import yaml
from utils import load_config, parse_arguments, \
    load_dicom_series, alpha_fusion, visualize_MIP_per_plane, visualize_AIP_per_plane, \
    apply_window_level, plot_ct_histogram, calculate_snr, calculate_cnr, \
    create_animation, apply_intensity_windowing
import matplotlib.pyplot as plt

VISUALIZE = False 

def load_segmentations(segment_dir):
    """Loads segmentation data from DICOM files in a directory."""
    seg_dcms = []
    
    for f in os.listdir(segment_dir):
        file_path = os.path.join(segment_dir, f)
        seg_dcm = pydicom.dcmread(file_path)
        seg_dcms.append(seg_dcm)
                    
    if not seg_dcms:
        raise IOError(f"No valid segmentation files found in {segment_dir}")
        
    return seg_dcms


def map_segmentation_to_slices(seg_dicoms):
    valid_masks = {}
    for i, seg_dcm in enumerate(seg_dicoms):
        series_dt = seg_dcm.PerFrameFunctionalGroupsSequence
        for k in range(len(series_dt)):
            ref_ct_image = (
                series_dt[k]
                .DerivationImageSequence[0]
                .SourceImageSequence[0]
                .ReferencedSOPInstanceUID
            )
            frame_it_references = series_dt[k].DerivationImageSequence[0].SourceImageSequence[0].ReferencedFrameNumber
            if i not in valid_masks:
                valid_masks[i] = [{'seg_id': k, 'referenced_id': int(frame_it_references - 1)}]
            else:
                valid_masks[i].append({'seg_id': k, 'referenced_id': int(frame_it_references - 1)})
    return valid_masks
    
def create_ordered_segmentation_mask(image_3d, seg_dicoms, valid_masks):
    """Creates an ordered 3D mask volume of all segmentation masks corresponding to the CT slices."""
    reordered_seg = np.zeros_like(image_3d, dtype=np.uint8)
    print(reordered_seg.shape)
    
    for i, seg_dcm in enumerate(seg_dicoms):
        seg_3d = np.array(seg_dcm.pixel_array)
        mask = np.zeros(seg_3d[0].shape, dtype=np.uint8)
        
        for info in valid_masks[i]:
            ref_mask = seg_3d[info['seg_id']] 
            mask[ref_mask == 1] = int(i+1)
            #reordered_seg[info['referenced_id']] = mask
            current_slice = reordered_seg[info['referenced_id']]
            current_slice[ref_mask == 1] = mask[ref_mask == 1]
            reordered_seg[info['referenced_id']] = current_slice
            
    return reordered_seg


def create_fused_volume(
    img_3d_np,
    reordered_seg_np,
    objects,
    object_colors,
    pixel_len_mm,
    visualize_slices=False,
):
    """Creates a 3D volume of alpha-fused images."""
    img_segmented = []
    
    for i in range(len(img_3d_np)):
        fused_slice = alpha_fusion(
            img_3d_np[i], reordered_seg_np[i], len(objects), object_colors
        )
        
        img_segmented.append(fused_slice)
        if visualize_slices and i % 10 == 0:  # Visualize every 10th slice
            plt.imshow(fused_slice, aspect=pixel_len_mm[1] / pixel_len_mm[2])
            plt.title(f"Alpha-fused Slice {i} on Axial Plane")
            plt.show()
    return np.array(img_segmented)


def main():
    args = parse_arguments()
    config = load_config(args.config)

    dataset_dir = args.dataset_dir or config.get("dataset_dir")
    segment_dir = args.segment_dir or config.get("segment_dir")
    output_dir = args.output_dir or config.get(
        "output_dir", "results/animation"
    )  
    visualize_flag = args.visualize or config.get("visualize", VISUALIZE)

    if not dataset_dir or not segment_dir:
        raise ValueError(
            "Dataset directory and segmentation file path must be provided via command line or config file."
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Object names and color configuration
    objects = ["Liver", "Tumor"]
    object_colors = [(0, 1, 0), (1, 0, 0)]  # Red for Liver, Green for Tumor

    try:
        # 1. Load the DICOM series
        
        ct_slices, pixel_len_mm = load_dicom_series(dataset_dir, key_info=True)
        # Convert to float to avoid overflow
        img_3d_np = np.array([f.pixel_array.astype(float) for f in ct_slices])
        
        if visualize_flag:
            print("Visualizing raw CT slices before contrast enhancement...")
            for i, img in enumerate(img_3d_np):
                if i % 10 == 0:  # Visualize every 10th slice
                    plt.imshow(
                        img, aspect=pixel_len_mm[1] / pixel_len_mm[2], cmap="bone"
                    )
                    plt.title(f"CT slice {i} on axial plane - Before Contrast")
                    plt.show()
        
        # Plot histogram of pixel values before contrast enhancement
        if visualize_flag:
            print("Plotting histogram of pixel values before contrast enhancement...")
            plot_ct_histogram(img_3d_np)
        
        # Apply window/level adjustment
        img_3d_np_adjusted, noise_mask = apply_window_level(img_3d_np, window=400, level=40)
        
        img_3d_np_mip, _ = apply_window_level(img_3d_np, window=800, level=400)
        
        # Find the most frequent value (likely background or air)
        vals, counts = np.unique(img_3d_np_adjusted.flatten(), return_counts=True)
        most_frequent_value = vals[np.argmax(counts)]
        second_most_frequent_value = vals[np.argsort(counts)[-2]]
        
        # Create a new array excluding the most frequent value for histogram plotting
        img_3d_np_adjusted_no_outliers = img_3d_np_adjusted[img_3d_np_adjusted != most_frequent_value]
        img_3d_np_adjusted_no_outliers = img_3d_np_adjusted_no_outliers[img_3d_np_adjusted_no_outliers != second_most_frequent_value]
        if visualize_flag:
            print("Plotting histogram of pixel values after contrast enhancement...")
            plot_ct_histogram(img_3d_np_adjusted_no_outliers)
        
        #img_3d_np_adjusted = apply_intensity_windowing(img_3d_np, a=20, b=300)
        if visualize_flag:
            print("Visualizing raw CT slices before after enhancement...")
            for i, img in enumerate(img_3d_np_adjusted):
                if i % 10 == 0:  # Visualize every 10th slice
                    plt.imshow(
                        img, aspect=pixel_len_mm[1] / pixel_len_mm[2], cmap="bone"
                    )
                    plt.title(f"CT slice {i} on axial plane - After Contrast")
                    plt.show()
        
        print("Calculating image quality metrics...")
        snr = calculate_snr(img_3d_np_adjusted, noise_mask)
        print(f"Signal-to-Noise Ratio (SNR): {snr:.2f}")
        cnr = calculate_cnr(img_3d_np_adjusted, noise_mask)
        print(f"Contrast-to-Noise Ratio (CNR): {cnr:.2f}")

        # 2. Load Segmentation
        seg_dcm = load_segmentations(segment_dir)

        # 3. Map Segmentation to Series
        valid_masks = map_segmentation_to_slices(seg_dcm)

        # 4. Create Ordered Mask Volume
        reordered_seg = create_ordered_segmentation_mask(
            img_3d_np_adjusted,
            seg_dcm,
            valid_masks
        )
        
        # 5. Create Fused Volume
        img_fused_3d = create_fused_volume(
            img_3d_np_adjusted,
            reordered_seg,
            objects,
            object_colors,
            pixel_len_mm,
            visualize_slices=visualize_flag,
        )
        
        img_fused_3d_mip = create_fused_volume(
            img_3d_np_mip,
            reordered_seg,
            objects,
            object_colors,
            pixel_len_mm,
            visualize_slices=False,
        )

        # 6. Visualize MIPs
        if visualize_flag:
            print("Visualizing MIPs...")
            visualize_MIP_per_plane(img_fused_3d_mip, pixel_len_mm)
            
            
        # 7. Create animation
        create_animation(img_fused_3d_mip, pixel_len_mm, objects, object_colors, n=24, save_dir=output_dir, show=VISUALIZE)


    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
