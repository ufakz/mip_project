import os
import pydicom
import yaml
import argparse
from typing import List
import matplotlib
import numpy as np
from matplotlib import pyplot as plt, animation

# -------General Functions-------

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error loading config file: {exc}")
                return {}
    return {}

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and visualize DICOM series with segmentations."
    )
    parser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the DICOM series."
    )
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing the DICOM series for the Input Image."
    )
    parser.add_argument(
        "--segment_dir", type=str, help="Path to the DICOM segmentation file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of intermediate steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/animation",
        help="Directory to save the output animation.",
    )
    return parser.parse_args()

def load_dicom_series(dataset_dir, key_info=False):
    """Loads and sorts a DICOM series from a directory."""
    ct_slices = []
    count = 0
    pixel_len_mm = None
    num_acquisitions = 0 # Initialize num_acquisitions

    # To register the acquisition number we are using
    acquisition = -1

    for f in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, f)
        try:
            ct_dcm = pydicom.dcmread(img_path)
            if acquisition == -1 and hasattr(ct_dcm, "SliceLocation"): # Ensure first slice has SliceLocation
                acquisition = ct_dcm.AcquisitionNumber
                pixel_len_mm = [ct_dcm.SliceThickness] + list(ct_dcm.PixelSpacing)
                num_acquisitions = 1 # Start counting acquisitions

                if key_info:
                    print("\n--- DICOM Series Key Information ---")
                    print(f"Patient ID: {ct_dcm.get('PatientID', 'N/A')}")
                    print(f"Study Description: {ct_dcm.get('StudyDescription', 'N/A')}")
                    print(f"Series Description: {ct_dcm.get('SeriesDescription', 'N/A')}")
                    print(f"Modality: {ct_dcm.get('Modality', 'N/A')}")
                    print(f"Acquisition Number: {ct_dcm.get('AcquisitionNumber', 'N/A')}")
                    print(f"Pixel Spacing (mm): {ct_dcm.get('PixelSpacing', 'N/A')}")
                    print(f"Slice Thickness (mm): {ct_dcm.get('SliceThickness', 'N/A')}")
                    print(f"Rows: {ct_dcm.get('Rows', 'N/A')}")
                    print(f"Columns: {ct_dcm.get('Columns', 'N/A')}")
                    print("------------------------------------\n")

            slice_acquisition = ct_dcm.AcquisitionNumber
            if hasattr(ct_dcm, "SliceLocation") and slice_acquisition == acquisition:
                ct_slices.append(ct_dcm)
            elif slice_acquisition != acquisition and acquisition != -1: # Check if acquisition was set
                # Only count additional acquisitions if they differ from the first valid one
                if slice_acquisition not in [s.AcquisitionNumber for s in ct_slices]:
                     num_acquisitions += 1
                print(f"Skipping slice {f} due to different acquisition number ({slice_acquisition} vs {acquisition}).")
                count += 1
            elif not hasattr(ct_dcm, "SliceLocation"):
                 print(f"Skipping file {f} as it lacks SliceLocation attribute.")
                 count += 1


        except Exception as e:
            print(f"Could not read file {img_path}: {e}")
            count += 1

    # The sorting of the slices is based on the SliceLocation attribute
    ct_slices = sorted(ct_slices, key=lambda x: -x.SliceLocation)

    print(f"Loaded {len(ct_slices)} slices from acquisition {acquisition}.")
    if num_acquisitions > 1:
        print(f"Detected {num_acquisitions} total acquisitions in the directory.")
    print(f"Skipped {count} files.")
    return ct_slices, pixel_len_mm


#------Visualization Functions------

def alpha_fusion(img: np.ndarray, mask: np.ndarray, n_objects: int, object_colors: List, alpha: float=0.5)->np.ndarray:
    """ Visualize both image and mask in the same plot. """
    
    cmap = matplotlib.colormaps['bone']
    cmap2 = matplotlib.colormaps['Set1']
    norm = matplotlib.colors.Normalize(vmin=np.amin(img), vmax=np.amax(img))
    fused_slice = \
        (1-alpha)*cmap(norm(img)) + \
        alpha*cmap2((mask/4))*mask[..., np.newaxis].astype('bool')

    return (fused_slice * 255).astype('uint8')

def MIP_per_plane(img_dcm: np.ndarray, axis: int = 2) -> np.ndarray:
    """ Compute the maximum intensity projection on the defined orientation. """
    return np.max(img_dcm, axis=axis)

def visualize_MIP_per_plane(img_vol: np.ndarray, pixel_len_mm: List):
    """ Creates an MIP visualize for each of the axis planes. """
    labels = ['Axial Plane', 'Coronal Plane', 'Sagittal Plane']
    ar = [(1,2),(0,2),(0,1)]
    for i in range(3):
        plt.imshow(MIP_per_plane(img_vol, i), aspect=pixel_len_mm[ar[i][0]]/pixel_len_mm[ar[i][1]])
        plt.title(f'MIP for {labels[i]}')
        plt.show()