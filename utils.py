import os
import pydicom
import yaml
import argparse
from typing import List
import matplotlib
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt, animation
import matplotlib.patches as mpatches
import tqdm
from matplotlib.widgets import Slider


# -------Preprocessing Functions-------
def apply_window_level(image, window=150, level=30):
    """
    Apply window/level adjustment to medical images.
    Args:
        image: Input image array
        window: Controls contrast (default 350 for soft tissue)
        level: Controls brightness (default 40 for soft tissue)
    Returns:
        Adjusted image array
    """
    img_min = level - window // 2
    img_max = level + window // 2

    noise_mask = estimate_noisy_pixels(image)

    return np.clip(image, img_min, img_max), noise_mask

def apply_intensity_windowing(image_array, a, b, k=255):
    """Applies intensity windowing to a grayscale image."""

    img_float = image_array.astype(np.float32)

    output_image_float = np.zeros_like(img_float)

    # Apply linear scaling for pixels within the window [a, b]
    # (b - a) is guaranteed to be positive due to the check a < b.
    mask_middle = (img_float >= a) & (img_float <= b)
    output_image_float[mask_middle] = k * (img_float[mask_middle] - a) / (b - a)

    # Pixels above the window are set to the maximum output intensity k.
    mask_upper = img_float > b
    output_image_float[mask_upper] = k

    # This handles any potential floating point inaccuracies.
    output_image_float = np.clip(output_image_float, 0, k)

    # If k is 255 (common for 8-bit display) and input was uint8, output uint8.
    if k == 255 and image_array.dtype == np.uint8:
        return output_image_float.astype(np.uint8)
    else:
        return output_image_float


def plot_ct_histogram(img_3d_np, bins=256):
    """
    Plot histogram of CT values from a 3D image array.

    Args:
        img_3d_np: 3D numpy array of CT values
        bins: Number of histogram bins (default: 256)
    """
    plt.figure()
    plt.hist(img_3d_np.flatten(), bins=bins)
    plt.title("Histogram of CT values")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()

def center_crop(vol: np.ndarray, dim):
    """Returns center cropped volume.
    Args:
    vol: volume to be center cropped
    dim: dimensions to be cropped (target_depth, target_height, target_width)
    """
    X, Y, Z = vol.shape 

    target_X, target_Y, target_Z = dim 

    # Calculate starting and ending indices for the crop for X dimension
    mid_x_vol = X // 2
    half_target_X = target_X // 2
    start_x = mid_x_vol - half_target_X
    end_x = start_x + target_X

    # Calculate starting and ending indices for the crop for Y dimension
    mid_y_vol = Y // 2
    half_target_Y = target_Y // 2
    start_y = mid_y_vol - half_target_Y
    end_y = start_y + target_Y

    # Calculate starting and ending indices for the crop for Z dimension
    mid_z_vol = Z // 2
    half_target_Z = target_Z // 2
    start_z = mid_z_vol - half_target_Z
    end_z = start_z + target_Z

    crop_vol = vol[start_x:end_x, start_y:end_y, start_z:end_z]
    
    return crop_vol


# -------Measurement Functions-------
def estimate_noisy_pixels(img: np.ndarray):
    """Estimate the noisy pixels in the background of an image."""
    noise_threshold = -150  # Medido en [T1]
    noise_mask = (img < noise_threshold)
    return noise_mask

def power_of_signal(signal_or_img: np.ndarray) -> float:
    """ Compute the average power of a signal or image. """
    if signal_or_img.size == 0:
        return 0.0
    return np.mean(np.square(signal_or_img))


def contrast_of_signal(signal_or_img: np.ndarray) -> float:
    """ Compute the contrast of a signal or image (max - min). """
    if signal_or_img.size == 0:
        return 0.0
    return np.max(signal_or_img) - np.min(signal_or_img)

def calculate_snr(image_3d: np.ndarray, noise_mask: np.ndarray) -> float:
    """Calculate the Signal-to-Noise Ratio (SNR) of a 3D image."""
    signal_pixels = image_3d[~noise_mask]
    noise_pixels = image_3d[noise_mask]

    signal_power = power_of_signal(signal_pixels)
    noise_power = power_of_signal(noise_pixels)

    snr_value = np.sqrt(signal_power) / np.sqrt(noise_power)
    return snr_value

def calculate_cnr(image_3d: np.ndarray, noise_mask: np.ndarray) -> float:
    """Calculate the Contrast-to-Noise Ratio (CNR) of a 3D image."""
    signal_pixels = image_3d[~noise_mask]
    noise_pixels = image_3d[noise_mask]

    signal_contrast = contrast_of_signal(signal_pixels)
    noise_power = power_of_signal(noise_pixels)
    
    rms_noise = np.sqrt(noise_power)
    
    cnr_value = signal_contrast / rms_noise
    return cnr_value


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
        "--input_dir",
        type=str,
        help="Directory containing the DICOM series for the Input Image.",
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
    ct_slices_raw = []  # Store all successfully read slices before filtering
    count = 0
    pixel_len_mm = None
    num_acquisitions = 0  # Initialize num_acquisitions

    # To register the acquisition number we are using
    acquisition = -1

    for f in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, f)
        try:
            ct_dcm = pydicom.dcmread(img_path)
            # Store all successfully read slices with their ImagePositionPatient
            if hasattr(ct_dcm, "SliceLocation") and hasattr(
                ct_dcm, "ImagePositionPatient"
            ):
                ct_slices_raw.append(ct_dcm)
            elif not hasattr(ct_dcm, "SliceLocation"):
                print(f"Skipping file {f} as it lacks SliceLocation attribute.")
                count += 1
            elif not hasattr(ct_dcm, "ImagePositionPatient"):
                print(f"Skipping file {f} as it lacks ImagePositionPatient attribute.")
                count += 1

        except Exception as e:
            print(f"Could not read file {img_path}: {e}")
            count += 1

    if not ct_slices_raw:
        print("No suitable DICOM files found.")
        return [], None

    # Determine the majority ImagePositionPatient (x, y)
    image_positions = {}
    for s in ct_slices_raw:
        pos_xy = tuple(s.ImagePositionPatient[:2])
        image_positions[pos_xy] = image_positions.get(pos_xy, 0) + 1

    if not image_positions:
        print("No ImagePositionPatient data found in any slice.")
        return [], None

    majority_pos_xy = max(image_positions, key=image_positions.get)
    print(f"Majority ImagePositionPatient (x,y): {majority_pos_xy}")

    ct_slices = []

    # Filtering the slices based on the majority ImagePositionPatient (x,y)
    acquisition = -1
    slices_after_pos_filter = [
        s for s in ct_slices_raw if tuple(s.ImagePositionPatient[:2]) == majority_pos_xy
    ]

    if not slices_after_pos_filter:
        print("No slices matched the majority ImagePositionPatient (x,y).")
        return [], None

    for ct_dcm in slices_after_pos_filter:
        if acquisition == -1:  # First valid slice after position filtering
            acquisition = ct_dcm.AcquisitionNumber
            pixel_len_mm = [ct_dcm.SliceThickness] + list(ct_dcm.PixelSpacing)
            num_acquisitions = 1  # Start counting acquisitions

            if key_info:
                print("\\n--- DICOM Series Key Information (Post-Filtering) ---")
                print(f"Patient ID: {ct_dcm.get('PatientID', 'N/A')}")
                print(f"Study Description: {ct_dcm.get('StudyDescription', 'N/A')}")
                print(f"Series Description: {ct_dcm.get('SeriesDescription', 'N/A')}")
                print(f"Modality: {ct_dcm.get('Modality', 'N/A')}")
                print(f"Acquisition Number: {ct_dcm.get('AcquisitionNumber', 'N/A')}")
                print(f"Pixel Spacing (mm): {ct_dcm.get('PixelSpacing', 'N/A')}")
                print(f"Slice Thickness (mm): {ct_dcm.get('SliceThickness', 'N/A')}")
                print(f"Rows: {ct_dcm.get('Rows', 'N/A')}")
                print(f"Columns: {ct_dcm.get('Columns', 'N/A')}")
                print("------------------------------------\\n")

        slice_acquisition = ct_dcm.AcquisitionNumber
        if slice_acquisition == acquisition:
            ct_slices.append(ct_dcm)
        elif slice_acquisition != acquisition:
            if slice_acquisition not in [
                s.AcquisitionNumber
                for s in ct_slices
                if hasattr(s, "AcquisitionNumber")
            ]:  # Check if already counted
                num_acquisitions += 1
            print(
                f"Skipping slice from file {ct_dcm.filename.split('/')[-1]} due to different acquisition number ({slice_acquisition} vs {acquisition}) after position filtering."
            )
            count += 1

    # Count slices filtered out by ImagePositionPatient
    position_filtered_count = len(ct_slices_raw) - len(slices_after_pos_filter)
    count += position_filtered_count
    if position_filtered_count > 0:
        print(
            f"Skipped {position_filtered_count} slices due to non-majority ImagePositionPatient (x,y)."
        )

    # The sorting of the slices is based on the SliceLocation attribute
    final_slices = sorted(slices_after_pos_filter, key=lambda x: -x.SliceLocation)

    print(f"Loaded {len(final_slices)} slices from acquisition {acquisition}.")
    if num_acquisitions > 1:
        print(f"Detected {num_acquisitions} total acquisitions in the directory.")
    print(f"Skipped {count} files.")
    return final_slices, pixel_len_mm


# ------Visualization Functions------

def alpha_fusion(
    img: np.ndarray,
    mask: np.ndarray,
    n_objects: int,
    object_colors: List,
    alpha: float = 0.4,
) -> np.ndarray:
    """Visualize both image and mask in the same plot."""

    # Convert grayscale image to RGB using 'bone' colormap
    cmap = matplotlib.colormaps["bone"]
    norm = matplotlib.colors.Normalize(vmin=np.amin(img), vmax=np.amax(img))
    img_rgb_float = cmap(norm(img))[..., :3]  # Shape (H, W, 3), values 0.0-1.0

    # Initialize fused_slice with the RGB image.
    # We will only modify the parts covered by masks.
    fused_slice_float = img_rgb_float.copy()

    # Create a layer that will hold the specific color for each segmented pixel
    object_color_layer = np.zeros_like(img_rgb_float)
    for k in range(n_objects):
        # Boolean mask for the current object
        current_object_pixels = (mask == (k + 1))
        # Assign the object's color to these pixels in the object_color_layer
        object_color_layer[current_object_pixels] = object_colors[k]

    # Boolean mask for all pixels that are part of any segmentation
    any_segment_pixels = (mask > 0)

    # Apply alpha blending only to the pixels covered by any segment
    if np.any(any_segment_pixels): # Proceed only if there are segmented pixels
        fused_slice_float[any_segment_pixels] = (
            (1 - alpha) * img_rgb_float[any_segment_pixels] +
            alpha * object_color_layer[any_segment_pixels]
        )

    # Convert to uint8 (0-255 range)
    return (fused_slice_float * 255).astype("uint8")


def MIP_per_plane(img_dcm: np.ndarray, axis: int = 2) -> np.ndarray:
    """Compute the maximum intensity projection on the defined orientation."""
    return np.max(img_dcm, axis=axis)

def AIP_per_plane(img_dcm: np.ndarray, axis: int = 2) -> np.ndarray:
    """Compute the average intensity projection on the defined orientation."""
    return np.mean(img_dcm, axis=axis)

def visualize_AIP_per_plane(img_fused_vol: np.ndarray, pixel_len_mm: List):
    """Creates an AIP visualize for each of the axis planes."""
    labels = ["Axial Plane", "Coronal Plane", "Sagittal Plane"]
    ar_indices = [(1, 2), (0, 2), (0, 1)]
    for i in range(3):
        plt.imshow(
            AIP_per_plane(img_fused_vol, i),
            aspect=pixel_len_mm[ar_indices[i][0]] / pixel_len_mm[ar_indices[i][1]],
            vmin=np.amin(img_fused_vol),
            vmax=np.amax(img_fused_vol),
        )
        plt.title(f"AIP for {labels[i]}")
        plt.show()

def visualize_MIP_per_plane(img_fused_vol: np.ndarray, pixel_len_mm: List):
    """Creates an MIP visualize for each of the axis planes."""
    labels = ["Axial Plane", "Coronal Plane", "Sagittal Plane"]
    ar_indices = [(1, 2), (0, 2), (0, 1)]
    for i in range(3):
        plt.imshow(
            MIP_per_plane(img_fused_vol, i),
            aspect=pixel_len_mm[ar_indices[i][0]] / pixel_len_mm[ar_indices[i][1]],
            vmin=np.amin(img_fused_vol),
            vmax=np.amax(img_fused_vol),
        )
        plt.title(f"MIP for {labels[i]}")
        plt.show()

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    return ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)


def create_3d_demo(projections: List, n: int, pixel_len_mm: List):
    """creates an interactive demo to scroll through the projections."""
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    current_index = 0
    current_image = projections[current_index]
    img_plot = ax.imshow(current_image, aspect=pixel_len_mm[0] / pixel_len_mm[1])

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax_slider, "Rotation index", 0, n - 1, valinit=current_index, valstep=1
    )

    def update_image(val):
        current_index = int(val)
        current_image = projections[current_index]
        img_plot.set_data(current_image)
        fig.canvas.draw_idle()

    slider.on_changed(update_image)
    plt.show()


def create_animation(
    img_dcm: np.ndarray,
    pixel_len_mm: List,
    labels: List,
    object_colors: List,
    n=24,
    save_dir="results",
    show=True,
):
    """creates an animation by rotating the image on its sagittal plane"""

    #  Configure visualization colormap
    img_min = np.amin(img_dcm)
    img_max = np.amax(img_dcm)
    fig, _ = plt.subplots(figsize=(8, 10))

    #   Configure directory to save results
    os.makedirs(save_dir, exist_ok=True)
    #   Create projections
    projections = []
    #   Creating legend for figures
    patches = []
    object_colors = [(0, 1, 0), (1, 1, 0)]  # Red for Liver, Green for Tumor
    for k in range(len(labels)):
        patches.append(mpatches.Patch(color=object_colors[k], label=labels[k]))

    for idx, alpha in tqdm.tqdm(
        enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)), total=n
    ):
        rotated_img = rotate_on_axial_plane(img_dcm, alpha)
        projection = MIP_per_plane(rotated_img)
        plt.imshow(
            projection,
            vmin=img_min,
            vmax=img_max,
            aspect=pixel_len_mm[0] / pixel_len_mm[1],
        )
        legend = plt.legend(handles=patches, loc="center left", bbox_to_anchor=(1, 0.5))
        legend.set_title("Legend")
        plt.subplots_adjust(right=0.75)
        plt.savefig(
            os.path.join(save_dir, f"Projection_{idx}.png"), bbox_inches="tight"
        )  # Save animation
        projections.append(projection)  # Save for later animation

    # Save and visualize animation
    animation_data = [
        [
            plt.imshow(
                img,
                animated=True,
                vmin=img_min,
                vmax=img_max,
                aspect=pixel_len_mm[0] / pixel_len_mm[1],
            )
        ]
        for img in projections
    ]

    legend = plt.legend(handles=patches, loc="center left", bbox_to_anchor=(1, 0.5))
    legend.set_title("Legend")
    anim = animation.ArtistAnimation(fig, animation_data, interval=200, blit=True)
    anim.save(os.path.join(save_dir, "Animation.gif"))  # Save animation

    if show:
        plt.show()  # Show animation
    plt.close()

    create_3d_demo(projections, n, pixel_len_mm)
