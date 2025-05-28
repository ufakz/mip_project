import numpy as np
from scipy.optimize import least_squares
from typing import Optional
from utils import (
    load_config,
    parse_arguments,
    load_dicom_series,
    apply_window_level,
    create_animation,
    visualize_MIP_per_plane
)

from visualization import (
    load_segmentations,
    map_segmentation_to_slices,
    create_ordered_segmentation_mask,
    create_fused_volume,
)
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate, shift
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation as R


def min_max_normalization(vol: np.ndarray) -> np.ndarray:
    """Applies min-max normalization to the input volume."""
    min_val = np.min(vol)
    max_val = np.max(vol)
    normalized_image = (vol - min_val) / (max_val - min_val)
    return normalized_image


def mean_absolute_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """Compute the MAE between two images."""
    return np.mean(np.abs(img_input - img_reference))


def mean_squared_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """Compute the MSE between two images."""
    return np.mean((img_input - img_reference) ** 2)


def mutual_information(img_input: np.ndarray, img_reference) -> np.ndarray:
    """Compute the Shannon Mutual Information between two images."""
    nbins = [10, 10]
    # Compute entropy of each image
    hist = np.histogram(img_input.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_input = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))  # Why +1e-7?
    hist = np.histogram(img_reference.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_reference = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))  # Why +1e-7?
    # Compute joint entropy
    joint_hist = np.histogram2d(img_input.ravel(), img_reference.ravel(), bins=nbins)[0]
    prob_distr = joint_hist / np.sum(joint_hist)
    joint_entropy = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))
    # Compute mutual information
    return entropy_input + entropy_reference - joint_entropy


def visualize_landmark_differences(
    input_volume: np.ndarray, reference_volume: np.ndarray, cmap: str = "bone", general_title: Optional[str] = None
) -> None:
    """
    Visualizes the difference between the middle slice from an input volume and a reference volume.
    Also displays MAE, MSE, and Mutual Information metrics.

    """
    
    input_slice_index = input_volume.shape[0] // 2
    reference_slice_index = reference_volume.shape[0] // 2

    if not (
        0 <= input_slice_index < input_volume.shape[0]
        and 0 <= reference_slice_index < reference_volume.shape[0]
    ):
        print(f"Error: Calculated slice_index is out of bounds for the given volumes.")
        print(
            f"Input volume depth: {input_volume.shape[0]}, Reference volume depth: {reference_volume.shape[0]}"
        )
        return

    # Calculate metrics
    mae = mean_absolute_error(input_volume, reference_volume)
    mse = mean_squared_error(input_volume, reference_volume)
    mi = mutual_information(input_volume, reference_volume)

    img_input = input_volume[input_slice_index, :, :]
    img_reference = reference_volume[reference_slice_index, :, :]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Input Image
    axs[0].imshow(img_input, cmap=cmap)
    axs[0].set_title(f"Input Middle Slice ({input_slice_index})")
    axs[0].axis("off")

    # Difference Image
    diff_img = img_input - img_reference
    im = axs[1].imshow(diff_img, cmap=cmap)
    axs[1].set_title("Difference (Input - Reference)")
    axs[1].axis("off")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)  # Adjusted for better layout

    # Reference Image
    axs[2].imshow(img_reference, cmap=cmap)
    axs[2].set_title(f"Reference Middle Slice ({reference_slice_index})")
    axs[2].axis("off")

    title_text = f"Middle Slices: Input vs. Reference\\nMAE: {mae:.4f} | MSE: {mse:.4f} | Mutual Information: {mi:.4f}"
    if general_title:
        title_text = f"{general_title}\\n{title_text}"

    fig.suptitle(
        title_text,
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle
    plt.show()


def create_rgb_overlay(
    ref_vol: np.ndarray, transformed_input_vol: np.ndarray
) -> np.ndarray:
    """Creates an RGB overlay of two volumes, assuming they are normalized (0-1)."""
    if ref_vol.shape != transformed_input_vol.shape:
        raise ValueError(
            "Reference and transformed input volumes must have the same shape for overlay."
        )

    overlay_rgb = np.zeros((*ref_vol.shape, 3), dtype=ref_vol.dtype)
    overlay_rgb[..., 0] = ref_vol  # Red channel for reference
    overlay_rgb[..., 1] = transformed_input_vol  # Green channel for transformed input
    overlay_rgb[..., 2] = 0  # Blue channel (can be set to zero or another color)
    return overlay_rgb


def display_overlayed_planes(
    overlay_volume_rgb: np.ndarray,
    pixel_len_mm: list,
    ref_name: str = "Ref",
    input_name: str = "Input",
    general_title: str = None,
):
    """Displays the axial, coronal, and sagittal planes of an RGB overlayed volume."""
    depth, height, width, _ = overlay_volume_rgb.shape

    # Calculate central indices
    axial_idx = depth // 2
    coronal_idx = height // 2
    sagittal_idx = width // 2

    plt.figure(figsize=(15, 7))

    legend_text = f"Overlay: Red={ref_name}, Green={input_name}"
    
    if general_title:
        plt.suptitle(general_title + "\n" + legend_text, y=1.05)
    else:
        plt.suptitle(legend_text, y=1.02)

    plt.subplot(131)
    plt.imshow(
        overlay_volume_rgb[axial_idx, :, :, :], aspect=pixel_len_mm[1] / pixel_len_mm[2]
    )
    plt.title(f"Axial Overlay (Slice {axial_idx})")
    plt.xlabel("Width")
    plt.ylabel("Height")

    plt.subplot(132)
    plt.imshow(
        overlay_volume_rgb[:, coronal_idx, :, :],
        aspect=pixel_len_mm[0] / pixel_len_mm[2],
    )
    plt.title(f"Coronal Overlay (Slice {coronal_idx})")
    plt.xlabel("Width")
    plt.ylabel("Depth")

    plt.subplot(133)
    plt.imshow(
        overlay_volume_rgb[:, :, sagittal_idx, :],
        aspect=pixel_len_mm[0] / pixel_len_mm[1],
    )
    plt.title(f"Sagittal Overlay (Slice {sagittal_idx})")
    plt.xlabel("Height")
    plt.ylabel("Depth")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


# --- Transformation functions ---


def get_landmarks(volume: np.ndarray) -> np.ndarray:
    """Get landmark indices from the volume."""
    depth, height, width = volume.shape

    # Generate a grid of indices
    z, y, x = np.meshgrid(
        np.arange(depth), np.arange(height), np.arange(width), indexing="ij"
    )
    coordinates = np.stack((z, y, x), axis=-1).reshape(-1, 3)
    return coordinates


def find_centroid(mask: np.ndarray) -> np.ndarray:
    # Your code here:
    #   Consider using `np.where` to find the indices of the voxels in the mask
    #   ...

    idcs = np.where(mask > 0.5)
    centroid = np.stack(
        [
            np.mean(idcs[0]),
            np.mean(idcs[1]),
            np.mean(idcs[2]),
        ]
    )
    return centroid


def rigid_transformation(
    input_volume: np.ndarray,
    parameters: tuple[float, ...],
    ref_volume_shape: tuple[int, ...],
    angle_in_rads: int = 1,
) -> np.ndarray:
    """Apply a 3D rigid transformation (translations and rotations) to input_volume."""
    t1, t2, t3, v1, v2, v3 = parameters

    t1, t2, t3 = np.array([t1, t2, t3])  # * 10 - 5

    transformed_volume = shift(input_volume, (t1, t2, t3))

    transformed_volume = rotate(transformed_volume, v1, axes=(0, 1), reshape=False)
    transformed_volume = rotate(transformed_volume, v2, axes=(1, 2), reshape=False)
    transformed_volume = rotate(transformed_volume, v3, axes=(2, 0), reshape=False)

    # Crop/pad to the reference volume shape
    # transformed_volume = center_crop(transformed_volume, ref_volume_shape)

    return transformed_volume


def inverse_rigid_transformation(
    input_volume: np.ndarray,
    parameters: tuple[float, ...],
    ref_volume_shape: tuple[int, ...],
    angle_in_rads: int = 1, # This parameter seems unused in the original, keeping for consistency
) -> np.ndarray:
    """Apply the inverse of a 3D rigid transformation to input_volume."""
    t1, t2, t3, v1, v2, v3 = parameters

    # Inverse rotations (applied in reverse order with negated angles)
    transformed_volume = rotate(input_volume, -v3, axes=(2, 0), reshape=False)
    transformed_volume = rotate(transformed_volume, -v2, axes=(1, 2), reshape=False)
    transformed_volume = rotate(transformed_volume, -v1, axes=(0, 1), reshape=False)

    # Inverse translation
    transformed_volume = shift(transformed_volume, (-t1, -t2, -t3))

    # Crop/pad to the reference volume shape (if needed, similar to original)
    # transformed_volume = center_crop(transformed_volume, ref_volume_shape)

    return transformed_volume


errors = []
def coregister_volumes(
    input_volume: np.ndarray,
    ref_volume: np.ndarray,
    optimizer: Optional[str] = "GD",
    initial_parameters: tuple[float, ...] = None,
) -> np.ndarray:
    """Apply a 3D rigid transformation (translations and rotations) to input_volume."""

    def function_to_minimize(parameters):
        transformed_volume = rigid_transformation(
            input_volume, parameters, ref_volume.shape
        )
        error = mean_squared_error(ref_volume, transformed_volume)
        errors.append(error)
        return error

    if optimizer == "GD":
        # Apply gradient descent optimization
        result = minimize(function_to_minimize, x0=initial_parameters)
    else:
        # Apply least squares optimization
        print(initial_parameters)
        result = least_squares(function_to_minimize, x0=initial_parameters, verbose=2)

    return result


def estimate_rigid_transform_from_points(points_m, points_f):
    """
    Estimates a rigid 3D transformation (R, t) from 8 or more
    corresponding points using Singular Value Decomposition (SVD).
    """
    N = points_m.shape[0]

    centroid_m = np.mean(points_m, axis=0)
    centroid_f = np.mean(points_f, axis=0)

    points_m_centered = points_m - centroid_m
    points_f_centered = points_f - centroid_f

    H = points_m_centered.T @ points_f_centered  # (3, N) @ (N, 3) -> (3, 3)

    U, S, Vh = np.linalg.svd(H)

    V = Vh.T  # Get the V matrix

    R = V @ U.T

    if np.linalg.det(R) < 0:
        V[:, 2] *= -1  # Flip the sign of the last column of V
        R = V @ U.T  # Recalculate R with the modified V

    t = centroid_f - R @ centroid_m

    return R, t


VISUALIZE = False

# Object names and color configuration
objects = ["Liver", "Tumor"]
object_colors = [(0, 1, 0), (1, 0, 0)]  # Red for Liver, Green for Tumor


def main():

    args = parse_arguments()
    config = load_config(args.config)

    dataset_dir = args.dataset_dir or config.get("dataset_dir")
    input_dir = args.input_dir or config.get("input_dir")
    output_dir = args.output_dir or config.get("output_dir", "results/animation")
    segment_dir = args.segment_dir or config.get("segment_dir")

    visualize_flag = args.visualize or config.get("visualize", VISUALIZE)

    if not dataset_dir or not input_dir:
        raise ValueError(
            "Reference directory and input directory path must be provided via command line or config file."
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    try:
        #### Step 1: Load the reference and input DICOM series
        ref_slices, ref_pixel_len_mm = load_dicom_series(dataset_dir, key_info=True)
        input_slices, input_pixel_len_mm = load_dicom_series(input_dir, key_info=True)

        # Check if pixel spacing is consistent between reference and input volumes
        if ref_pixel_len_mm != input_pixel_len_mm:
            print(
                f"Pixel spacing mismatch: Reference={ref_pixel_len_mm}mm, Input={input_pixel_len_mm}mm"
            )
        else:
            print(f"Pixel spacing is consistent in both images: {ref_pixel_len_mm}mm")

        ref_volume = np.array([f.pixel_array for f in ref_slices])
        input_volume = np.array([f.pixel_array for f in input_slices])

        print(f"Ref volume shape: {ref_volume.shape}")
        print(f"Input volume shape: {input_volume.shape}")

        #### Step 2: Preprocessing
        ref_volume_processed, _ = apply_window_level(ref_volume, window=400, level=40)
        input_volume_processed, _ = apply_window_level(
            input_volume, window=400, level=40
        )

        ref_volume_normalized = min_max_normalization(ref_volume_processed)
        input_volume_normalized = min_max_normalization(input_volume_processed)

        #### Step 3: Load the segmentation mask
        
        # Load Segmentation
        seg_dcm = load_segmentations(segment_dir)

        # Map Segmentation to Series
        valid_masks = map_segmentation_to_slices(seg_dcm)

        # Create Ordered Mask Volume
        reordered_seg = create_ordered_segmentation_mask(
            input_volume_processed, seg_dcm, valid_masks
        )

        #### Step 4: Get initial parameters (gotten from manual landmark selection)
        ref_landmarks = [
            [37, 214, 255],
            [8, 156, 255],
            [23, 167, 230],
            [23, 261, 251],
            [6, 263, 255],
            [21, 166, 300],
            [21, 194, 294],
            [21, 331, 293],
        ]

        inp_landmarks = [
            [39, 204, 254],
            [8, 153, 255],
            [23, 170, 234],
            [23, 262, 248],
            [8, 264, 255],
            [22, 166, 301],
            [22, 196, 296],
            [22, 328, 297],
        ]

        Rot_mat, t = estimate_rigid_transform_from_points(
            np.array(ref_landmarks), np.array(inp_landmarks)
        )
        
        rotations = R.from_matrix(Rot_mat).as_euler("xyz", degrees=True)

        initial_roll_rad = rotations[0]
        initial_pitch_rad = rotations[1]
        initial_yaw_rad = rotations[2]

        print(f"Initial Translation: {t}")
        print(
            f"Initial Roll (rad): {initial_roll_rad}, Pitch (rad): {initial_pitch_rad}, Yaw (rad): {initial_yaw_rad}"
        )

        initial_parameters = [
            t[0],
            t[1],
            t[2],
            initial_roll_rad,
            initial_pitch_rad,
            initial_yaw_rad,
        ]

        transformed_input = rigid_transformation(
            input_volume_processed, initial_parameters, ref_volume.shape
        )

        if visualize_flag:
            visualize_landmark_differences(transformed_input, ref_volume_processed, general_title="Before Registration")
            
            overlayed_before_reg = create_rgb_overlay(
                ref_volume_normalized, input_volume_normalized
            )
            
            display_overlayed_planes(
                overlayed_before_reg,
                ref_pixel_len_mm,
                ref_name="Reference",
                input_name="Input",
                general_title="Overlayed Input Volume Before Registration",
            )

        second_fused = create_fused_volume(
            transformed_input,
            reordered_seg,
            objects,
            object_colors,
            ref_pixel_len_mm,
            visualize_slices=visualize_flag,
        )

        #### Step 5: Perform coregistration
        # result = coregister_volumes(
        #     input_volume_normalized,
        #     ref_volume_normalized,
        #     optimizer="GD",
        #     initial_parameters=initial_parameters,
        # )

        # t1, t2, t3, v1, v2, v3 = result.x
        # best_params = result.x

        # print("Best parameters:")
        # print(f"  >> Translation: ({t1:0.02f}, {t2:0.02f}, {t3:0.02f}).")
        # print(
        #     f"  >> Rotation: ({v1:0.02f} in axis 0, {v2:0.02f} in axis 1, {v3:0.02f}) in axis 2."
        # )
        
        # # Plot the errors
        # if errors:
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(errors, label="Normalized MSE")
        #     plt.title('Normalized MSE vs. Iterations')
        #     plt.xlabel('Iteration')
        #     plt.ylabel('Normalized MSE')
        #     plt.grid(True)
        #     plt.show()

        ###### Step 6: Apply the best transformation to the input volume
        # First eval
        # best_params = [0.24, -0.19, 0.25, 0.14, 0.02, -0.10] # From the output of the optimizer
        
        # Least Squares
        # best_params = [0.00, -0.23, 0.44, -0.03, 0.00, 0.04]
        
        # Gradient Descent
        best_params = [0.27, -0.18, 0.15, 0.16, -0.00, 0.10]
        
        best_transformed_input = rigid_transformation(
            input_volume_processed, best_params, ref_volume.shape
        )
        
        best_transformed_input_normalized = min_max_normalization(best_transformed_input)
        
        if visualize_flag:
            visualize_landmark_differences(
                best_transformed_input, ref_volume_processed, general_title="After Registration"
            )
            
            overlayed_after_reg = create_rgb_overlay(   
                ref_volume_normalized, best_transformed_input_normalized
            )
            
            display_overlayed_planes(
                overlayed_after_reg,
                ref_pixel_len_mm,
                ref_name="Reference",
                input_name="Input",
                general_title="Overlayed Input Volume After Registration",
            )
            
        ##### Step 7: Get the inverse transformation to convert the mask to the input space
        inverse_transformed_mask = inverse_rigid_transformation(
            reordered_seg, best_params, input_volume.shape
        )
        
        ##### Step 8: Create the final fused volume
        final_fused_preprocessed, _ = apply_window_level(
            best_transformed_input, window=800, level=400
        )
        
        
        final_fused_mip = create_fused_volume(
            final_fused_preprocessed,
            inverse_transformed_mask,
            objects,
            object_colors,
            ref_pixel_len_mm,
            visualize_slices=visualize_flag,
        )
        
        if visualize_flag:
            print("Visualizing MIPs...")
            visualize_MIP_per_plane(final_fused_mip, ref_pixel_len_mm)
        
        # ##### Step 9: Create animation
        # create_animation(
        #     final_fused_mip,
        #     pixel_len_mm=ref_pixel_len_mm,
        #     labels=objects,
        #     object_colors=object_colors,
        #     n=24,
        #     save_dir="results/registration_animation",
        #     show=visualize_flag,
        # )
        

    except (FileNotFoundError, ValueError, IOError) as e:
        import traceback

        print(f"An error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
