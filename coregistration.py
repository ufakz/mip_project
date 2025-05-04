import os
import numpy as np
from utils import load_config, load_dicom_series, parse_arguments, alpha_fusion, visualize_MIP_per_plane
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from scipy.optimize import least_squares

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
    
    
def rigid_transformation(volume: np.ndarray, parameters: tuple[float, ...], angle_in_rads: int = 360) -> np.ndarray:
    """ Apply to `volume` a translation followed by an axial rotation defined by `parameters`. """
    t1, t2, t3, v1, v2, v3 = parameters
    # Scale to original values
    t1, t2, t3 = np.array([t1, t2, t3])
    
    # apply transformation
    trans_volume = shift(volume,(t1, t2, t3))
    trans_volume = rotate(trans_volume, angle_in_rads * v3, (0, 1))
    trans_volume = rotate(trans_volume, angle_in_rads * v1, (1, 2))
    trans_volume = rotate(trans_volume, angle_in_rads * v2, (2, 0))
    
    return trans_volume

#Note: Using SSD for now. Will change to MI later
def vector_of_residuals(ref_points: np.ndarray, inp_points: np.ndarray) -> np.ndarray:
    """ Given arrays of 3D points with shape (point_idx, 3), compute vector of residuals as their respective distance """
    # Your code here:
    #   ...
    # Ensure the volumes have the same shape
    
    diff = ref_points - inp_points
        
    return diff.flatten()

def coregister_landmarks(ref_landmarks: np.ndarray, inp_landmarks: np.ndarray):
    """ Coregister two sets of landmarks using a rigid transformation. """
    # Your code here:
    # ....
    initial_parameters = [
        0.0, 0.0, 0.0,  
        0.0, 0.0, 0.0,   
    ]

    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""
        inp_landmarks_transf = rigid_transformation(inp_landmarks, parameters)
        return vector_of_residuals(ref_landmarks, inp_landmarks_transf)

    # Apply least squares optimization
    result = least_squares(
        function_to_minimize,
        x0=initial_parameters,
        verbose=2)
    
    return result

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
        display_planes(input_volume, input_pixel_len_mm)
        
        # Step 3: Apply rigid transformation to the input volume
        #sample_parameters = (10, 20, -5, 0.25, 0.7, 0.15) 
        #transformed_volume = rigid_transformation(input_volume, sample_parameters)
        
        #print(f"Transformed volume shape: {transformed_volume.shape}")
        
        #display_planes(transformed_volume, input_pixel_len_mm)
        
        # Step 4: Apply the optimization algorithm
        # result = coregister_landmarks(ref_volume, input_volume)
        # solution_found = result.x
        # print(f"Solution found: {solution_found}")
            
            
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
if __name__ == "__main__":
    main()