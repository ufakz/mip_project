# Medical Image Processing Project

Final Project of Medical Image Processing at UIB.

## Project Structure

```
mip_project/
├── data/                 # Contains input DICOM series and segmentation masks
│   ├── Reference_Series/ # Reference DICOM series
│   ├── Input_Series/     # Input DICOM series for coregistration
│   └── masks/            # Segmentation masks (e.g., Liver, Tumor)
├── results/              # Output directory for results (e.g., animations)
│   └── animation_output/
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── config.yaml           # Configuration file for script parameters
├── coregistration.py     # Script for coregistering two DICOM volumes
├── visualization.py      # Script for visualizing DICOM series with segmentations
├── utils.py              # Utility functions for loading data, configuration, etc.
└── README.md             # This file
```

## Configuration

The `config.yaml` file allows you to set default paths and parameters for the scripts:

-   `dataset_dir`: Path to the reference DICOM series directory.
-   `input_dir`: Path to the input DICOM series directory (for coregistration).
-   `segment_dir`: Path to the directory containing DICOM segmentation files.
-   `output_dir`: Path where output files (like animations) will be saved.
-   `visualize`: Boolean flag (`true`/`false`) to enable/disable intermediate visualizations.

Command-line arguments can override the settings in `config.yaml`.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/ufakz/mip_project
    cd mip_project
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install pydicom numpy matplotlib scipy pyyaml
    ```
4.  **Place your data:**
    -   Put your reference DICOM series in a directory (e.g., `data/reference_series`).
    -   Put your input DICOM series (for coregistration) in another directory (e.g., `data/input_series`).
    -   Put your DICOM segmentation files in a directory (e.g., `data/segmentations`).
    -   Update `config.yaml` or use command-line arguments to point to these directories.

## Usage

### Visualization

This visualizes the loaded `reference image` with the associated `liver` and `tumor` masks.

**NB**: Animation part is WIP.

To visualize a DICOM series with its segmentations:

```bash
python visualization.py --dataset_dir path/to/dicom/series --segment_dir path/to/segmentations --visualize
```

Or using the configuration file:

```bash
python visualization.py --config config.yaml --visualize
```

### Coregistration

To coregister an input DICOM volume to a reference volume:

```bash
python coregistration.py --dataset_dir path/to/reference/series --input_dir path/to/input/series --visualize
```

Or using the configuration file:

```bash
python coregistration.py --config config.yaml --visualize
```

**Note:** In the coregistration script, currently have a placeholder logic for the optimization step and uses Sum of Squared Differences (SSD) for residuals. Will be improving using techniques like Mutual Information and better optimization methods.

## Data Requirements

-   Input data should be in DICOM format.
-   The scripts expect DICOM series organized in directories, where each file represents a slice.
-   Segmentation data should also be in DICOM format (specifically, DICOM Segmentation Objects).
-   The `load_dicom_series` function attempts to load slices belonging to the same acquisition number based on the first valid slice encountered.
