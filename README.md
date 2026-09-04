## Notes on Running the Code

### System Requirements

1. **Plotting scripts** (`Objective Function - Plotting.py`, `plot_analysis_figures.py`, `Plot - Sample Green Signal Ratio Distribution.py`):
   - **Windows** is recommended to ensure that the Times New Roman font renders correctly.
   - These scripts can also run on **Linux**; however, if the Times New Roman font is not installed on your system, it may be substituted with a fallback font.

2. **Computation scripts** (all other `.py` files):
   - Must be run on **Linux**.
   - A GPU with **CUDA** support is required.

### Pre-computed Results

Due to file size limitations on GitHub, some pre-computed results (`.npz` files) are not included in this repository. You will need to run the corresponding computation scripts to generate these files before running the plotting scripts.

### Estimated Runtime

Under the reference hardware configuration listed below, the total runtime for all computation scripts is approximately **50–60 hours**:

| Component | Specification |
|-----------|--------------|
| CPU       | Intel i9-13900H |
| RAM       | 64 GB |
| GPU       | NVIDIA RTX 4060 |

### Python Package Versions

The code was developed and tested with the following package versions:
Python   : 3.12.3
numpy    : 2.5.2
matplotlib: 3.11.1
scipy    : 1.18.0
numba    : 0.67.0
cupy     : 14.2.0
