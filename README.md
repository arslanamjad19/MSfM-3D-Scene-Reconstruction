# MSfM-3D-Scene-Reconstruction

A complete **Structure from Motion (SfM)** pipeline that reconstructs 3D scenes from 2D image sequences, featuring multi-view reconstruction merging and an interactive web-based virtual tour viewer.

## Project Overview

This project implements a full SfM pipeline that:
- Extracts and matches features across image sequences
- Estimates camera poses using PnP and RANSAC
- Triangulates 3D points from 2D correspondences
- Refines reconstructions using Bundle Adjustment with sparse Jacobian optimization
- Merges multiple partial reconstructions using interactive point cloud alignment
- Provides an immersive Photosynth-style web viewer for exploring reconstructed scenes

## Key Features

### Phase 1 & 2: Core Reconstruction
- **Feature Detection**: SIFT/ORB keypoint extraction and matching with Lowe's ratio test
- **Robust Pose Estimation**: Essential matrix decomposition with RANSAC outlier rejection
- **Sequential SfM**: Incremental reconstruction using PnP with multiple reference frames
- **Smart Triangulation**: Parallax-based point filtering with reprojection error validation
- **Bundle Adjustment**: Sparse Jacobian optimization using `scipy.optimize.least_squares`
- **Quality Filtering**: Automatic outlier removal based on reprojection errors and 3D spatial distribution

### Phase 3: Reconstruction Merging & Visualization
- **Interactive Alignment**: Manual point correspondence selection using Open3D
- **Similarity Transform**: Umeyama's method for scale-rotation-translation estimation
- **Multi-View Merging**: Combine multiple partial reconstructions into unified scene
- **Web Viewer**: Three.js-based interactive tour with:
  - Smooth camera transitions with lerp/slerp interpolation
  - View graph navigation (click to jump to best neighbor)
  - Cross-fading image transitions
  - Point cloud visibility toggle
  - Full-screen image display

## 🛠️ Tech Stack

- **OpenCV** - Feature detection, matching, and camera calibration
- **NumPy** - Linear algebra and matrix operations
- **SciPy** - Bundle Adjustment optimization with sparse matrices
- **Open3D** - Point cloud visualization and interactive alignment
- **Three.js** - Web-based 3D rendering
- **Matplotlib** - Data visualization and debugging

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/arslanamjad19/MSfM-3D-Scene-Reconstruction.git
cd MSfM-3D-Scene-Reconstruction
```

2. **Create a virtual environment (recommended)**
```bash
# Using venv
python -m venv sfm_env
source sfm_env/bin/activate  # On Windows: sfm_env\Scripts\activate

# OR using conda
conda create -n sfm_env python=3.9
conda activate sfm_env
```

3. **Install dependencies**
```bash
pip install numpy opencv-python matplotlib scipy open3d
```

### Prepare Your Data

Place your image sequence in the `Images/test_set/` directory:
```
Images/test_set/
├── Img1.jpg
├── Img2.jpg
├── Img3.jpg
└── ...
```

**Image Requirements:**
- Sequential overlapping views of the scene
- Consistent naming convention: `Img{N}.jpg` (or .jpeg, .png)
- Sufficient overlap between adjacent frames (>30%)
- Good lighting and texture for feature matching

## 📊 Usage

### Phase 2: Single Reconstruction

1. **Configure parameters in `phase2.py`:**
```python
folder = "Images/test_set"      # Input image directory
indices = list(range(1, 34))     # Image indices (Img1 to Img33)
algo = "SIFT"                    # Feature detector: "SIFT" or "ORB"
output_dir = "Ply_pose1_DLC"     # Output directory name
```

2. **Run the reconstruction:**
```bash
python phase2.py
```

3. **Outputs:**
   - `sequence_12views_sparse_raw.ply` - Raw 3D point cloud
   - `sequence_12views_sparse_filtered.ply` - Filtered point cloud
   - `cameras.json` - Camera poses and parameters
   - `points.json` - Point cloud for web viewer
   - `view_graph.json` - Camera connectivity graph
   - `index.html` - Interactive web viewer

### Phase 3: Merge Multiple Reconstructions

If you have multiple partial reconstructions (e.g., different walls of a room):

1. **Run reconstructions separately:**
```bash
# Edit phase2.py: set output_dir = "Ply_pose1_DLC"
python phase2.py

# Edit phase2.py: set output_dir = "Ply_pose2_DLC" (with different images)
python phase2.py
```

2. **Merge using interactive alignment:**
```bash
python phase3_merge_complete.py
```

**Merging Instructions:**
- The script will open two Open3D windows sequentially
- In each window, **Shift + Left Click** to pick corresponding landmark points
- Pick **4-6 clearly identifiable points** in the **same order** in both clouds
- Common landmarks work best (corners, edges, distinctive features)
- Press **Q** when done with each window
- The script automatically computes and applies the transformation

3. **Result:** Merged reconstruction in `Ply_merged_DLC/`

### View Results

**Option 1: Open3D Viewer (automatic)**
- Automatically opens during `phase2.py` execution
- Shows point cloud and camera frustums

**Option 2: Web Viewer (recommended)**
```bash
cd Ply_pose1_DLC  # or Ply_merged_DLC
python -m http.server 8000
```
Then open: `http://localhost:8000/index.html`

**Web Viewer Controls:**
- **Click** on view to navigate to best neighbor camera
- **Next/Prev** buttons for sequential navigation
- **Show point cloud** checkbox to toggle visibility
- **Reset** button to return to first camera

## 🔧 Tuning Parameters

Key parameters in `phase2.py`:

### Feature Matching
```python
algo = "SIFT"              # "SIFT" (accurate) or "ORB" (faster)
target_w = 3000            # Resize width for processing
```

### PnP Thresholds
```python
min_pnp_pts = 20           # Minimum 3D-2D correspondences
min_pnp_inliers = 60       # Minimum inliers to accept pose
pnp_reproj_err_px = 2.0    # RANSAC reprojection threshold
```

### Triangulation Quality
```python
triang_E_thresh_px = 1.5        # Essential matrix RANSAC threshold
triang_reproj_thresh_px = 2.0   # Max reprojection error for new points
triang_min_parallax_deg = 2.0   # Minimum parallax angle
```

### Bundle Adjustment
```python
MAX_POINTS = 3000          # Max points for BA (performance vs accuracy)
```

## 📈 Pipeline Details

### 1. Initialization (Bootstrap)
- Match features between first two images
- Estimate Essential matrix with RANSAC
- Decompose to 4 possible [R|t] solutions
- Select pose using cheirality test (points in front of both cameras)

### 2. Sequential Camera Addition
- For each new image:
  - Match against previously posed cameras
  - Build 3D-2D correspondences from existing map
  - Estimate pose using PnP RANSAC
  - Triangulate new points with quality checks
  - Fall back to earlier reference frames if needed

### 3. Bundle Adjustment
- Simultaneously optimize all camera poses and 3D points
- Minimize reprojection error: `Σ ||projected(X_i) - observed(x_i)||²`
- Uses sparse Jacobian for efficiency (most derivatives are zero)
- Huber loss for robustness to outliers

### 4. Post-Processing
- Filter points by reprojection error (median ≤ 2.0 px)
- Require points visible in ≥2 views
- Remove 3D spatial outliers (>97th percentile distance from median)

## 📝 Output Files Explained

- **`.ply` files**: Standard point cloud format (open with MeshLab/CloudCompare)
- **`cameras.json`**: Camera positions, orientations (quaternions), and extrinsics
- **`points.json`**: Flat array of 3D coordinates for web rendering
- **`view_graph.json`**: Connectivity between cameras (shared visible points)
- **`index.html`**: Self-contained web viewer (requires local server)
