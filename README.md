# MSfM-3D-Scene-Reconstruction

This project implements a complete Structure from Motion (SfM) pipeline to reconstruct a 3D scene from a sequence of 2D images. The project is divided into phases, moving from initial feature matching and pose estimation (Phase 1) to a refined implementation using Bundle Adjustment and interactive 3D visualization (Phase 2).

# Project Overview

The core objective is to determine the 3D structure of a scene and the trajectory of the camera simultaneously. This is achieved by tracking feature points across image frames, triangulating their 3D positions, and optimizing the results to minimize reprojection errors.

# Key Features:

Feature Extraction: Detection and matching of keypoints across image sequences.
Camera Pose Estimation: recovering rotation ($R$) and translation ($t$) matrices.
Triangulation: Converting 2D image points into a 3D Point Cloud.
Bundle Adjustment: A non-linear optimization step using scipy.optimize to refine camera poses and 3D point coordinates simultaneously.
Sparse Jacobian Optimization: Implementation of a sparsity matrix to make Bundle Adjustment computationally feasible for large datasets.
Interactive Visualization: 3D rendering of the point cloud and camera frustums using Open3D.

# Code Structure & Pipeline

The project logic is encapsulated primarily in CV-Project-Phase2-ver2.ipynb. Below is the detailed breakdown of the internal code architecture:

## 1. Dependencies and Setup

The notebook relies on standard Computer Vision and scientific computing libraries:

cv2 (OpenCV): For image processing and feature handling.

numpy: For linear algebra and matrix operations.

scipy.optimize: specifically least_squares for the Bundle Adjustment minimization.

scipy.sparse: For constructing the Jacobian sparsity pattern.

matplotlib: For 2D plotting and initial debug views.

open3d: For high-fidelity 3D visualization and the "Virtual Tour".

## 2. Phase 1: Initialization & Helper Functions

This section contains the building blocks for the reconstruction:

Feature Matching: Functions to detect SIFT/ORB features and match them between adjacent frames using FLANN or BruteForce matchers.

Outlier Rejection: Usage of RANSAC (Random Sample Consensus) to remove bad matches during Fundamental/Essential matrix estimation.

Pose Retrieval: Extracting the initial Rotation ($R$) and Translation ($t$) from the Essential Matrix.

## 3. Phase 2: Bundle Adjustment (Optimization)

This is the core of the project. Simply triangulating points leads to "drift" and error accumulation. Bundle Adjustment refines estimates by minimizing the reprojection error.

Parameter Vectorization: The code flattens all parameters (camera extrinsic parameters and 3D point coordinates) into a single 1D array required by the optimizer.

Reprojection Error Function (fun):

Projects the estimated 3D points back onto the 2D image planes using the current camera parameters.

Calculates the residual (difference) between observed 2D points and projected points.

Bundle Adjustment Sparsity (bundle_adjustment_sparsity):

Because the interaction matrix (Jacobian) is extremely large but mostly empty (a 3D point is only seen by a few cameras), a Sparse Matrix (scipy.sparse.lil_matrix) is constructed.

This significantly speeds up the least_squares solver by indicating exactly which parameters affect which residuals.

## 4. 3D Visualization (The Virtual Tour)

The final section handles the rendering of the reconstructed scene.

Point Cloud Generation: The optimized 3D points ($X, Y, Z$) are converted into an Open3D PointCloud object.

Camera Frustums: The code iterates through the solved camera poses and draws 3D axes/frustums to represent the camera's path through the scene.

Rendering: Uses o3d.visualization.draw to render the interactive scene.

### Note: The code includes try-except blocks to handle rendering contexts, ensuring it works in different environments (e.g., enabling WebRTC for remote notebooks).

# Setup

git clone https://github.com/arslanamjad19/MSfM-3D-Scene-Reconstruction.git

cd MSfM-3D-Scene-Reconstruction

## Install Requirements:
It is recommended to use a virtual environment or conda environment
pip install numpy opencv-python matplotlib scipy open3d


## Data Preparation:
Ensure your image dataset is placed in the correct directory as referenced in the "Imports" or "Data Loading" section of the notebook.

## Execute the Notebook:
Launch Jupyter and run the cells sequentially.

jupyter notebook CV-Project-Phase2-ver2.ipynb

# Results:
The output of the notebook will include:
2D Feature Matches: Visualizations of keypoints matched between frames.
Optimization Logs: Output from scipy.optimize showing the reduction in cost (residual error) after Bundle Adjustment.
Interactive 3D Window: An Open3D window displaying:
Blue points: The reconstructed 3D structure.
Camera Axes: Representing the trajectory and orientation of the camera during the virtual tour.

# 🛠 Tech Stack

OpenCV
Core Logic: Structure from Motion (SfM)
Optimization: Levenberg-Marquardt (via SciPy)
Visualization: Open3D
