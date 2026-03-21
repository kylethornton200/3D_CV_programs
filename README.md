# 3D CV Programs

These are one off programs that helped me understand 3D CV as well as implementations. (Will update with each commit.)

## Current Structure

```
.
├── 2_camera_estimation/
│   ├── pipeline.py
│   ├── output/
├── cam_projection_visualizer/
│   ├── main.cpp 
```

### Camera Projection Visualizer
A one off program that helps visualize the intrinsic and extrinsic factors of projecting a cube onto the 2D plane.
```cpp
Camera camera(800, 800, 320, 240); // K or the intrinsic matrix of the camera.
Eigen::MatrixXd cube = createCube(.75); // Changes the cube size.
Eigen::Matrix3d R = rotationFromEuler(45, 45, 45); // Rotation matrix R. Changes the orientation of the camera relative to the cube.
Eigen::Vector3d t(0, 0, 3); // Translation vector T of the 3x4 matrix that will change where the camera is relative to 0,0,0
```
This is the code block that will change things.

### 2 Camera Scene Visualizer
Another one off program about 3d cv (make sure you have an output dir in the 2_camera_estimation folder)
#### What it does
1. Generates a synthetic scene — random 3D points plus a cube surface, projected through two cameras with different intrinsics and a known relative pose (30° rotation).
2. Adds realistic noise — Gaussian perturbation (σ = 0.8 px) and 20% random outlier correspondences.
3. Estimates the fundamental matrix via multiple methods:
   * Normalized 8-point algorithm (Hartley normalization for numerical stability)
   * 7-point minimal solver (cubic root finding for the det(F) = 0 constraint)
   * RANSAC with adaptive iteration count using Sampson distance
   * MLE refinement via Levenberg-Marquardt
4. Recovers camera pose — extracts the essential matrix, decomposes into four (R, t) candidates, and selects via cheirality testing.
5. Triangulates 3D points — both linear DLT and Hartley-Sturm optimal correction.
6. Produces diagnostic visualizations — epipolar lines, inlier/outlier matches, 3D reconstruction comparison, error histograms, and SVD structure of F.

#### Current deficiencies
1. No bundle adjustment
2. Pose recover uses noises points for the cheirality test
3. Limited camera intrinsics and limited to 2 cameras

### NEXT:
Doing monocular depth estimation of my workspace with depth anything V2. **Goal is seeing how high I can get fps on my GPU using my webcame as the input.**