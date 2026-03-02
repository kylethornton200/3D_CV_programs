# 3D CV Programs

These are one off programs that helped me understand 3D CV as well as implementations. (Will update with each commit.)

## Current Structure

```
.
├── cam_projection_visualizer/              # 
│   ├── main.cpp                 # main file for visualizing camera position and intrinsic matrix effect on a 3D cube.
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