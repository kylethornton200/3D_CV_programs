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
├── ekf_attempt/
│   ├── lie_ekf_examples.cpp
│   ├── user_lie.hpp
│   ├── CmakeLists.txt
├── sophus_attempt/
│   ├── main.cpp
│   ├── CmakeLists.txt
├── webcam_depth_estimator/
│   ├── reading_webcam.py
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

### Sophus Attempt (SO(3) / SE(3) from scratch)
A from-scratch rewrite of the core Lie group math from [Sophus](https://github.com/strasdat/Sophus), using only Eigen. Sophus is linked in only as a reference to test against.

#### What's implemented
* `hat`: 3-vector to skew-symmetric matrix.
* `expSO3` / `logSO3`: Rodrigues' formula for exp. Log goes through a quaternion so it stays stable near θ = 0 and θ = π, and picks w ≥ 0 so θ ∈ [0, π].
* `leftJacobian` / `rightJacobian` and their inverses, with small-angle fallbacks. J_l uses `2 sin²(θ/2)/θ` instead of `(1 - cos θ)/θ` to avoid cancellation.
* `expSE3` / `logSE3`: twists are ordered `xi = [rho; phi]` (translation first, rotation second, same as Sophus). The translation is `J_l(phi) * rho`.
* `adjointSE3`: the 6x6 adjoint `[C, [r]x C; 0, C]`.
* `inverseSE3`: the analytic inverse `[Cᵀ, -Cᵀr; 0, 1]`.

#### Tests
`main()` runs every test group with a fixed seed (42), prints the worst error for each check, then prints `PASS`/`FAIL` (exit code 0/1). A failing group doesn't stop the rest from running. Random rotations use angles in [0, π], and every 5th sample uses a tiny angle (1e-12 to 1e-6) to hit the small-angle branches.
1. **SO(3) exp/log**: exp vs Sophus, orthogonality, det = 1, round trip, log vs Sophus, and `exp(log(R)) = R`.
2. **SO(3) log edge cases**: θ = 0, 1e-10, π - 1e-6, and π about each axis.
3. **Jacobians**: J_l and J_r vs central finite differences, plus `J_r = J_lᵀ = J_l(-phi)` and `J_l phi = phi`.
4. **Jacobian inverses**: `J J⁻¹ = I`, `J_l⁻¹ phi = phi`, and `J_r⁻¹ = (J_l⁻¹)ᵀ`.
5. **SE(3) exp/log**: rotation, translation and bottom-row checks, round trip, and log vs Sophus. For small angles the translation is checked against a Taylor series instead of Sophus, because Sophus's own `SE3::exp` loses precision for θ ≈ 1e-10 to 1e-5.
6. **SE(3) log edge cases**: the same edge angles with several translations, plus pure translations (log must give exactly `[t; 0]`).
7. **Adjoint**: `T exp(xi) T⁻¹ = exp(Ad(T) xi)`, matches Sophus's `Adj()`, `Ad(T T2) = Ad(T) Ad(T2)`, `Ad(T⁻¹) = Ad(T)⁻¹`, and `Ad(I) = I`.

Tolerances: 1e-12 for SO(3), 1e-11 for SE(3) (translations are O(10)), 1e-10 for the chained adjoint products, and 1e-9 for finite differences.

#### Building
Needs Eigen3 and Sophus findable by CMake (`find_package(... CONFIG)`), e.g. through vcpkg.
```
cmake -S sophus_attempt -B sophus_attempt/build
cmake --build sophus_attempt/build
```


### EKF Attempt (Lie group EKF on SO(3) / SE(3))
Extended Kalman filters that live on the group instead of a vector space, built on the Lie functions from the Sophus attempt. Only needs Eigen.

Conventions follow Barfoot: the state is `C_vi` / `T_vi` (world to vehicle), perturbations are on the left (`T = exp(dxi^) * T̄`), and twists are `xi = [rho; phi]`.

#### Files
* `user_lie.hpp`: the SO(3)/SE(3) functions from `sophus_attempt` (`hat`, `expSO3`/`logSO3`, Jacobians, `expSE3`/`logSE3`, `adjointSE3`) turned into a header. The math is unchanged.
* `lie_ekf_examples.cpp`: the extra pieces an EKF needs (`hatSE3`, the `odot` operator, `invSE3`, a generic `ekfGain`) and two worked examples.

#### Examples
1. **Identity checks**: `hatSE3(xi) p = odot(p) xi` and `T exp(x^) = exp((Ad(T) x)^) T` over 1000 random samples.
2. **SO(3) attitude EKF**: a gyro predicts a 90° yaw (with a small bias against the truth), so `F = Ad(Psi) = Psi`. The correction uses known world "up" and "north" vectors seen in the body frame, with `G = [-(C u)^; -(C m)^]`. The update is applied on the group (`C = exp(dphi^) C̃`). Prints the attitude error before and after the correction, the sigmas, and an orthogonality check.
3. **SE(3) pose EKF (planar case)**: odometry drives 2 m forward and turns 90° left, so `F = Ad(Xi)`. One known landmark is measured in the vehicle frame, with `G = Dᵀ (T̃ p)^odot`. Prints the correction `dxi`, the robot's pose in the world, the planar sigmas, and an orthogonality check.

#### Building
```
cmake -S ekf_attempt -B ekf_attempt/build
cmake --build ekf_attempt/build
```
