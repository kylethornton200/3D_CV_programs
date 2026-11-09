# Assumptions in `lie_ekf_examples.cpp`

**Claude Wrote This and I will have to go through and make sure I want/need all of it. For now it is an enumeration of the stated (in comments) or implicit assumptions I made in the EKF calculations. From a cursory glance it looks as though I made some implicit assumptions I don't want to have made.**

Notation follows the source comments: `x^` is the hat operator, `(.)check` is a predicted quantity, and `(.)hat` is a corrected one.

## 1. Frames and conventions

| Assumption | Where | Status |
|---|---|---|
| The state is the world-to-vehicle transform, `C_vi` or `T_vi`, not the vehicle pose in the world. A point maps as `p_v = T_vi p_i`. | L2, L47, L97 | Stated |
| Perturbations are on the left: `C = exp(dphi^) Cbar` and `T = exp(dxi^) Tbar`. The error is therefore expressed in the current vehicle frame. | L2, L47, L97 | Stated |
| The se(3) ordering is translation first, `xi = [rho; phi]`. | L3 | Stated |
| The rotation over one step is `C_{v_k v_{k-1}} = exp(-dt omega^)` for a body-frame rate `omega`. Positive `omega_z` is a left yaw. | L48, L62 | Stated |
| In the SO(3) example, world +x is north and world +z is up, so world +y is west. | L55 | Implicit |
| The world frame is inertial and does not move. Earth rotation is ignored. | L55, L116 | Implicit |
| The sensors sit at the vehicle frame origin and are aligned with it. There are no extrinsics and no calibration error. | L50, L99 | Implicit |
| Units are metres and radians, and measurements are time-synchronized with the state. | L63, L110 | Implicit |

## 2. Assumptions shared by both filters

### Probabilistic model

| Assumption | Where | Status |
|---|---|---|
| Uncertainty is a zero-mean Gaussian on the Lie algebra, wrapped onto the group around the estimate. It is not a Gaussian on matrix entries or Euler angles. | L47, L97 | Stated |
| Process noise enters on the left, after the motion: `exp(w^) * Psi * C` and `exp(w^) * Xi * T`. | L48, L98 | Stated |
| Measurement noise is additive in the measurement space. | L50, L99 | Stated |
| Initial error, process noise and measurement noise are mutually independent and white. The code carries no cross-covariance terms. | L41, L71, L123 | Implicit |
| The covariances `P`, `Q` and `R` are known exactly and are constant. No adaptive tuning is done. | L58-60, L106-108 | Implicit |
| Every measurement is valid. There is no outlier rejection or innovation gating. | L82, L130 | Implicit |

### Approximations in the EKF equations

| Step | Exact form | What the code uses | Comment |
|---|---|---|---|
| Propagation Jacobian | `Psi exp(d^) = exp((Ad(Psi) d)^) Psi` | `F = Ad(Psi)` or `Ad(Xi)` | This is exact and does not depend on the estimate, so a large 90 degree step does not degrade `F`. |
| Combining noise and error | `exp(w^) exp((F d)^)` | `exp((F d + w)^)` | First-order BCH. It assumes both terms are small. |
| Measurement model | `exp(d^) = I + d^ + ...` | `I + d^` | First-order truncation about the prediction, done once. There is no iterated update. |
| Posterior shape | Non-Gaussian in general | Gaussian with covariance `(I - K G) Pcheck` | Standard EKF assumption. The short form also requires `K` to be the exact optimal gain, and the code does not use the Joseph form. |
| Covariance reset | `P' = J(d) Phat J(d)^T`, where `J` is the left Jacobian of the applied update | `P' = Phat`, so `J` is taken as identity | Valid when the update is small. See below. |
| Group update | `exp(dphi^) * Ccheck` | Same | Exact, and it keeps the estimate on the group. No re-orthonormalization is done, which is fine for a single step. |

In the current run the rotational updates are 0.082 rad for SO(3) and 0.076 rad for SE(3), so the rotational part of the neglected reset Jacobian differs from identity by roughly 4%.

## 3. SO(3) attitude example

The filter runs one predict and one correct step (L52-93).

### Motion model

| Assumption | Where | Status |
|---|---|---|
| The body rate is constant over the step (zero-order hold), even with `dt = 1 s` and a 90 degree rotation. | L63-69 | Implicit |
| The gyro has no bias, scale-factor or misalignment state. Its error is only the white term `w`. | L48, L59 | Implicit |
| `Q = 0.005 I` is a per-step variance (about 4.05 degrees per axis). It does not scale with `dt`, so changing `dt` requires retuning it. | L59 | Implicit |
| `Q` is isotropic and independent of the rotation being applied. | L59, L71 | Implicit |

If the noise were modelled inside the gyro integral, `exp(-dt (omega + n)^)`, it would pass through the left Jacobian of the step before reaching the left-perturbation frame. At 90 degrees that Jacobian scales noise perpendicular to the rotation axis by about 0.9, so a constant isotropic `Q` is an approximation.

### Measurement model

| Assumption | Where | Status |
|---|---|---|
| The reference directions `u_i = [0,0,1]` and `m_i = [1,0,0]` are known perfectly and are constant. | L55 | Stated |
| The two directions are not parallel, which makes `G` rank 3 and attitude fully observable. Here they are orthogonal. | L55, L78 | Implicit |
| Each measurement is a body-frame vector with independent additive noise, `R = 0.0025 I` (sigma 0.05 per component). | L60 | Implicit |
| Measurements are treated as unconstrained 3-vectors. The code does not enforce or use the unit norm, so noise along each direction is modelled even though it carries no attitude information. | L50, L60 | Implicit |

If these vectors come from an accelerometer and a magnetometer, which the code does not say, two further assumptions apply. The accelerometer must see only gravity, with no linear acceleration. The magnetometer must point at `m_i` exactly, which is horizontal in this example and has no dip, declination or local disturbance.

### Scenario

- The true initial attitude equals the initial estimate (L57, L66), so the initial error is zero even though `P` claims 0.1 rad per axis.
- The gyro error is a fixed offset of `[0.05, -0.03, -0.08]` rad/s (L64-65). It behaves like a bias, while the filter treats it as white noise. One step cannot show the difference, but many steps would build up correlated error that this model does not capture.
- The measurement is noise-free by design (L73), while the filter still assumes `R`. The 0.786 degree error after correction is therefore likely better than real noisy data would give.

## 4. SE(3) pose example

The filter runs one predict and one correct step (L100-143).

### Motion model

| Assumption | Where | Status |
|---|---|---|
| The odometry is one known rigid transform: translate 2 m along +x, then yaw 90 degrees. | L110-114 | Stated |
| Odometry noise is a fixed diagonal `Q` in the vehicle frame at the end of the step. It is not derived from a wheel or sensor model, and it does not grow with distance or turn angle. | L107 | Implicit |
| `Q` has no cross-correlation between translation and rotation, or between axes. | L107 | Implicit |
| The `P` and `Q` diagonals are variances, not standard deviations. For example, `P(0,0) = 0.10` means sigma of about 0.316 m. | L106-107 | Implicit |

### Planar motion

| Assumption | Where | Status |
|---|---|---|
| Motion stays in the plane. | L96 | Stated |
| Planarity is a soft constraint only. `z`, roll and pitch stay in the 6-DoF state with small variances. Nothing forces them to zero. | L106-107 | Implicit |
| The landmark lies in the plane and its measured `z` is 0 with a 0.2 m sigma, which effectively acts as a planar pseudo-measurement. | L116-117 | Implicit |

### Measurement model

| Assumption | Where | Status |
|---|---|---|
| The sensor returns the landmark's 3D Cartesian position in the vehicle frame, `y = D^T T p + n`. It is not a pixel, or a range and bearing. | L99, L117 | Stated |
| The landmark position `p = (4, 5, 0, 1)` is known exactly. It has no map uncertainty and is finite (`eta = 1`). | L116 | Stated |
| Data association is correct. | L116-117 | Implicit |
| `R = 0.04 I` (sigma 0.2 m) is isotropic and independent of range. | L108 | Implicit |

A single point gives `G` rank 3 against 6 states, so one update cannot observe the full pose. The remaining directions, such as rotation about the line of sight, stay constrained only by the prior.

### Scenario

- The initial pose is exactly the identity (L105).
- The measurement `(5.3, -1.6, 0)` is hard-coded (L117) and the example has no ground truth. The output can be compared with the SE(2) note mentioned at L96, but pose error cannot be computed.

## 5. Reading the printed output

- `heading` (L134) reads yaw from `C_iv` and is only meaningful when roll and pitch are near zero.
- The printed sigmas are marginal values for the perturbation, expressed in the current vehicle frame. They are not world-axis uncertainties.
- To first order, the position error in vehicle coordinates is `-rho`. After the 90 degree turn, `rho1` is mostly a world-y error. In the current run, the first component of the `rho` update is 0.1417, and it moves the robot to `y = -0.142`.
- `|C^T C - I|` confirms the update stayed on the group. It says nothing about whether the estimate is accurate.

## 6. What the examples do not test

- Each filter runs a single step, so long-term consistency, covariance symmetry and numerical drift are never exercised.
- `identityChecks()` verifies the `odot` and Adjoint algebra only. It uses homogeneous points with `eta = 1`, so other values of `eta` are untested, and it does not test any filter behaviour.
- No Monte Carlo or NEES test checks whether `P` matches the actual error.

The assumptions most likely to break first on real data are the missing gyro bias state, the fixed `Q`, and the single known landmark with correct association.
