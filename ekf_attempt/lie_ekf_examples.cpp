// Lie group EKF/IEKF examples (SO(3) attitude, SE(3) pose) built on user_lie.hpp.
// Barfoot conventions: state is C_vi / T_vi (world -> vehicle), LEFT perturbations,
// xi = [rho; phi]. Build:
//   g++ -std=c++17 -O2 -I/usr/include/eigen3 lie_ekf_examples.cpp -o lie_ekf_examples
#include "user_lie.hpp"
#include <cstdio>

// ======================= ADDED: pieces an EKF needs =======================

// se(3) hat: 6-vector -> 4x4
Eigen::Matrix4d hatSE3(const Vector6d& xi) {
    Eigen::Matrix4d X = Eigen::Matrix4d::Zero();
    X.topLeftCorner<3, 3>()  = hat(xi.tail<3>());
    X.topRightCorner<3, 1>() = xi.head<3>();
    return X;
}

// Odot: for homogeneous p = [eps; eta],  hatSE3(xi) * p == odot(p) * xi.   (4x6)
Eigen::Matrix<double, 4, 6> odot(const Eigen::Vector4d& p) {
    Eigen::Matrix<double, 4, 6> P = Eigen::Matrix<double, 4, 6>::Zero();
    P.topLeftCorner<3, 3>()  = p(3) * Eigen::Matrix3d::Identity();
    P.topRightCorner<3, 3>() = -hat(p.head<3>());
    return P;
}

// Cheap SE(3) inverse: [C^T, -C^T r]
Eigen::Matrix4d invSE3(const Eigen::Matrix4d& T) {
    Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
    Ti.topLeftCorner<3, 3>()  = T.topLeftCorner<3, 3>().transpose();
    Ti.topRightCorner<3, 1>() = -Ti.topLeftCorner<3, 3>() * T.topRightCorner<3, 1>();
    return Ti;
}

Eigen::Matrix3d rotZ(double th) { return expSO3(Eigen::Vector3d(0, 0, th)); }

// ---- generic EKF correction on the perturbation (same as the vector EKF) ----
template <int N, int M>
void ekfGain(const Eigen::Matrix<double, N, N>& Pc, const Eigen::Matrix<double, M, N>& G,
             const Eigen::Matrix<double, M, M>& R, Eigen::Matrix<double, N, M>& K,
             Eigen::Matrix<double, N, N>& Ph) {
    const Eigen::Matrix<double, M, M> S = G * Pc * G.transpose() + R;
    K  = Pc * G.transpose() * S.inverse();
    Ph = (Eigen::Matrix<double, N, N>::Identity() - K * G) * Pc;
}

// ======================= Example 1: SO(3) attitude EKF =======================
// State: C_k = C_{v_k i}.  Perturbation: C = exp(dphi^) * Cbar,  dphi ~ N(0, P)  (3x3)
// Motion: C_k = exp(w^) * Psi_k * C_{k-1},  Psi_k = exp(-dt * omega^)  (gyro)
//         -> F = Ad(Psi_k) = Psi_k   (on SO(3) the Adjoint is the rotation itself)
// Meas:   y = [C u_i; C m_i] + n   (known world 'up' and 'north' seen in the body)
//         C u ~= Cc u - (Cc u)^ dphi  ->  G = [-(Cc u)^; -(Cc m)^]   (6x3)
void so3Example() {
    std::puts("\n================ SO(3) attitude EKF ================");
    using M3 = Eigen::Matrix3d;
    const Eigen::Vector3d u_i(0, 0, 1), m_i(1, 0, 0);

    M3 C_hat = M3::Identity();
    M3 P = 0.01 * M3::Identity();                       // sigma 0.1 rad per axis
    const M3 Q = 0.005 * M3::Identity();
    const Eigen::Matrix<double, 6, 6> R = 0.0025 * Eigen::Matrix<double, 6, 6>::Identity();

    // gyro says: yaw left 90 deg in 1 s. Truth differs a bit (gyro error).
    const double dt = 1.0;
    const Eigen::Vector3d omega_meas(0, 0, M_PI / 2);
    const Eigen::Vector3d omega_true(0.05, -0.03, M_PI / 2 - 0.08);
    const M3 C_true = expSO3(-dt * omega_true) * M3::Identity();

    // ---- predict ----
    const M3 Psi = expSO3(-dt * omega_meas);
    M3 C_check = Psi * C_hat;                            // compose
    M3 P_check = Psi * P * Psi.transpose() + Q;          // F = Ad(Psi) = Psi

    // ---- correct ---- (noise-free measurement from the truth, to see convergence)
    Eigen::Matrix<double, 6, 1> y, y_check;
    y << C_true * u_i, C_true * m_i;
    y_check << C_check * u_i, C_check * m_i;
    Eigen::Matrix<double, 6, 3> G;
    G << -hat(C_check * u_i), -hat(C_check * m_i);

    Eigen::Matrix<double, 3, 6> K; M3 P_hat;
    ekfGain<3, 6>(P_check, G, R, K, P_hat);
    const Eigen::Vector3d dphi = K * (y - y_check);
    C_hat = expSO3(dphi) * C_check;                      // update ON the group

    auto errDeg = [&](const M3& C) { return logSO3(C_true * C.transpose()).norm() * 180 / M_PI; };
    std::printf("attitude error  after predict: %.3f deg   after correct: %.3f deg\n",
                errDeg(C_check), errDeg(C_hat));
    std::printf("dphi = [%.4f %.4f %.4f]\n", dphi(0), dphi(1), dphi(2));
    std::printf("sigma (deg) before: [%.2f %.2f %.2f]  after: [%.2f %.2f %.2f]\n",
                std::sqrt(P_check(0,0))*180/M_PI, std::sqrt(P_check(1,1))*180/M_PI, std::sqrt(P_check(2,2))*180/M_PI,
                std::sqrt(P_hat(0,0))*180/M_PI, std::sqrt(P_hat(1,1))*180/M_PI, std::sqrt(P_hat(2,2))*180/M_PI);
    std::printf("|C^T C - I| = %.1e\n", (C_hat.transpose() * C_hat - M3::Identity()).norm());
}

// ======================= Example 2: SE(3) pose EKF =======================
// Same problem as the SE(2) note, lifted to 3D (motion stays in the plane).
// State T_k = T_{v_k i}, T = exp(dxi^) Tbar, dxi = [rho; phi] ~ N(0, P)  (6x6)
// Motion: T_k = exp(w^) Xi_k T_{k-1}     -> F = Ad(Xi_k)
// Meas:   y = D^T T_k p + n              -> G = D^T (Tc p)^odot   (3x6)
void se3Example() {
    std::puts("\n================ SE(3) pose EKF (planar case) ================");
    using M4 = Eigen::Matrix4d;
    Vector6d d;

    M4 T_hat = M4::Identity();
    d << 0.10, 0.10, 0.01, 0.01, 0.01, 0.05;  Matrix6d P = d.asDiagonal();
    d << 0.05, 0.05, 0.01, 0.005, 0.005, 0.02; const Matrix6d Q = d.asDiagonal();
    const Eigen::Matrix3d R = 0.04 * Eigen::Matrix3d::Identity();

    // odometry: drive 2 m forward then turn left 90 deg -> Xi = T_{v1 v0}
    M4 T_iv1 = M4::Identity();
    T_iv1.topLeftCorner<3, 3>() = rotZ(M_PI / 2);
    T_iv1.topRightCorner<3, 1>() << 2, 0, 0;
    const M4 Xi = invSE3(T_iv1);

    const Eigen::Vector4d p(4, 5, 0, 1);                 // known landmark
    const Eigen::Vector3d y(5.3, -1.6, 0.0);             // measured, vehicle frame
    Eigen::Matrix<double, 3, 4> DT; DT << Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero();

    // ---- predict ----
    const M4 T_check = Xi * T_hat;
    const Matrix6d F = adjointSE3(Xi);
    const Matrix6d P_check = F * P * F.transpose() + Q;

    // ---- correct (IEKF) ----
    // Work in the chart centred on the prior: T = exp(xi^) T_check, xi ~ N(0, P_check).
    // This is the vector IEKF (4.45) with x_check = 0 and x_op = xi_op.
    Vector6d xi_op = Vector6d::Zero();                   // first pass == EKF
    Vector6d dxi;
    Eigen::Matrix<double, 3, 6> G;
    Eigen::Matrix<double, 6, 3> K;
    Matrix6d P_hat;
    const int max_iters = 20;
    for (int it = 0; it < max_iters; ++it) {
        const M4 T_op = expSE3(xi_op) * T_check;          // current operating point
        const Eigen::Vector4d q = T_op * p;              // relinearise at T_op, not T_check
        const Eigen::Vector3d y_op = DT * q;
        G = DT * odot(q) * leftJacobianSE3(xi_op);       // d y / d xi at xi_op
        ekfGain<6, 3>(P_check, G, R, K, P_hat);          // P_check stays frozen
        dxi = K * (y - y_op + G * xi_op);                // (4.45c): -G (x_check - x_op) = +G xi_op
        const double step = (dxi - xi_op).norm();
        xi_op = dxi;
        if (step < 1e-10) break;
    }
    T_hat = expSE3(dxi) * T_check;                       // anchored at the prior, not at T_op
    // P_hat is from the final linearisation. It describes xi about T_check; to express it as a
    // left perturbation about T_hat, use J P J^T with J = leftJacobianSE3(dxi) (small change here).
    const Matrix6d Jh = leftJacobianSE3(dxi);
    P_hat = Jh * P_hat * Jh.transpose();

    const M4 T_iv = invSE3(T_hat);
    const double heading = std::atan2(T_iv(1, 0), T_iv(0, 0)) * 180 / M_PI;
    std::printf("dxi = [rho %.4f %.4f %.4f | phi %.4f %.4f %.4f]\n",
                dxi(0), dxi(1), dxi(2), dxi(3), dxi(4), dxi(5));
    std::printf("robot in world for IEKF: x=%.3f  y=%.3f  z=%.3f  heading=%.2f deg\n",
                T_iv(0, 3), T_iv(1, 3), T_iv(2, 3), heading);
    std::printf("planar sigmas after: rho1=%.3f rho2=%.3f phi3=%.3f\n",
                std::sqrt(P_hat(0,0)), std::sqrt(P_hat(1,1)), std::sqrt(P_hat(5,5)));
    const Eigen::Matrix3d C = T_hat.topLeftCorner<3, 3>();
    std::printf("|C^T C - I| = %.1e\n", (C.transpose() * C - Eigen::Matrix3d::Identity()).norm());
}

// ======================= quick identity checks for the added pieces =======================
void identityChecks() {
    std::mt19937 rng(7);
    std::normal_distribution<double> g(0, 1);
    double eOdot = 0, eAdj = 0;
    for (int i = 0; i < 1000; ++i) {
        Vector6d xi, x2; for (int k = 0; k < 6; ++k) { xi(k) = g(rng); x2(k) = 0.3 * g(rng); }
        Eigen::Vector4d p(g(rng), g(rng), g(rng), 1.0);
        eOdot = std::max(eOdot, (hatSE3(xi) * p - odot(p) * xi).norm());
        const Eigen::Matrix4d T = expSE3(xi);
        // Adjoint identity: T exp(x2^) = exp((Ad(T) x2)^) T
        eAdj = std::max(eAdj, (T * expSE3(x2) - expSE3(adjointSE3(T) * x2) * T).norm());
    }
    std::printf("identity checks: odot max err = %.1e, adjoint max err = %.1e\n", eOdot, eAdj);
}

int main() {
    identityChecks();
    so3Example();
    se3Example();
}