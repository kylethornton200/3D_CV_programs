// ===== Your Lie group functions: math unchanged. Made inline for header use; removed the unused cot(), the commented-out leftJacobian, and the Sophus includes. =====
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

inline Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

inline Eigen::Matrix3d expSO3(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity() + hat(phi);
    }
    const Eigen::Vector3d a = phi / theta;
    return std::cos(theta) * Eigen::Matrix3d::Identity()
         + (1.0 - std::cos(theta)) * a * a.transpose()
         + std::sin(theta) * hat(a);
}

inline Eigen::Vector3d logSO3(const Eigen::Matrix3d& C) {
    double w;
    Eigen::Vector3d V;
    const double t = C(0,0) + C(1,1) + C(2,2);
    if (t > 0.0) {
        w = 0.5 * std::sqrt(1.0 + t);
        V << C(2,1) - C(1,2),
             C(0,2) - C(2,0),
             C(1,0) - C(0,1);
        V /= (4.0 * w);
    } else {
        Eigen::Index i;
        C.diagonal().maxCoeff(&i);
        const Eigen::Index j = (i + 1) % 3;
        const Eigen::Index k = (i + 2) % 3;
        double s = std::sqrt(C(i,i) - C(j,j) - C(k,k) + 1.0);
        V(i) = 0.5 * s;
        s = 0.5 / s;
        V(j) = (C(j,i) + C(i,j)) * s;
        V(k) = (C(k,i) + C(i,k)) * s;
        w    = (C(k,j) - C(j,k)) * s;
    }
    if (w < 0.0) { w = -w; V = -V; }
    const double n = V.norm();
    double factor;
    if (n < 1e-8) factor = 2.0 / w - 2.0 * n * n / (3.0 * w * w * w);
    else          factor = 2.0 * std::atan2(n, w) / n;
    return factor * V;
}

inline Eigen::Matrix3d leftJacobian(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    if (theta < 1e-8) return Eigen::Matrix3d::Identity() + 0.5 * hat(phi);
    const Eigen::Vector3d a = phi / theta;
    const double s  = std::sin(theta) / theta;
    const double sh = std::sin(0.5 * theta);
    const double c  = 2.0 * sh * sh / theta;
    return s * Eigen::Matrix3d::Identity() + (1.0 - s) * a * a.transpose() + c * hat(a);
}

inline Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& phi) { return leftJacobian(-phi); }

inline Eigen::Matrix3d leftJacobianInverse(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    if (theta < 1e-8) return Eigen::Matrix3d::Identity() - 0.5 * hat(phi);
    const Eigen::Vector3d a = phi / theta;
    const double half = 0.5 * theta;
    const double c = half / std::tan(half);
    return c * Eigen::Matrix3d::Identity() + (1.0 - c) * a * a.transpose() - half * hat(a);
}

inline Eigen::Matrix3d rightJacobianInverse(const Eigen::Vector3d& phi) { return leftJacobianInverse(-phi); }

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

inline Eigen::Matrix4d expSE3(const Vector6d& xi) {
    const Eigen::Vector3d rho = xi.head<3>();
    const Eigen::Vector3d phi = xi.tail<3>();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>()  = expSO3(phi);
    T.topRightCorner<3, 1>() = leftJacobian(phi) * rho;
    return T;
}

inline Vector6d logSE3(const Eigen::Matrix4d& T){
    const Eigen::Matrix3d C = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d r = T.topRightCorner<3, 1>();
    Eigen::Vector3d phi = logSO3(C);
    Eigen::Vector3d rho = leftJacobianInverse(phi) * r;
    Vector6d xi; xi << rho, phi;
    return xi;
}

inline Matrix6d adjointSE3(const Eigen::Matrix4d& T){
    const Eigen::Matrix3d C = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d r = T.topRightCorner<3, 1>();
    Matrix6d adjT;
    adjT << C, hat(r)*C, Eigen::Matrix3d::Zero(), C;
    return adjT;
}

inline Eigen::Vector3d randomPhi(std::mt19937& rng, int i) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> angleDist(0.0, M_PI);
    std::uniform_real_distribution<double> tinyExp(-12.0, -6.0);
    Eigen::Vector3d axis(gauss(rng), gauss(rng), gauss(rng));
    axis.normalize();
    const double theta = (i % 5 == 0) ? std::pow(10.0, tinyExp(rng)) : angleDist(rng);
    return theta * axis;
}
