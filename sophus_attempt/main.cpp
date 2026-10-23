#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity() + hat(phi);
    }
    const Eigen::Vector3d a = phi / theta;
    return std::cos(theta) * Eigen::Matrix3d::Identity()
         + (1.0 - std::cos(theta)) * a * a.transpose()
         + std::sin(theta) * hat(a);
}

Eigen::Vector3d logSO3(const Eigen::Matrix3d& C) {
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

    // q and -q are the same rotation; pick w >= 0 so theta is in [0, pi].
    if (w < 0.0) {
        w = -w;
        V = -V;
    }

    const double n = V.norm();
    double factor;
    if (n < 1e-8) {
        factor = 2.0 / w - 2.0 * n * n / (3.0 * w * w * w);
    } else {
        factor = 2.0 * std::atan2(n, w) / n;
    }
    return factor * V;
}

Eigen::Matrix3d leftJacobian(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity() + 0.5 * hat(phi);
    }
    const Eigen::Vector3d a = phi / theta;
    const double s = std::sin(theta) / theta;
    return s * Eigen::Matrix3d::Identity()
         + (1.0 - s) * a * a.transpose()
         + ((1.0 - std::cos(theta)) / theta) * hat(a);
}

Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& phi) {
    return leftJacobian(-phi);        // equivalently leftJacobian(phi).transpose()
}

static Eigen::Vector3d randomPhi(std::mt19937& rng, int i) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> angleDist(0.0, M_PI);
    std::uniform_real_distribution<double> tinyExp(-12.0, -6.0);

    Eigen::Vector3d axis(gauss(rng), gauss(rng), gauss(rng));
    axis.normalize();
    const double theta = (i % 5 == 0) ? std::pow(10.0, tinyExp(rng))
                                      : angleDist(rng);
    return theta * axis;
}

// ---------------- exp / log ----------------
static bool testExpLog(std::mt19937& rng) {
    const int N = 10000;
    const double tol = 1e-12;
    const double piMargin = 1e-3;   // vector comparisons only valid away from theta = pi

    double worstSophus = 0.0, worstOrtho = 0.0, worstDet = 0.0;
    double worstRoundTrip = 0.0, worstLogSophus = 0.0, worstRecon = 0.0;

    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d phi = randomPhi(rng, i);
        const double theta = phi.norm();

        // exp
        const Eigen::Matrix3d R = expSO3(phi);
        const Eigen::Matrix3d R_sophus = Sophus::SO3d::exp(phi).matrix();
        worstSophus = std::max(worstSophus, (R - R_sophus).cwiseAbs().maxCoeff());
        worstOrtho  = std::max(worstOrtho,
            (R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff());
        worstDet    = std::max(worstDet, std::abs(R.determinant() - 1.0));

        // log
        const Eigen::Vector3d phi_log = logSO3(R);
        worstRecon = std::max(worstRecon, (expSO3(phi_log) - R).cwiseAbs().maxCoeff());

        if (theta < M_PI - piMargin) {
            worstRoundTrip = std::max(worstRoundTrip, (phi_log - phi).cwiseAbs().maxCoeff());
            const Eigen::Vector3d phi_sophus = Sophus::SO3d(R).log();
            worstLogSophus = std::max(worstLogSophus,
                (phi_log - phi_sophus).cwiseAbs().maxCoeff());
        }
    }

    std::cout << "exp: worst |R - R_sophus|:       " << worstSophus    << "\n"
              << "exp: worst |R^T R - I|:          " << worstOrtho     << "\n"
              << "exp: worst |det(R) - 1|:         " << worstDet       << "\n"
              << "log: worst round trip |phi|:     " << worstRoundTrip << "\n"
              << "log: worst |log - log_sophus|:   " << worstLogSophus << "\n"
              << "log: worst |exp(log(R)) - R|:    " << worstRecon     << "\n";

    return worstSophus < tol && worstOrtho < tol && worstDet < tol
        && worstRoundTrip < tol && worstLogSophus < tol && worstRecon < tol;
}

// ---------------- log edge cases ----------------
static bool testLogEdgeCases() {
    const double tol = 1e-12;
    double worstEdge = 0.0;
    const double edgeAngles[] = {0.0, 1e-10, M_PI - 1e-6, M_PI};
    for (int ax = 0; ax < 3; ++ax) {
        for (double th : edgeAngles) {
            Eigen::Vector3d phi = Eigen::Vector3d::Zero();
            phi(ax) = th;
            const Eigen::Matrix3d R = expSO3(phi);
            const Eigen::Matrix3d Rrec = expSO3(logSO3(R));
            worstEdge = std::max(worstEdge, (R - Rrec).cwiseAbs().maxCoeff());
        }
    }
    std::cout << "log: worst edge-case recon:      " << worstEdge << "\n";
    return worstEdge < tol;
}

// ---------------- Jacobians ----------------
static bool testJacobians(std::mt19937& rng) {
    const int N = 2000;
    const double eps = 1e-6;       // finite-difference step
    const double tolFD = 1e-8;     // FD error ~ rounding/eps ~ 1e-10
    const double tolExact = 1e-12;

    double worstLeft = 0, worstRight = 0, worstTranspose = 0,
           worstNeg = 0, worstFixed = 0;

    for (int n = 0; n < N; ++n) {
        const Eigen::Vector3d phi = randomPhi(rng, n);
        const Eigen::Matrix3d R = expSO3(phi);

        Eigen::Matrix3d Jl_num, Jr_num;
        for (int k = 0; k < 3; ++k) {
            Eigen::Vector3d d = Eigen::Vector3d::Zero();
            d(k) = eps;
            // exp(phi + d) ~ exp(Jl d) exp(phi)
            Jl_num.col(k) = (logSO3(expSO3(phi + d) * R.transpose())
                           - logSO3(expSO3(phi - d) * R.transpose())) / (2.0 * eps);
            // exp(phi + d) ~ exp(phi) exp(Jr d)
            Jr_num.col(k) = (logSO3(R.transpose() * expSO3(phi + d))
                           - logSO3(R.transpose() * expSO3(phi - d))) / (2.0 * eps);
        }

        const Eigen::Matrix3d Jl = leftJacobian(phi);
        const Eigen::Matrix3d Jr = rightJacobian(phi);

        worstLeft      = std::max(worstLeft,      (Jl - Jl_num).cwiseAbs().maxCoeff());
        worstRight     = std::max(worstRight,     (Jr - Jr_num).cwiseAbs().maxCoeff());
        worstTranspose = std::max(worstTranspose, (Jr - Jl.transpose()).cwiseAbs().maxCoeff());
        worstNeg       = std::max(worstNeg,       (Jr - leftJacobian(-phi)).cwiseAbs().maxCoeff());
        worstFixed     = std::max(worstFixed,     (Jl * phi - phi).cwiseAbs().maxCoeff());
    }

    std::cout << "J_l vs finite diff:              " << worstLeft      << "\n"
              << "J_r vs finite diff:              " << worstRight     << "\n"
              << "|J_r - J_l^T|:                   " << worstTranspose << "\n"
              << "|J_r - J_l(-phi)|:               " << worstNeg       << "\n"
              << "|J_l phi - phi|:                 " << worstFixed     << "\n";

    return worstLeft < tolFD && worstRight < tolFD
        && worstTranspose < tolExact && worstNeg < tolExact && worstFixed < tolExact;
}

int main() {
    std::mt19937 rng(42);

    // Run every group (no short-circuiting) so one failure doesn't hide another.
    bool ok = true;
    ok &= testExpLog(rng);
    ok &= testLogEdgeCases();
    ok &= testJacobians(rng);

    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}