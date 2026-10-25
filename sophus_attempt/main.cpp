#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
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
    const double s  = std::sin(theta) / theta;
    const double sh = std::sin(0.5 * theta);
    const double c  = 2.0 * sh * sh / theta;      // attempt to fix problems arising in the inverse tests.
    return s * Eigen::Matrix3d::Identity()
         + (1.0 - s) * a * a.transpose()
         + c * hat(a);
}

// Eigen::Matrix3d leftJacobian(const Eigen::Vector3d& phi) {
//     const double theta = phi.norm();
//     if (theta < 1e-8) {
//         return Eigen::Matrix3d::Identity() + 0.5 * hat(phi);
//     }
//     const Eigen::Vector3d a = phi / theta;
//     const double s = std::sin(theta) / theta;
//     return s * Eigen::Matrix3d::Identity()
//          + (1.0 - s) * a * a.transpose()
//          + ((1.0 - std::cos(theta)) / theta) * hat(a);
// }

Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& phi) {
    return leftJacobian(-phi);        // equivalently leftJacobian(phi).transpose()
}

double cot(double x) {
    return 1.0 / std::tan(x);
}

Eigen::Matrix3d leftJacobianInverse(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity() - 0.5 * hat(phi);
    }
    const Eigen::Vector3d a = phi / theta;
    const double half = 0.5 * theta;
    const double c = half / std::tan(half);
    return c * Eigen::Matrix3d::Identity()
         + (1.0 - c) * a * a.transpose()
         - half * hat(a);
}

Eigen::Matrix3d rightJacobianInverse(const Eigen::Vector3d& phi) {
    return leftJacobianInverse(-phi);
}

using Vector6d = Eigen::Matrix<double, 6, 1>;

// xi = [rho; phi]: translation part first, rotation vector second.
Eigen::Matrix4d expSE3(const Vector6d& xi) {
    const Eigen::Vector3d rho = xi.head<3>();
    const Eigen::Vector3d phi = xi.tail<3>();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>()  = expSO3(phi);
    T.topRightCorner<3, 1>() = leftJacobian(phi) * rho;
    return T;
}

Vector6d logSE3(const Eigen::Matrix4d& T){
    const Eigen::Matrix3d C = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d r = T.topRightCorner<3, 1>();

    Eigen::Vector3d phi = logSO3(C);
    Eigen::Vector3d rho = leftJacobianInverse(phi) * r;

    Vector6d xi;
    xi << rho,  phi;
    return xi;
}

using Matrix6d = Eigen::Matrix<double, 6, 6>;

static Matrix6d adjointSE3(const Eigen::Matrix4d& T){
    const Eigen::Matrix3d C = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d r = T.topRightCorner<3, 1>();
    Matrix6d adjT;
    adjT << C, hat(r)*C, Eigen::Matrix3d::Zero(), C;
    return adjT;
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
static bool testExpLogSO3(std::mt19937& rng) {
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
static bool testLogEdgeCasesSO3() {
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
    const double tolFD = 1e-9;     // FD error ~ rounding/eps ~ 1e-10
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

static bool testJacobianInverses(std::mt19937& rng) {
    const int N = 5000;
    const double tol = 1e-12;
    double worstLeftInv = 0, worstRightInv = 0, worstFixed = 0, worstTranspose = 0;

    for (int n = 0; n < N; ++n) {
        const Eigen::Vector3d phi = randomPhi(rng, n);
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        const Eigen::Matrix3d Jl = leftJacobian(phi);
        const Eigen::Matrix3d Jli = leftJacobianInverse(phi);
        const Eigen::Matrix3d Jr = rightJacobian(phi);
        const Eigen::Matrix3d Jri = rightJacobianInverse(phi);

        worstLeftInv  = std::max(worstLeftInv,  (Jl * Jli - I).cwiseAbs().maxCoeff());
        worstRightInv = std::max(worstRightInv, (Jr * Jri - I).cwiseAbs().maxCoeff());
        worstFixed    = std::max(worstFixed,    (Jli * phi - phi).cwiseAbs().maxCoeff());
        worstTranspose = std::max(worstTranspose, (Jri - Jli.transpose()).cwiseAbs().maxCoeff());
    }

    std::cout << "|J_l J_l^-1 - I|:         " << worstLeftInv   << "\n"
              << "|J_r J_r^-1 - I|:         " << worstRightInv  << "\n"
              << "|J_l^-1 phi - phi|:       " << worstFixed     << "\n"
              << "|J_r^-1 - (J_l^-1)^T|:    " << worstTranspose << "\n";

    return worstLeftInv < tol && worstRightInv < tol && worstFixed < tol && worstTranspose < tol;
}

// ---------------- SE(3) helpers ----------------
// xi = [rho; phi]. Random translation (Gaussian, scaled) + random rotation vector.
static Vector6d randomXi(std::mt19937& rng, int i, double rhoScale = 5.0) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    Vector6d xi;
    xi.head<3>() = rhoScale * Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng));
    xi.tail<3>() = randomPhi(rng, i);
    return xi;
}

// Reference for the translation part of exp(xi): t = V(phi) * rho, with
//   V = I + c1 [phi]x + c2 [phi]x^2.
// For small angles the Taylor series of c1, c2 is accurate (no cancellation);
// otherwise fall back to Sophus as an independent implementation.
// (Sophus's own SE3::exp loses precision for theta ~ 1e-10..1e-5 because of
// cancellation in 1 - cos(theta), so it is NOT a valid reference there.)
static Eigen::Vector3d referenceTranslation(const Eigen::Vector3d& rho,
                                            const Eigen::Vector3d& phi,
                                            const Vector6d& xi) {
    const double th = phi.norm();
    if (th < 1e-2) {
        const double t2 = th * th;
        const double c1 = 0.5 - t2 / 24.0 + t2 * t2 / 720.0;
        const double c2 = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0;
        const Eigen::Vector3d p1 = phi.cross(rho);
        return rho + c1 * p1 + c2 * phi.cross(p1);
    }
    return Sophus::SE3d::exp(xi).translation();
}

// ---------------- SE(3) exp / log ----------------
static bool testExpLogSE3(std::mt19937& rng) {
    const int N = 10000;
    const double tol = 1e-11;       // looser than SO(3): translations are O(10)
    const double piMargin = 1e-3;   // vector comparisons only valid away from theta = pi

    double worstRot = 0.0, worstTrans = 0.0, worstLastRow = 0.0;
    double worstRoundTrip = 0.0, worstLogSophus = 0.0, worstRecon = 0.0;

    for (int i = 0; i < N; ++i) {
        const Vector6d xi = randomXi(rng, i);
        const double theta = xi.tail<3>().norm();

        // exp
        const Eigen::Matrix4d T = expSE3(xi);
        const Eigen::Matrix3d R_sophus = Sophus::SO3d::exp(xi.tail<3>()).matrix();
        worstRot   = std::max(worstRot,
            (T.topLeftCorner<3, 3>() - R_sophus).cwiseAbs().maxCoeff());
        worstTrans = std::max(worstTrans,
            (T.topRightCorner<3, 1>()
             - referenceTranslation(xi.head<3>(), xi.tail<3>(), xi)).cwiseAbs().maxCoeff());
        worstLastRow = std::max(worstLastRow,
            (T.bottomRows<1>() - Eigen::RowVector4d(0, 0, 0, 1)).cwiseAbs().maxCoeff());

        // log
        const Vector6d xi_log = logSE3(T);
        worstRecon = std::max(worstRecon, (expSE3(xi_log) - T).cwiseAbs().maxCoeff());

        if (theta < M_PI - piMargin) {
            worstRoundTrip = std::max(worstRoundTrip, (xi_log - xi).cwiseAbs().maxCoeff());
            const Sophus::SE3d S(Sophus::SO3d(Eigen::Matrix3d(T.topLeftCorner<3, 3>())),
                                 Eigen::Vector3d(T.topRightCorner<3, 1>()));
            const Vector6d xi_sophus = S.log();
            worstLogSophus = std::max(worstLogSophus,
                (xi_log - xi_sophus).cwiseAbs().maxCoeff());
        }
    }

    std::cout << "SE3 exp: worst rotation error:     " << worstRot       << "\n"
              << "SE3 exp: worst translation error:  " << worstTrans     << "\n"
              << "SE3 exp: worst bottom-row error:   " << worstLastRow   << "\n"
              << "SE3 log: worst round trip |xi|:    " << worstRoundTrip << "\n"
              << "SE3 log: worst |log - log_sophus|: " << worstLogSophus << "\n"
              << "SE3 log: worst |exp(log(T)) - T|:  " << worstRecon     << "\n";

    return worstRot < tol && worstTrans < tol && worstLastRow == 0.0
        && worstRoundTrip < tol && worstLogSophus < tol && worstRecon < tol;
}

// ---------------- SE(3) log edge cases ----------------
static bool testLogEdgeCasesSE3() {
    const double tol = 1e-11;
    double worstRecon = 0.0;

    const double edgeAngles[] = {0.0, 1e-10, M_PI - 1e-6, M_PI};
    const Eigen::Vector3d rhos[] = {
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(-4.0, 0.5, 7.0),
        Eigen::Vector3d(100.0, -50.0, 25.0)};

    for (int ax = 0; ax < 3; ++ax) {
        for (double th : edgeAngles) {
            for (const Eigen::Vector3d& rho : rhos) {
                Vector6d xi = Vector6d::Zero();
                xi.head<3>() = rho;
                xi(3 + ax) = th;
                const Eigen::Matrix4d T = expSE3(xi);
                const Eigen::Matrix4d Trec = expSE3(logSE3(T));
                // Compare transforms, not xi: at theta = pi the sign of phi is ambiguous.
                worstRecon = std::max(worstRecon, (T - Trec).cwiseAbs().maxCoeff());
            }
        }
    }

    // Pure translation: log must return exactly [t; 0].
    double worstPure = 0.0;
    for (const Eigen::Vector3d& t : rhos) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topRightCorner<3, 1>() = t;
        Vector6d expected;
        expected << t, Eigen::Vector3d::Zero();
        worstPure = std::max(worstPure, (logSE3(T) - expected).cwiseAbs().maxCoeff());
    }

    std::cout << "SE3 log: worst edge-case recon:    " << worstRecon << "\n"
              << "SE3 log: worst pure-translation:   " << worstPure  << "\n";

    return worstRecon < tol && worstPure < tol;
}

// Analytic SE(3) inverse: [C^T, -C^T r; 0, 1].
static Eigen::Matrix4d inverseSE3(const Eigen::Matrix4d& T) {
    Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
    Ti.topLeftCorner<3, 3>()  = T.topLeftCorner<3, 3>().transpose();
    Ti.topRightCorner<3, 1>() = -T.topLeftCorner<3, 3>().transpose() * T.topRightCorner<3, 1>();
    return Ti;
}

static bool testAdjointSE3(std::mt19937& rng) {
    const int N = 5000;
    const double tol = 1e-10;       // chained products with O(10) translations
    const double tolExact = 1e-12;

    double worstConj = 0, worstSophus = 0, worstHomo = 0, worstInv = 0, worstId = 0;

    // Ad(I) = I
    worstId = (adjointSE3(Eigen::Matrix4d::Identity()) - Matrix6d::Identity()).cwiseAbs().maxCoeff();

    for (int n = 0; n < N; ++n) {
        const Eigen::Matrix4d T  = expSE3(randomXi(rng, n));
        const Eigen::Matrix4d T2 = expSE3(randomXi(rng, n + 1));
        const Vector6d xi = randomXi(rng, n + 2, 1.0);

        const Matrix6d Ad = adjointSE3(T);

        // T exp(xi) T^-1 = exp(Ad(T) xi)
        const Eigen::Matrix4d lhs = T * expSE3(xi) * inverseSE3(T);
        const Eigen::Matrix4d rhs = expSE3(Ad * xi);
        worstConj = std::max(worstConj, (lhs - rhs).cwiseAbs().maxCoeff());

        // Against Sophus's adjoint (same [translation; rotation] ordering).
        const Sophus::SE3d S(Sophus::SO3d(Eigen::Matrix3d(T.topLeftCorner<3, 3>())),
                             Eigen::Vector3d(T.topRightCorner<3, 1>()));
        worstSophus = std::max(worstSophus, (Ad - S.Adj()).cwiseAbs().maxCoeff());

        // Homomorphism: Ad(T T2) = Ad(T) Ad(T2)
        worstHomo = std::max(worstHomo,
            (adjointSE3(T * T2) - Ad * adjointSE3(T2)).cwiseAbs().maxCoeff());

        // Ad(T^-1) = Ad(T)^-1
        worstInv = std::max(worstInv,
            (adjointSE3(inverseSE3(T)) * Ad - Matrix6d::Identity()).cwiseAbs().maxCoeff());
    }

    std::cout << "Ad: |T exp(xi) T^-1 - exp(Ad xi)|: " << worstConj   << "\n"
              << "Ad: |Ad - Ad_sophus|:              " << worstSophus << "\n"
              << "Ad: |Ad(T T2) - Ad(T) Ad(T2)|:     " << worstHomo   << "\n"
              << "Ad: |Ad(T^-1) Ad(T) - I|:          " << worstInv    << "\n"
              << "Ad: |Ad(I) - I|:                   " << worstId     << "\n";

    return worstConj < tol && worstSophus < tolExact && worstHomo < tol
        && worstInv < tol && worstId == 0.0;
}

int main() {
    std::mt19937 rng(42);

    // Run every group (no short-circuiting) so one failure doesn't hide another.
    bool ok = true;
    ok &= testExpLogSO3(rng);
    ok &= testLogEdgeCasesSO3();
    ok &= testJacobians(rng);
    ok &= testJacobianInverses(rng);
    ok &= testExpLogSE3(rng);
    ok &= testLogEdgeCasesSE3();
    ok &= testAdjointSE3(rng);
    
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}