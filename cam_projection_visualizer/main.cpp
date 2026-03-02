#include <cstdio>
#include <opencv2/viz.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <Eigen/Geometry>
#include <ostream>

Eigen::Matrix3d rotationFromEuler(double roll, double pitch, double yaw) {
    // Convert degrees to radians
    double roll_rad = roll * M_PI / 180.0;
    double pitch_rad = pitch * M_PI / 180.0;
    double yaw_rad = yaw * M_PI / 180.0;
    
    // Method 1: Using Eigen's rotation classes
    Eigen::AngleAxisd rollAngle(roll_rad, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch_rad, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw_rad, Eigen::Vector3d::UnitZ());
    
    Eigen::Quaterniond q =  rollAngle * pitchAngle * yawAngle;
    Eigen::Matrix3d R = q.matrix();
    
    return R;
}

// Method 2: Manual construction
Eigen::Matrix3d rotationFromEulerManual(double roll, double pitch, double yaw) {
    double r = roll * M_PI / 180.0;
    double p = pitch * M_PI / 180.0;
    double y = yaw * M_PI / 180.0;
    
    // Rotation around X
    Eigen::Matrix3d Rx;
    Rx << 1,      0,       0,
          0, cos(r), -sin(r),
          0, sin(r),  cos(r);
    
    // Rotation around Y
    Eigen::Matrix3d Ry;
    Ry << cos(p),  0, sin(p),
               0,  1,      0,
         -sin(p),  0, cos(p);
    
    // Rotation around Z
    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0,
          sin(y),  cos(y), 0,
               0,       0, 1;
    
    return Rx * Ry * Rz;
}
Eigen::MatrixXd createCube(double size = 1.0) {
    Eigen::MatrixXd vertices(8, 3);  // 8 vertices, 3 coordinates
    
    double s = size / 2.0;
    
    // Fill using comma initializer (row by row)
    vertices << -s, -s, -s,   // vertex 0
                 s, -s, -s,   // vertex 1
                 s,  s, -s,   // vertex 2
                -s,  s, -s,   // vertex 3
                -s, -s,  s,   // vertex 4
                 s, -s,  s,   // vertex 5
                 s,  s,  s,   // vertex 6
                -s,  s,  s;   // vertex 7
    
    return vertices;
}

class Camera {
public:
    // Intrinsic matrix (3×3)
    Eigen::Matrix3d K;
    
    // Constructor: initialize K from parameters
    Camera(double fx, double fy, double cx, double cy) {
        // Method 1: Comma initializer (cleanest)
        K << fx,  0, cx,
              0, fy, cy,
              0,  0,  1;
        
        std::cout << "Camera intrinsic matrix K:\n" << K << std::endl;
    }
    
    // Project 3D points to 2D
    Eigen::MatrixXd project(const Eigen::MatrixXd& X_world,
                            const Eigen::Matrix3d& R,
                            const Eigen::Vector3d& t) {
        // X_world is N×3 (N points, 3 coordinates each)
        // We need to transpose to 3×N for matrix multiplication
        
        // Step 1: Transform to camera coordinates
        // X_cam = R * X_world^T + t (broadcasted)
        Eigen::MatrixXd X_cam = R * X_world.transpose();  // 3×N
        
        // Add translation to each column
        X_cam.colwise() += t;
        
        // Step 2: Project with K
        Eigen::MatrixXd x_homogeneous = K * X_cam;  // 3×N
        
        // Step 3: Normalize (divide by z-coordinate)
        Eigen::MatrixXd x_image(2, X_world.rows());  // 2×N
        
        for (int i = 0; i < X_world.rows(); ++i) {
            x_image(0, i) = x_homogeneous(0, i) / x_homogeneous(2, i);
            x_image(1, i) = x_homogeneous(1, i) / x_homogeneous(2, i);
        }
        
        return x_image.transpose();  // Return N×2
    }
    
    // Get full projection matrix P = K[R|t] (3×4)
    Eigen::Matrix<double, 3, 4> getProjectionMatrix(
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t) {
        
        // Create 3×4 matrix [R|t]
        Eigen::Matrix<double, 3, 4> Rt;
        Rt.block<3, 3>(0, 0) = R;  // First 3 columns = R
        Rt.block<3, 1>(0, 3) = t;  // Last column = t
        
        // P = K * [R|t]
        return K * Rt;
    }
};
void visualizeBoth(const Eigen::MatrixXd& vertices_3d,
                   const Eigen::MatrixXd& projected_2d,
                   const std::vector<std::pair<int, int>>& edges,
                   const Eigen::Matrix3d& R,
                   const Eigen::Vector3d& t,
                   const Eigen::Matrix3d& K) {
    
    // ===== Part 1: 3D Visualization =====
    cv::viz::Viz3d window_3d("3D Scene");
    window_3d.setBackgroundColor(cv::viz::Color::black());
    
    std::vector<cv::Point3d> points_3d;
    for (int i = 0; i < vertices_3d.rows(); ++i) {
        points_3d.push_back(cv::Point3d(vertices_3d(i, 0),
                                        vertices_3d(i, 1),
                                        vertices_3d(i, 2)));
    }
    
    // Draw 3D edges
    for (size_t i = 0; i < edges.size(); ++i) {
        cv::viz::WLine line(points_3d[edges[i].first], 
                            points_3d[edges[i].second], 
                            cv::viz::Color::cyan());
        window_3d.showWidget("edge3d_" + std::to_string(i), line);
    }
    
    cv::viz::WCoordinateSystem axes(2.0);
    window_3d.showWidget("axes", axes);
    Eigen::Vector3d cam_pos = -R.transpose() * t;

    // Build a 4x4 OpenCV affine transform from R and cam_pos
    // OpenCV viz expects the camera-to-world transform
    cv::Matx33d R_cv;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R_cv(i, j) = R(i, j);

    // R^T rotates from camera space back to world space
    cv::Matx33d Rt_cv = R_cv.t();

    cv::Vec3d pos_cv(cam_pos.x(), cam_pos.y(), cam_pos.z());

    cv::Affine3d cam_pose(Rt_cv, pos_cv);

    // Draw a camera frustum widget
    // K parameters: focal length and principal point
    cv::Matx33d K_cv;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            K_cv(i, j) = K(i, j);

    cv::Size img_size(640, 480);
    double frustum_scale = 1.5;  // how large to draw it

    cv::viz::WCameraPosition frustum(K_cv, frustum_scale, cv::viz::Color::yellow());
    cv::viz::WCameraPosition cam_axes(0.5);  // small coordinate frame at camera

    window_3d.showWidget("camera_frustum", frustum, cam_pose);
    window_3d.showWidget("camera_axes", cam_axes, cam_pose);


    // ===== Part 2: 2D Projection =====
    cv::Mat image_2d(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
    
    std::vector<cv::Point2f> points_2d;
    for (int i = 0; i < projected_2d.rows(); ++i) {
        points_2d.push_back(cv::Point2f(projected_2d(i, 0), 
                                        projected_2d(i, 1)));
    }
    
    // Draw 2D edges
    for (const auto& edge : edges) {
        cv::line(image_2d, points_2d[edge.first], points_2d[edge.second],
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    
    for (const auto& pt : points_2d) {
        cv::circle(image_2d, pt, 5, cv::Scalar(0, 255, 0), -1);
    }
    
    // ===== Show Both =====
    // cv::imshow("2D Projection", image_2d);
    window_3d.spinOnce(1, true);
    cv::Affine3d initial_pose = window_3d.getViewerPose();
    // Render loop - no waitKey needed here
    while (!window_3d.wasStopped()) {
        window_3d.spinOnce(1, true);
        // cv::Affine3d initial_pose = window_3d.getViewerPose();
        // Force focus to the 2D window so waitKey actually works
        cv::imshow("2D Projection", image_2d);
        int key = cv::waitKey(1);
        
        if (key == 'q') break;
        if (key == 't') {
            window_3d.setViewerPose(cv::viz::makeCameraPose(
                cv::Vec3d(0, 10, 0),
                cv::Vec3d(0, 0, 0),
                cv::Vec3d(0, 0,10)));
        }
        if (key == 'f') {
            window_3d.setViewerPose(cv::viz::makeCameraPose(
                cv::Vec3d(0, 0, 10),
                cv::Vec3d(0, 0, 0),
                cv::Vec3d(0, 10, 0)));
        }
        if (key == 's') {
            window_3d.setViewerPose(cv::viz::makeCameraPose(
                cv::Vec3d(10, 0, 0),
                cv::Vec3d(0, 0, 0),
                cv::Vec3d(0, 10, 0)));
        }
        if (key == 'r'){
            window_3d.setViewerPose(initial_pose);
        }
    }
}
int main() {
    // Setup
    Camera camera(800, 800, 320, 240);
    Eigen::MatrixXd cube = createCube(.75);
    Eigen::Matrix3d R = rotationFromEuler(45, 45, 45);
    Eigen::Vector3d t(0, 0, 3);
    
    // Project
    Eigen::MatrixXd projected = camera.project(cube, R, t);
    
    // Define edges
    std::vector<std::pair<int, int>> edges = {
        {0,1}, {1,2}, {2,3}, {3,0},  // back
        {4,5}, {5,6}, {6,7}, {7,4},  // front
        {0,4}, {1,5}, {2,6}, {3,7}   // connect
    };
    visualizeBoth(cube, projected, edges, R, t, camera.K);

    return 0;
}