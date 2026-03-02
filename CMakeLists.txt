cmake_minimum_required(VERSION 3.10)
project(viz)
cmake_policy(SET CMP0144 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(viz cam_projection_visualizer/main.cpp)

target_link_libraries(viz 
    ${OpenCV_LIBS}
    Eigen3::Eigen
)