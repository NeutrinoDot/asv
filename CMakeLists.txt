cmake_minimum_required(VERSION 3.10)
project(asv_eval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/asv/asv.cpp
    src/util/asv_utils.cpp
    src/dataset/oxford_dataset.cpp
    src/dataset/dataset_utils.cpp
    src/features/descriptors.cpp
    src/eval/matching.cpp
    src/metrics/metrics.cpp
    src/main_oxford_eval.cpp
)

# Create executable
add_executable(oxford_eval ${SOURCES})

# Link libraries
target_link_libraries(oxford_eval ${OpenCV_LIBS})
