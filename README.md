# asv-sift

## ASV Detector Source File Location

The SIFT detector source file is here:

```
opencv-4.12.0/modules/features2d/src/sift.dispatch.cpp
```

## SIFT Detector Source File Location

The SIFT detector source file is here:

```
opencv-4.12.0/modules/features2d/src/sift.dispatch.cpp
```

## Building OpenCV on macOS

### Prerequisites

```bash
brew install cmake pkg-config
brew install jpeg libpng libtiff openexr
brew install eigen tbb
```

### Build Steps

```bash
# Clone OpenCV repository
cd opencv-4.12.0
mkdir build && cd build

# Configure with CMake
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_ENABLE_NONFREE=ON ..

# Build (use -j flag for parallel compilation)
make -j$(sysctl -n hw.ncpu)

# Install on the computer (optional)
sudo make install
```

## Building OpenCV on Windows

### Prerequisites

1. Install Visual Studio 2019
2. Install CMake

### Build Steps

```bash
# Navigate to OpenCV directory
cd opencv-4.12.0
mkdir build
cd build

# Configure with CMake (adjust generator version as needed)
cmake -G "Visual Studio 16 2019" -A x64 -D CMAKE_BUILD_TYPE=RELEASE ..

# Build using CMake
cmake --build . --config Release

# Install on the computer (optional)
cmake --build . --config Release --target INSTALL
```

### Notes

- Add OpenCV bin directory to your system PATH: `C:\opencv\build\install\x64\vc16\bin`
- Set `OpenCV_DIR` environment variable to: `C:\opencv\build\install`
