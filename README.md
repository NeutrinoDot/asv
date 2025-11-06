# asv-sift

## SIFT Detector Source File Location

The SIFT (Scale-Invariant Feature Transform) detector source file is located in the OpenCV library's `modules/features2d/src/` directory in the `sift.dispatch.cpp` file. In the OpenCV source tree, you can find it at:

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
cmake -D CMAKE_BUILD_TYPE=RELEASE ..

# Build (use -j flag for parallel compilation)
make -j$(sysctl -n hw.ncpu)

# Install on the computer (optional)
sudo make install
```

## Building OpenCV on Windows

### Prerequisites

1. Install [Visual Studio](https://visualstudio.microsoft.com/) (2019 or later) with C++ development tools
2. Install [CMake](https://cmake.org/download/) (3.15 or later)
3. Download OpenCV source code

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

# Install (optional, requires admin privileges)
cmake --build . --config Release --target INSTALL
```

### Notes

- Add OpenCV bin directory to your system PATH: `C:\opencv\build\install\x64\vc16\bin`
- Set `OpenCV_DIR` environment variable to: `C:\opencv\build\install`
