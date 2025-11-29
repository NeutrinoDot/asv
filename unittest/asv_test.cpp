#include "asv.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

// Load an image from file.
// Parameters: image - output image, filepath - path to the image file
// Returns: void
// Preconditions: image file exists at filepath. image is a valid Mat.
// Postconditions: image file is saved as a Mat in image.
void loadImage(Mat& image, const std::string& filepath) {
  image = imread(filepath);
  if (image.empty()) {
    std::cout << "Error loading image: " << filepath << "\n";
    exit(1);
  }
  std::cout << "Loaded image: " << filepath << " (" << image.cols << "x"
            << image.rows << ")\n";
}

// Detect keypoints and compute descriptors using a specified detector.
// Parameters: keypoints - output vector of detected keypoints, descriptors -
// output matrix of keypoint descriptors, image - input image, detectorType -
// type of keypoint detector to use (0 = SIFT, 1 = ORB, 2 = AKAZE), display -
// whether to display keypoints.
// Returns: void
// Preconditions: image is a valid Mat. keypoints and descriptors are valid
// outputs.
// Postconditions: keypoints and descriptors are filled with detected keypoints
// and their descriptors. If display is true, keypoints are shown in a user
// display window.
void detectKeypoints(std::vector<KeyPoint>& keypoints, Mat& descriptors,
                     const Mat& image, int detectorType, bool display = false) {
  if (detectorType == 0) {  // SIFT
    Ptr<SIFT> detector = SIFT::create();
    detector->detectAndCompute(image, noArray(), keypoints, descriptors);
  } else if (detectorType == 1) {  // ORB
    Ptr<ORB> detector = ORB::create();
    detector->detectAndCompute(image, noArray(), keypoints, descriptors);
  } else if (detectorType == 2) {  // AKAZE
    Ptr<AKAZE> detector = AKAZE::create();
    detector->detectAndCompute(image, noArray(), keypoints, descriptors);
  } else {
    std::cout << "Invalid detector type specified.\n";
    return;
  }

  std::cout << "Detected " << keypoints.size() << " keypoints.\n";

  if (!display) {
    return;
  }

  Mat output;
  drawKeypoints(image, keypoints, output);

  imshow("Keypoints", output);
  std::cout << "Displaying keypoints. Press any key to continue...\n";
  waitKey(0);
}

int main() {
  // Load images
  const char* img1_file = "kittens1.jpg";
  // const char* img2_file = "kittens2.jpg";
  Mat img1, img2;
  loadImage(img1, img1_file);
  // loadImage(img2, img2_file);

  // Detect keypoints
  std::vector<KeyPoint> img1_kp, img2_kp;
  Mat img1_descriptors, img2_descriptors;
  detectKeypoints(img1_kp, img1_descriptors, img1, 0, false);  // SIFT detector
  // detectKeypoints(img2_kp, img2_descriptors, img2, 0, false);

  // Create an ASV instance
  Ptr<ASV> asv = ASV::create(0, 5, 1.414, 1, 1, false);

  // Set a target position
  Mat realDescriptor;
  Mat binaryDescriptor;
  asv->compute(img1, img1_kp, realDescriptor, binaryDescriptor);

  return 0;
}