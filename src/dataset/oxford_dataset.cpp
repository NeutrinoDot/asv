// src/dataset/oxford_dataset.cpp
#include "dataset/oxford_dataset.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <stdexcept>

DatasetLoader::DatasetLoader(const std::vector<ImagePairSpec>& specs)
  : specs_(specs) {}

// Load homography (3x3) from a text file
cv::Mat DatasetLoader::loadHographyFormTxt(const std::string& path) const {
  std::ifstream in(path);

  if (!in.is_open()) {
    throw std::runtime_error("Failed to open homography file: " + path);
  }

  double h[9]{};
  for (int i = 0; i < 9; ++i) {
    if (!(in >> h[i])) {
      throw std::runtime_error("Invalid homography format in: " + path);
    }
  }

  cv::Mat H(3, 3, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      H.at<double>(r, c) = h[r * 3 + c];
    }
  }
  return H;
}

std::vector<ImagePair> DatasetLoader::loadAll() const {
  std::vector<ImagePair> pairs;
  pairs.reserve(specs_.size());

  for (const auto& spec : specs_) {
    cv::Mat imgA = cv::imread(spec.pathImageA, cv::IMREAD_GRAYSCALE);
    cv::Mat imgB = cv::imread(spec.pathImageB, cv::IMREAD_GRAYSCALE);
    if (imgA.empty()) {
      throw std::runtime_error("Failed to load image A: " + spec.pathImageA);
    }
    if (imgB.empty()) {
      throw std::runtime_error("Failed to load image B: " + spec.pathImageB);
    }

    cv::Mat H_AtoB = loadHographyFormTxt(spec.pathHomography);

    ImagePair pair;
    pair.id     = spec.id;
    pair.imgA = imgA;
    pair.imgB = imgB;
    pair.H_AtoB = H_AtoB;

    pairs.push_back(std::move(pair));
  }
  return pairs;
}