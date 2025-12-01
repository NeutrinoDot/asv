// include/dataset/oxford_dataset.hpp
#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

// Specification of an image pair to load.
struct ImagePairSpec {
  std::string id;
  std::string pathImageA;
  std::string pathImageB;
  std::string pathHomography;  // Path to homography from A to B
};

// In-memory representation of a loaded pair.
struct ImagePair {
  std::string id;
  cv::Mat imgA;
  cv::Mat imgB;
  cv::Mat H_AtoB;  // Homography from A to B
};

class DatasetLoader {
public:
  explicit DatasetLoader(const std::vector<ImagePairSpec>& specs);

  std::vector<ImagePair> loadAll() const;

private:
  std::vector<ImagePairSpec> specs_;
  cv::Mat loadHographyFormTxt(const std::string& path) const;
};