// include/dataset/oxford_dataset.hpp
#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

/** 
 * Specification of an image pair to load.
 */
struct ImagePairSpec {
  std::string id;
  std::string pathImageA;
  std::string pathImageB;
  std::string pathHomography;  // Path to homography from A to B
};

/**
 * In-memory representation of a loaded pair.
 */
struct ImagePair {
  std::string id;
  cv::Mat imgA;
  cv::Mat imgB;
  cv::Mat H_AtoB;  // Homography from A to B
};

/** 
 * Loader for Oxford dataset image pairs. 
 */
class DatasetLoader {
public:
  /**
   * @brief Construct a DatasetLoader with a list of image pair specifications.
   */
  explicit DatasetLoader(const std::vector<ImagePairSpec>& specs);

  /**
   * @brief Load all image pairs specified in the dataset.
   * @return A vector of loaded ImagePair objects.
   */
  std::vector<ImagePair> loadAll() const;

private:
  std::vector<ImagePairSpec> specs_;

  /** 
   * Helper to load a single image pair given its specification. 
   */
  cv::Mat loadHographyFormTxt(const std::string& path) const;
};
