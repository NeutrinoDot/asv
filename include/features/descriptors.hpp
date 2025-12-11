// include/features/descriptors.hpp
#pragma once

#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

/**
 * @brief Enumeration of descriptor types supported by the feature extractor.
 * ASV descriptors are represented as either real-valued (ASV_REAL) or binary
 * (ASV_BINARY), depending on ASVConfig or ASV type.
 */
enum class DescriptorType {
  SIFT,
  SURF,
  ASV_REAL,
  ASV_BINARY,
  ORB,
  BRISK,
  CUSTOM
};

/**
 * @brief Configuration parameters for the ASV descriptor extractor.
 */
struct ASVConfig {
  int nScales = 5;
  float scale_min = 0.7f;
  float scale_max = 1.4f;
  int nThreshold1 = 1;
  int nThreshold2 = 1;
  int detectorType = 0; // 0=SIFT, 1=ORB, 2=BRISK, 3=SURF
};

/**
 * @brief Contains keypoints and the associated descriptor matrix.
 */
struct DescriptorSet {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

/**
 * @brief Abstract Interface for all feature descriptors used by the evaluation script.
 */
class IDescriptor {
public:
  virtual ~IDescriptor() = default;
  virtual DescriptorType type() const = 0;

  /**
   * @brief Detect keypoints and compute descriptors for an image.
   * @param image Input image.
   * @param out Output DescriptorSet containing keypoints and descriptors.
   */
  virtual void detectAndCompute(const cv::Mat& image,
                                DescriptorSet& out) = 0;
};

/**
 * @brief Factory for constructing descriptor extractors by type.
 */
std::unique_ptr<IDescriptor> createDescriptor(DescriptorType type,
                                              const ASVConfig& asvConfig = ASVConfig{});
