// include/features/descriptors.hpp
#pragma once

#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

/**
 * Descriptor types supported.
 */
enum class DescriptorType {
  SIFT,
  ASV_REAL,
  ASV_BINARY,
  ORB,
  BRISK,
  CUSTOM
};

/**
 * Configuration for ASV descriptor.
 */
struct ASVConfig {
  int nScales = 5;
  float scale_min = 0.7f;
  float scale_max = 1.4f;
  int nThreshold1 = 1;
  int nThreshold2 = 1;
  int detectorType = 0; // 0=SIFT, 1=ORB, 2=BRISK
};

/**
 * A set of descriptors and their corresponding keypoints.
 */
struct DescriptorSet {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

/**
 * Interface for feature descriptors.
 */
class IDescriptor {
public:
  virtual ~IDescriptor() = default;
  virtual DescriptorType type() const = 0;

  /** 
   * @brief Detect keypoints and compute descriptors for an image.
   * @param image The input image.
   * @param out The output DescriptorSet containing keypoints and descriptors.
   */
  virtual void detectAndCompute(const cv::Mat& image,
                                DescriptorSet& out) = 0;
};

// Factory function to create descriptor for given type
std::unique_ptr<IDescriptor> createDescriptor(DescriptorType type, 
                                              const ASVConfig& asvConfig = ASVConfig{});