// Header for Accumulated Stability Voting (ASV) descriptor

#ifndef ASV_H
#define ASV_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv {

/** @brief Class for computing descriptors using Accumulated Stability Voting
 (ASV) algorithm based on: Yang, T.-Y., Lin, Y.-Y. & Chuang, Y.-Y.,
 "Accumulated Stability Voting: A Robust Descriptor from Descriptors of
 Multiple Scales," CVPR 2016.

ASV calculates descriptors at multiple scales and uses stability voting to
create more better descriptors. The ASV descriptor can work with any Feature2D
detector (SIFT, ORB, BRISK, ...)
*/
class CV_EXPORTS_W ASV : public Feature2D {
 public:
  /** Constructor
  @param detector The feature detector used by ASV.
  @param nScales Number of scales to sample for each keypoint.
  @param scaleStep Step size between scales.
  @param nThreshold1 Number of thresholds for each bin for 1st-stage
  thresholding
  @param nThreshold2 Number of thresholds for 2nd-stage thresholding
  @param isInter Flag to interpolate features between scales
  */
  ASV(const int detectorType, const int nScales, const float scale_min,
      const float scale_max, const double scaleStep, const double nThreshold1,
      const double nThreshold2, const bool isInter);

  /** updates current object */
  CV_WRAP static Ptr<ASV> create(const int detectorType, const int nScales,
                                 const float scale_min, const float scale_max,
                                 const double nThreshold1,
                                 const double nThreshold2, const bool isInter);

  /** Computes ASV descriptors
   */
  virtual void compute(const InputArray image,
                       const std::vector<KeyPoint>& keypoints,
                       OutputArray realDescriptors,
                       OutputArray binaryDescriptors);

 protected:
  Ptr<Feature2D> detector;
  int nScales;
  std::vector<float> scaleFactors;
  double nThreshold1;
  double nThreshold2;
  bool isInter;
  int descriptorSize;
  int descriptorType;

  // extract an array of descriptors at multiple scales per keypoint
  void extractMultiScaleDescriptors(
      const Mat& image, const std::vector<KeyPoint>& keypoints,
      std::vector<std::vector<Mat>>& multiScaleDescriptors);

  // perform stability voting given multi-scale descriptors
  void computeRealASV(
      const std::vector<std::vector<Mat>>& multiScaleDescriptors,
      std::vector<Mat>& realDescriptors);

  // calculate real stability vote of a multi-scale feature descriptor
  void computeFeatureASV(const std::vector<Mat>& keypointDescriptors,
                         Mat& votes);

  // calculate stability votes between two descriptors
  void computeStabilityVote(const Mat& desc1, const Mat& desc2, Mat& votes);

  // Convert real-valued descriptors to binary descriptor
  void computeBinaryASV(const std::vector<Mat>& realDescriptors,
                        std::vector<Mat>& binaryDescriptors);

  void computeScaleFactors(const float scale_min, const float scale_max);
};

} /* namespace cv */

#endif
