/** @brief Class for computing descriptors using Accumulated Stability Voting
 (ASV) algorithm based on: Yang, T.-Y., Lin, Y.-Y. & Chuang, Y.-Y.,
 "Accumulated Stability Voting: A Robust Descriptor from Descriptors of
 Multiple Scales," CVPR 2016.

ASV calculates descriptors at multiple scales and uses stability voting to
create more better descriptors. The ASV descriptor can work with any Feature2D
detector (SIFT, ORB, BRISK, ...)
*/

#include "asv.h"

#include <iostream>
#include <numeric>
#include <opencv2/features2d.hpp>

namespace cv {

// constructor
ASV::ASV(const int detectorType = 0, const int _nScales, const float scale_min,
         const float scale_max, const double _scaleStep,
         const double _nThreshold1 = 1, const double _nThreshold2 = 1,
         const bool _isInter = false)
    : nScales(_nScales),
      nThreshold1(_nThreshold1),
      nThreshold2(_nThreshold2),
      isInter(_isInter) {
  // default detector
  detector = SIFT::create(0, 3, 0, 0);
  descriptorSize = 128;
  descriptorType = CV_32F;

  // precompute scale factors
  computeScaleFactors(scale_min, scale_max);
}

Ptr<ASV> ASV::create(const int detectorType, int nScales, const float scale_min,
                     const float scale_max, double nThreshold1,
                     double nThreshold2, bool isInter) {
  return makePtr<ASV>(detectorType, nScales, scale_min, scale_max, nThreshold1,
                      nThreshold2, isInter);
}

// compute descriptors
void ASV::compute(const InputArray _image,
                  const std::vector<KeyPoint>& keypoints,
                  OutputArray _descriptor, OutputArray _binaryDescriptors) {
  CV_Assert(!detector.empty());

  Mat image = _image.getMat();

  if (keypoints.empty()) {
    _descriptor.release();
    _binaryDescriptors.release();
    CV_Error(Error::StsBadArg, "No keypoints detected.");
  }

  // extract an array of descriptors at multiple scales per keypoint
  std::vector<std::vector<Mat>> multiScaleDescriptors;
  extractMultiScaleDescriptors(image, keypoints, multiScaleDescriptors);

  // compute stability voting on multiScaleDescriptors
  std::vector<Mat> realDescriptors;
  computeRealASV(multiScaleDescriptors, realDescriptors);

  // compute binary descriptors from real-valued descriptors
  std::vector<Mat> binaryDescriptors;
  computeBinaryASV(realDescriptors, binaryDescriptors);
}

// extract descriptors at multiple scales around each keypoint
// Output: multiScaleDescriptors [keypoint][scale][Mat]
void ASV::extractMultiScaleDescriptors(
    const Mat& image, const std::vector<KeyPoint>& keypoints,
    std::vector<std::vector<Mat>>& multiScaleDescriptors) {
  int nKeypoints = (int)keypoints.size();
  multiScaleDescriptors.clear();
  multiScaleDescriptors.resize(nKeypoints);

  // extract multiple descriptors per keypoint
  for (int i = 0; i < nKeypoints; i++) {
    std::vector<Mat> scaledDescriptors;  // [descriptor per scale]
    bool isValidKeypoint = true;
    scaledDescriptors.reserve(nScales);

    // generate scale space and extract descriptors
    for (int scaleIdx = 0; scaleIdx < nScales; scaleIdx++) {
      KeyPoint scaledKeypoint = keypoints[i];  // deep copy keypoint
      scaledKeypoint.size *= scaleFactors[scaleIdx];

      Mat descriptor;
      std::vector<KeyPoint> oneKp{scaledKeypoint};  // compute needs a vector
      detector->compute(image, oneKp, descriptor);

      if (descriptor.empty()) {
        isValidKeypoint = false;
        break;
      }

      scaledDescriptors.push_back(descriptor);
    }

    // skip keypoints where descriptors could not be computed at all scales
    // check with multiScaleDescriptors[i].empty() later
    if (!isValidKeypoint) {
      continue;
    }

    multiScaleDescriptors[i] = std::move(scaledDescriptors);
  }
}

void ASV::computeRealASV(
    const std::vector<std::vector<Mat>>& multiScaleDescriptors,
    std::vector<Mat>& realDescriptors) {
  // Check for empty input
  if (multiScaleDescriptors.empty()) {
    return;
  }

  int nKeypoints = (int)multiScaleDescriptors.size();
  realDescriptors.clear();
  realDescriptors.resize(nKeypoints);

  // Process each keypoint
  for (int i = 0; i < nKeypoints; i++) {
    const std::vector<Mat>& scaleDescs = multiScaleDescriptors[i];

    // Skip invalid keypoints
    if (scaleDescs.empty()) {
      continue;
    }

    // Compute stability votes for each descriptor dimension
    Mat featureASV = Mat::zeros(1, descriptorSize, CV_32F);

    // Calculate stability votes between all unique scale pairs
    for (size_t s1 = 0; s1 < nScales; s1++) {
      for (size_t s2 = s1 + 1; s2 < nScales; s2++) {
        computeStabilityVote(scaleDescs[s1], scaleDescs[s2], featureASV);
      }
    }

    // Copy to output
    realDescriptors[i] = std::move(featureASV);
  }
}

// calculate real stability vote of a multi-scale feature descriptor
void ASV::computeFeatureASV(const std::vector<Mat>& keypointDescriptors,
                            Mat& votes) {
  // Skip invalid keypoints
  if (keypointDescriptors.empty()) {
    return;
  }

  // Calculate stability votes between all unique scale pairs
  for (size_t s1 = 0; s1 < nScales; s1++) {
    for (size_t s2 = s1 + 1; s2 < nScales; s2++) {
      computeStabilityVote(keypointDescriptors[s1], keypointDescriptors[s2],
                           votes);
    }
  }
}

// Calculate stability votes between two descriptors
void ASV::computeStabilityVote(const Mat& desc1, const Mat& desc2, Mat& votes) {
  CV_Assert(desc1.size() == desc2.size() && desc1.type() == desc2.type());
  Mat diff = Mat::zeros(1, descriptorSize, descriptorType);
  for (int i = 0; i < descriptorSize; i++) {
    diff.at<float>(i) =
        std::abs(desc1.at<float>(i)) - std::abs(desc2.at<float>(i));
  }

  // Calculate the median of the values in diff
  Mat diffSorted;
  cv::sort(diff, diffSorted, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  float threshold = diffSorted.at<float>(descriptorSize / 2);

  // Accumulate votes based on threshold
  for (int i = 0; i < descriptorSize; i++) {
    if (diff.at<float>(i) < threshold) {
      votes.at<float>(i) += 1.0f;
    }
  }
}

// Convert real-valued descriptors to binary descriptor
void ASV::computeBinaryASV(const std::vector<Mat>& realDescriptors,
                           std::vector<Mat>& binaryDescriptors) {
  // TODO: Implement binary descriptor conversion
}

void ASV::computeScaleFactors(const float scale_min, const float scale_max) {
  float scaleStep = (scale_max - scale_min) / (nScales - 1);
  scaleFactors.resize(nScales);
  for (int i = 0; i < nScales; i++) {
    scaleFactors[i] = scale_min + i * scaleStep;
  }
}

}  // namespace cv
