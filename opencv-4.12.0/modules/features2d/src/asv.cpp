/** @brief Class for computing descriptors using Accumulated Stability Voting
 (ASV) algorithm based on: Yang, T.-Y., Lin, Y.-Y. & Chuang, Y.-Y.,
 "Accumulated Stability Voting: A Robust Descriptor from Descriptors of
 Multiple Scales," CVPR 2016.

ASV calculates descriptors at multiple scales and uses stability voting to
create more better descriptors. The ASV descriptor can work with any Feature2D
detector (SIFT, ORB, BRISK, ...)
*/

#include <iostream>
#include <numeric>

#include "opencv2/features2d.hpp"
// #include "precomp.hpp"

namespace cv {

// constructor
ASV::ASV(const int detectorType = 0, const int _nScales,
         const double _scaleStep, const double _nThreshold1,
         const double _nThreshold2, const bool _isInter, )
    : nScales(_nScales),
      scaleStep(_scaleStep),
      nThreshold1(_nThreshold1),
      nThreshold2(_nThreshold2),
      isInter(_isInter) {
  detector = SIFT::create(contrastThreshold = 0,
                          edgeThreshold = 0);  // default detector
}

Ptr<ASV> ASV::create(const int detectorType, int nScales, double scaleStep,
                     double nThreshold1, double nThreshold2, bool isInter) {
  return makePtr<ASV>(nScales, scaleStep, nThreshold1, nThreshold2, isInter,
                      detectorType);
}

// compute descriptors
void ASV::compute(InputArray _image, std::vector<KeyPoint>& keypoints,
                  OutputArray _descriptors, OutputArray _binaryDescriptors) {
  CV_Assert(!detector.empty());

  Mat image = _image.getMat();

  if (keypoints.empty()) {
    _descriptors.release();
    CV_Error(Error::StsBadArg, "No keypoints detected.");
  }

  // store multiscale descriptors as vector [keypoint][scale][Mat]
  std::vector<std::vector<Mat>> multiScaleDescriptors;
  extractMultiScaleDescriptors(image, keypoints, multiScaleDescriptors, );

  // compute stability voting on multiScaleDescriptors
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
      double scaleFactor = std::pow(scaleStep, scaleIdx - nScales / 2.0);
      scaledKeypoint.size *= static_cast<float>(scaleFactor);

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
}  // namespace cv
