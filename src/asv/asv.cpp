// src/asv/asv.cpp
#include "asv/asv.hpp"

#include <iostream>
#include <numeric>
#include <algorithm>
#include <opencv2/features2d.hpp>

#include <util/asv_utils.hpp>

using namespace asv::util;

namespace cv {

  ASV::ASV(const int detectorType, const int _nScales,
           const float scale_min, const float scale_max,
           const int _nThreshold1, const int _nThreshold2)
    : nScales(_nScales),
    nThreshold1(_nThreshold1),
    nThreshold2(_nThreshold2) {

    CV_Assert(nScales > 0);
    CV_Assert(scale_min > 0 && scale_max > 0 && scale_max >= scale_min);
    CV_Assert(nThreshold1 >= 1 && nThreshold2 >= 1);

    switch (detectorType) {
    case 1: // ORB
      detector = ORB::create();
      descriptorSize = 32;
      descriptorType = CV_8U;
      break;
    case 2: // BRISK
      detector = BRISK::create();
      descriptorSize = 64;
      descriptorType = CV_8U;
      break;
    default: // 0 or other -> SIFT
      detector = SIFT::create();
      descriptorSize = 128;
      descriptorType = CV_32F;
      break;
    }

    binaryDescriptorSize = descriptorSize * nThreshold2;

    realDescriptors.clear();
    binaryDescriptors.clear();

    computeScaleFactors(nScales, scale_min, scale_max, scaleFactors);
  }

  Ptr<ASV> ASV::create(const int detectorType, const int nScales,
                       const float scale_min, const float scale_max,
                       const int nThreshold1, const int nThreshold2) {
    return makePtr<ASV>(detectorType, nScales, scale_min, scale_max,
                        nThreshold1, nThreshold2);
  }

  // detect and compute descriptors
  void ASV::detectAndCompute(InputArray _image,
                             InputArray _mask,
                             std::vector<KeyPoint>& keypoints,
                             OutputArray _descriptor,
                             bool useProvidedKeypoints) {
    CV_Assert(!detector.empty());

    Mat image = _image.getMat();
    Mat mask = _mask.getMat();
    if (image.empty()) {
      _descriptor.release();
      CV_Error(Error::StsBadArg, "Input image is empty.");
    }

    // detect keypoints if not provided
    if (!useProvidedKeypoints || keypoints.empty()) {
      detector->detect(image, keypoints, mask);
    }

    if (keypoints.empty()) {
      _descriptor.release();
      CV_Error(Error::StsBadArg, "No keypoints detected.");
    }

    // compute descriptors
    Mat real, binary;
    compute(image, keypoints, real, binary);

    if (asvType == Real) {
      real.copyTo(_descriptor);
    }
    else if (asvType == Binary) {
      binary.copyTo(_descriptor);
    }
  }

  // compute descriptors
  void ASV::compute(InputArray _image,
                    const std::vector<KeyPoint>& keypoints,
                    OutputArray _descriptor, OutputArray _binaryDescriptors) {
    CV_Assert(!detector.empty());

    Mat image = _image.getMat();
    if (keypoints.empty() || image.empty()) {
      _descriptor.release();
      _binaryDescriptors.release();
      std::string msg = "Keypoints empty: " + std::to_string(keypoints.empty()) +
        " || Image empty: " + std::to_string(image.empty());
      CV_Error(Error::StsBadArg, msg.c_str());
    }

    // extract an array of descriptors at multiple scales per keypoint
    std::vector<std::vector<Mat>> multiScaleDescriptors;
    extractMultiScaleDescriptors(image, keypoints, multiScaleDescriptors);

    const int N = static_cast<int>(multiScaleDescriptors.size());
    if (N == 0) {
      _descriptor.release();
      _binaryDescriptors.release();
      CV_Error(Error::StsBadArg, "No valid keypoints after multi-scale extraction.");
    }

    // first stage ASV (1M): multi-thresholding + accumulated stability voting
    computeRealASV(multiScaleDescriptors);

    if (!realDescriptors.empty()) {
      Mat realMat = Mat::zeros(N, descriptorSize, CV_32F);
      CV_Assert(static_cast<int>(realDescriptors.size()) == N);
      for (int i = 0; i < N; ++i) {
        const Mat& desc = realDescriptors[i];
        if (desc.empty()) continue;
        CV_Assert(desc.rows == 1 && desc.cols == descriptorSize);
        CV_Assert(desc.type() == CV_32F);
        desc.row(0).copyTo(realMat.row(i));
      }
      realMat.copyTo(_descriptor);
    }
    else {
      _descriptor.release();
    }

    // Second-stage ASV (1M2M): multiple thresholds on 1M descriptor
    computeBinaryASV();

    if (_binaryDescriptors.needed()) {
      if (!binaryDescriptors.empty()) {
        CV_Assert(static_cast<int>(binaryDescriptors.size()) == N);
        Mat binMat = Mat::zeros(N, binaryDescriptorSize, CV_8U);
        for (int i = 0; i < N; ++i) {
          const Mat& desc = binaryDescriptors[i];
          if (desc.empty()) continue;
          CV_Assert(desc.rows == 1 && desc.cols == binaryDescriptorSize);
          CV_Assert(desc.type() == CV_8U);
          desc.row(0).copyTo(binMat.row(i));
        }
        binMat.copyTo(_binaryDescriptors);
      }
      else {
        _binaryDescriptors.release();
      }
    }
  }

  // extract descriptors at multiple scales around each keypoint
  // Output: multiScaleDescriptors [keypoint][scale][Mat]
  void ASV::extractMultiScaleDescriptors(const Mat& image,
                                         const std::vector<KeyPoint>& keypoints,
                                         std::vector<std::vector<Mat>>& multiScaleDescriptors) {
    const int nKeypoints = static_cast<int>(keypoints.size());
    multiScaleDescriptors.clear();
    multiScaleDescriptors.assign(nKeypoints, std::vector<Mat>(nScales));

    if (nKeypoints == 0) return;

    for (int scaleIdx = 0; scaleIdx < nScales; ++scaleIdx) {
      // 1) Build scaled keypoints for this scale
      std::vector<KeyPoint> scaledKeypoints;
      scaledKeypoints.reserve(nKeypoints);
      const float scaleFactor = static_cast<float>(scaleFactors[scaleIdx]);

      // 2) Scale keypoint size by scaleFactor
      for (int i = 0; i < nKeypoints; ++i) {
        KeyPoint kp = keypoints[i];
        kp.size *= scaleFactor;
        scaledKeypoints.push_back(kp);
      }

      // 3) Compute descriptors for all keypoints at this scale
      Mat descs;
      detector->compute(image, scaledKeypoints, descs);

      if (descs.empty() || descs.rows != nKeypoints) {
        for (int i = 0; i < nKeypoints; ++i) {
          multiScaleDescriptors[i][scaleIdx].release();
        }
        continue;
      }

      // 4) Slice each row into multiScaleDescriptors[i][scaleIdx]
      for (int i = 0; i < nKeypoints; ++i) {
        multiScaleDescriptors[i][scaleIdx] = descs.row(i).clone();
      }
    }
  }

  // First stage ASV (1M)
  void ASV::computeRealASV(const std::vector<std::vector<Mat>>& multiScaleDescriptors) {
    const int N = static_cast<int>(multiScaleDescriptors.size());
    if (N == 0) return;

    realDescriptors.clear();
    realDescriptors.resize(N);

    // Process each keypoint
    for (int i = 0; i < N; ++i) {
      const std::vector<Mat>& scaleDescs = multiScaleDescriptors[i];

      // if this keypoint has no scale descriptors, leave empty
      if (scaleDescs.empty()) {
        realDescriptors[i] = Mat();
        continue;
      }

      // Compute stability votes for each descriptor dimension
      Mat featureASV = Mat::zeros(nThreshold1, descriptorSize, CV_32F);
      computeFeatureASV(scaleDescs, featureASV);

      // Copy to output
      realDescriptors[i] = featureASV;
    }
  }

  // calculate real stability vote of a multi-scale feature descriptor
  // keypointDescriptors: array of descriptors at multiple scales for a keypoint
  // votes: output stability votes (nThreshold1 x descriptorSize)
  void ASV::computeFeatureASV(const std::vector<Mat>& keypointDescriptors,
                              Mat& votes) const {
    if (keypointDescriptors.empty()) return;

    const int dim = descriptorSize;

    // 1) Collect indices of valid scales
    std::vector<int> valid;
    valid.reserve(keypointDescriptors.size());
    for (int s = 0; s < static_cast<int>(keypointDescriptors.size()); ++s) {
      if (!keypointDescriptors[s].empty()) {
        valid.push_back(s);
      }
    }

    const int Sv = static_cast<int>(valid.size());
    if (Sv < 2) return;  // need at least 2 valid scales to compare

    // 2) Number of unique scale pairs C(Sv, 2)
    const int numPairs = Sv * (Sv - 1) / 2;

    // Build M: dim x numPairs, where each column is |x_s1 - x_s2|
    Mat M(dim, numPairs, CV_32F);
    int col = 0;

    for (int a = 0; a < Sv - 1; ++a) {
      const int s1 = valid[a];
      for (int b = a + 1; b < Sv; ++b) {
        const int s2 = valid[b];

        Mat diffCol;
        computeAbsDiff(keypointDescriptors[s1],
                       keypointDescriptors[s2], diffCol);

        CV_Assert(diffCol.rows == dim && diffCol.cols == 1);

        // Copy column into M
        diffCol.col(0).copyTo(M.col(col));
        ++col;
      }
    }
    CV_Assert(col == numPairs);

    // 3) First-stage multi-thresholding
    const int num_q = nThreshold1 + 1;

    Mat outVec;
    multiThresholdMatrix(M, num_q, outVec);

    CV_Assert(outVec.rows == dim && outVec.cols == 1);

    // 4) Convert outVec to votes
    for (int j = 0; j < dim; ++j) {
      votes.at<float>(0, j) = outVec.at<float>(j, 0);
    }
  }

  // Calculate absolute difference between two descriptors across all dimensions
  void ASV::computeAbsDiff(const Mat& desc1, const Mat& desc2, Mat& diffCol) const {
    CV_Assert(!desc1.empty() && !desc2.empty());
    CV_Assert(desc1.size() == desc2.size());

    Mat d1f, d2f;
    ensureRowFloat(desc1, d1f, descriptorSize);
    ensureRowFloat(desc2, d2f, descriptorSize);

    // Compute |d2 - d1| as dim x 1
    Mat diffRow;
    absdiff(d1f, d2f, diffRow);

    diffCol = diffRow.t(); // transpose to dim x 1
  }

  // Convert real-valued descriptors to binary descriptor
  void ASV::computeBinaryASV() {
    binaryDescriptors.clear();

    const size_t N = realDescriptors.size();
    if (N == 0) return;

    const int nPairs = nChooseK(nScales, 2);
    if (nPairs <= 0) return;

    // Precompute thresholds for each 2nd-stage slot
    std::vector<float> thresholds(nThreshold2);
    for (int k = 0; k < nThreshold2; ++k) {
      thresholds[k] = static_cast<float>(std::floor(nThreshold1 * nPairs / (nThreshold2 + 1) * (k + 1)));
    }

    binaryDescriptors.resize(N);

    for (size_t i = 0; i < N; ++i) {
      const Mat& real = realDescriptors[i];
      if (real.empty()) {
        binaryDescriptors[i] = Mat();
        continue;
      }

      CV_Assert(real.rows == 1 && real.cols == descriptorSize && real.type() == CV_32F);

      Mat bin = Mat::zeros(1, binaryDescriptorSize, CV_8U);
      for (int k = 0; k < nThreshold2; ++k) {
        const float thr = thresholds[k];
        const int offset = k * descriptorSize;

        for (int j = 0; j < descriptorSize; ++j) {
          const float v = real.at<float>(0, j);
          if (v >= thr) {
            bin.at<uchar>(0, offset + j) = static_cast<uchar>(1);
          }
        }
      }
      binaryDescriptors[i] = bin;
    }
  }
} // namespace cv