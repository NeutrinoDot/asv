// src/asv/asv.cpp
#include "asv/asv.hpp"

#include <iostream>
#include <numeric>
#include <algorithm>
#include <opencv2/features2d.hpp>

namespace cv {

  ASV::ASV(const int detectorType, const int _nScales,
           const float scale_min, const float scale_max,
           const double _nThreshold1, const double _nThreshold2,
           const bool _isInter)
    : nScales(_nScales),
    nThreshold1(_nThreshold1),
    nThreshold2(_nThreshold2),
    isInter(_isInter) {

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
      detector = SIFT::create(0, 3, 0, 0);
      descriptorSize = 128;
      descriptorType = CV_32F;
      break;
    }

    realDescriptors.clear();
    binaryDescriptors.clear();

    computeScaleFactors(scale_min, scale_max);
  }

  Ptr<ASV> ASV::create(const int detectorType, const int nScales,
                       const float scale_min, const float scale_max,
                       const double nThreshold1, const double nThreshold2,
                       const bool isInter) {
    return makePtr<ASV>(detectorType, nScales, scale_min, scale_max,
                        nThreshold1, nThreshold2, isInter);
  }

  // compute descriptors
  void ASV::compute(const InputArray _image,
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

    // compute stability voting on multiScaleDescriptors
    computeRealASV(multiScaleDescriptors);
    
    Mat realMat = Mat::zeros(N, descriptorSize, CV_32F);
    if (!realDescriptors.empty()) {
      CV_Assert(static_cast<int>(realDescriptors.size()) == N);
      for (int i = 0; i < N; ++i) {
        const Mat& desc = realDescriptors[i];
        if (desc.empty()) continue;
        CV_Assert(desc.rows == 1 && desc.cols == descriptorSize);
        CV_Assert(desc.type() == CV_32F);
        desc.row(0).copyTo(realMat.row(i));
      }
    }
    realMat.copyTo(_descriptor);

    // compute binary descriptors from real-valued descriptors
    computeBinaryASV();
    
    if (_binaryDescriptors.needed()) {
      if (!binaryDescriptors.empty()) {
        CV_Assert(static_cast<int>(binaryDescriptors.size()) == N);
        Mat binMat = Mat::zeros(N, descriptorSize, CV_8U);
        for (int i = 0; i < N; ++i) {
          const Mat& desc = binaryDescriptors[i];
          if (desc.empty()) continue;
          CV_Assert(desc.rows == 1 && desc.cols == descriptorSize);
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
    std::cout << "Extracting multi-scale descriptors for "
      << nKeypoints << " keypoints." << std::endl;

    for (int scaleIdx = 0; scaleIdx < nScales; ++scaleIdx) {
      // 1) Build scaled keypoints for this scale
      std::vector<KeyPoint> scaledKeypoints;
      scaledKeypoints.reserve(nKeypoints);
      for (int i = 0; i < nKeypoints; ++i) {
        KeyPoint kp = keypoints[i];  // deep copy keypoint
        kp.size *= static_cast<float>(scaleFactors[scaleIdx]);
        scaledKeypoints.push_back(kp);
      }

      // 2) Compute descriptors for all keypoints at this scale
      Mat descs;
      detector->compute(image, scaledKeypoints, descs);

      if (descs.empty() || descs.rows != nKeypoints) {
        for (int i = 0; i < nKeypoints; ++i) {
          multiScaleDescriptors[i][scaleIdx].release();
        }
        continue;
      }

      // 3) Slice each row into multiScaleDescriptors[i][scaleIdx]
      for (int i = 0; i < nKeypoints; ++i) {
        multiScaleDescriptors[i][scaleIdx] = descs.row(i).clone();
      }
      std::cout << "Extracted descriptors at scale index "
        << scaleIdx << std::endl;
    }
  }

  void ASV::computeRealASV(const std::vector<std::vector<Mat>>& multiScaleDescriptors) {
    const int N = static_cast<int>(multiScaleDescriptors.size());
    if (N == 0) return;

    realDescriptors.clear();
    realDescriptors.resize(N);

    // Process each keypoint
    for (int i = 0; i < N; ++i) {
      const std::vector<Mat>& scaleDescs = multiScaleDescriptors[i];

      int S = 0;
      for (const Mat& d : scaleDescs) {
        if (!d.empty()) ++S;
      }
      if (S < 2) {
        // Not enough valid scalepairs to compute stability votes
        realDescriptors[i] = Mat();
        continue;
      }

      // Compute stability votes for each descriptor dimension
      Mat featureASV = Mat::zeros(1, descriptorSize, CV_32F);
      computeFeatureASV(scaleDescs, featureASV);

      // Copy to output
      realDescriptors[i] = featureASV;
    }
  }

  // calculate real stability vote of a multi-scale feature descriptor
  void ASV::computeFeatureASV(const std::vector<Mat>& keypointDescriptors,
                              Mat& votes) {
    const int S = static_cast<int>(keypointDescriptors.size());
    if (S == 0) return;

    // Calculate stability votes between all unique scale pairs
    for (size_t s1 = 0; s1 < S; ++s1) {
      if (keypointDescriptors[s1].empty()) continue;
      for (size_t s2 = s1 + 1; s2 < S; ++s2) {
        if (keypointDescriptors[s2].empty()) continue;
        computeStabilityVote(keypointDescriptors[s1], keypointDescriptors[s2],
                             votes);
      }
    }
  }

  // Calculate stability votes between two descriptors
  void ASV::computeStabilityVote(const Mat& desc1, const Mat& desc2, Mat& votes) const {
    Mat d1f, d2f;
    if (desc1.type() != CV_32F) {
      desc1.convertTo(d1f, CV_32F);
    }
    else {
      d1f = desc1;
    }
    if (desc2.type() != CV_32F) {
      desc2.convertTo(d2f, CV_32F);
    }
    else {
      d2f = desc2;
    }

    CV_Assert(d1f.rows == 1 && d1f.cols == descriptorSize);
    CV_Assert(d2f.rows == 1 && d2f.cols == descriptorSize);
    CV_Assert(votes.rows == 1 && votes.cols == descriptorSize);
    CV_Assert(votes.type() == CV_32F);

    // diff(i) = |d1(i)| - |d2(i)|  (per dimension)
    Mat diff(1, descriptorSize, CV_32F);
    for (int i = 0; i < descriptorSize; ++i) {
      float v1 = d1f.at<float>(0, i);
      float v2 = d2f.at<float>(0, i);
      diff.at<float>(0, i) = std::abs(v1) - std::abs(v2);
    }

    // Simplified local thresholding: median of diff
    Mat sorted;
    cv::sort(diff, sorted, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);
    float median = sorted.at<float>(0, descriptorSize / 2);

    for (int j = 0; j < descriptorSize; ++j) {
      if (diff.at<float>(0, j) < median) {
        votes.at<float>(0, j) += 1.0f;
      }
    }
  }

  // Convert real-valued descriptors to binary descriptor
  void ASV::computeBinaryASV() {
    binaryDescriptors.clear();

    const size_t N = realDescriptors.size();
    if (N == 0) return;

    const int nPairs = nChooseK(nScales, 2);
    if (nPairs <= 0) return;

    const double num1m = nThreshold1;
    const double num2m = std::max(1.0, nThreshold2);
    const float baseThreshold = static_cast<float>(std::floor(num1m * nPairs / (num2m + 1.0)));

    binaryDescriptors.resize(N);

    for (size_t i = 0; i < N; ++i) {
      const Mat& real = realDescriptors[i];
      if (real.empty()) {
        binaryDescriptors[i] = Mat();
        continue;
      }

      CV_Assert(real.rows == 1 && real.cols == descriptorSize);
      CV_Assert(real.type() == CV_32F);

      Mat bin = Mat::zeros(1, descriptorSize, CV_8U);
      for (int j = 0; j < descriptorSize; ++j) {
        const float vote = real.at<float>(0, j);
        if (vote > baseThreshold) {
          bin.at<uchar>(0, j) = static_cast<uchar>(1);
        }
      }
      binaryDescriptors[i] = bin;
    }
  }

  void ASV::computeScaleFactors(const float scale_min, const float scale_max) {
    if (nScales <= 1) {
      scaleFactors.assign(1, 1.0);
      return;
    }

    const double step = (scale_max - scale_min) / (nScales - 1);
    scaleFactors.resize(nScales);
    for (int i = 0; i < nScales; ++i) {
      scaleFactors[i] = scale_min + i * step;
    }
  }

  int ASV::nChooseK(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    k = std::min(k, n - k);
    int c = 1;
    for (int i = 0; i < k; ++i) {
      c = c * (n - i) / (i + 1);
    }
    return c;
  }
} // namespace cv