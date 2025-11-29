// descriptors.h
// -------------------------------------------------------------------------------------------------
// Provides an abstraction over different descriptor types (SIFT baseline vs ASV-SIFT).
//
// INPUT:
// - Single grayscale image (cv::Mat, CV_8U).
//
// OUTPUT:
// - DescriptorSet:
//     keypoints  : vector<cv::KeyPoint> (detected in the given image).
//     descriptors: cv::Mat (N x D, CV_32F), row i corresponds to keypoints[i].
//
// ASSUMPTIONS:
// - We use OpenCV's cv::SIFT for detection. For ASV-SIFT, SIFT does detection, and our cv::ASV
//   class (from asv.h / asv.cpp) does descriptor computation over SIFT keypoints.
// - Descriptor distance is Euclidean (L2), consistent with SIFT/float descriptors.
//
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <vector>
#include <string>

#include "asv.h"  // our implementation of ASV

enum class DescriptorKind {
    SIFT,
    ASV_SIFT
};

struct DescriptorSet {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors; // float matrix, N x D
};

// Abstract interface so evaluation code doesn't care which descriptor we use.
class IDescriptorExtractor {
public:
    virtual ~IDescriptorExtractor() = default;

    // Which descriptor type this extractor implements.
    virtual DescriptorKind kind() const = 0;

    // Detect keypoints and compute descriptors for a single image.
    //
    // INPUT:
    //   image: grayscale image (CV_8U).
    // OUTPUT:
    //   out.keypoints   : filled with detected keypoints.
    //   out.descriptors : rows aligned with out.keypoints; CV_32F.
    //
    // NOTE:
    //   Implementations should clear 'out' before repopulating.
    virtual void detectAndCompute(const cv::Mat& image,
                                  DescriptorSet& out) = 0;
};

// Factory helper to create SIFT or ASV-SIFT extractor based on DescriptorKind.
std::unique_ptr<IDescriptorExtractor> createExtractor(DescriptorKind kind);

