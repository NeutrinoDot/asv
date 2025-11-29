// descriptors.cpp
#include "descriptors.h"

#include <stdexcept>

// --------------------------------------
// SIFT baseline implementation
// --------------------------------------
class SiftDescriptor : public IDescriptorExtractor {
public:
    SiftDescriptor() {
        // Using default SIFT parameters; these are standard OpenCV defaults.
        // SOURCE: OpenCV docs for cv::SIFT::create().
        sift_ = cv::SIFT::create();
    }

    DescriptorKind kind() const override {
        return DescriptorKind::SIFT;
    }

    void detectAndCompute(const cv::Mat& image,
                          DescriptorSet& out) override {
        out.keypoints.clear();
        out.descriptors.release();

        if (image.empty()) {
            throw std::invalid_argument("SiftDescriptor::detectAndCompute: empty image");
        }

        // 1) Detect keypoints
        // 2) Compute descriptors
        // This is the standard SIFT pipeline.
        sift_->detectAndCompute(image, cv::noArray(), out.keypoints, out.descriptors);
    }

private:
    cv::Ptr<cv::SIFT> sift_;
};

// --------------------------------------
// ASV-SIFT implementation
// --------------------------------------
//
// Logic:
// - Detect keypoints with SIFT (same as baseline, for fair comparison).
// - Use cv::ASV to compute descriptors at multiple scales.
// - Use the "real" ASV descriptors as the descriptor matrix for matching.
//
// ASSUMPTIONS:
// - cv::ASV::compute(image, keypoints, realDescriptors, binaryDescriptors) is implemented such
//   that realDescriptors is N x D, aligned with the input keypoints vector (dropped keypoints
//   should be consistently removed from both).
//
class AsvSiftDescriptor : public IDescriptorExtractor {
public:
    AsvSiftDescriptor() {
        // Underlying SIFT for keypoint detection.
        sift_ = cv::SIFT::create();

        // ASV parameters:
        //   nScales    = 6 (typical value in ASV paper for multi-scale sampling)
        //   scaleStep  = 1.2 (common multiplicative step between scales in SIFT-like pyramids)
        //   nThreshold1, nThreshold2: left as tunable; here example defaults.
        //   isInter    = false initially (no cross-scale interpolation).
        //
        // SOURCE:
        //   Yang et al., "Accumulated Stability Voting" (CVPR 2016) multi-scale sampling design.
        int nScales = 6;
        double scaleStep = 1.2;
        double nThreshold1 = 1.0; // TODO: tune based on paper / experiments
        double nThreshold2 = 1.0; // TODO: tune based on paper / experiments
        bool isInter = false;

        asv_ = cv::ASV::create(
            /*detectorType*/ 0,
            nScales,
            scaleStep,
            nThreshold1,
            nThreshold2,
            isInter
        );
    }

    DescriptorKind kind() const override {
        return DescriptorKind::ASV_SIFT;
    }

    void detectAndCompute(const cv::Mat& image,
                          DescriptorSet& out) override {
        out.keypoints.clear();
        out.descriptors.release();

        if (image.empty()) {
            throw std::invalid_argument("AsvSiftDescriptor::detectAndCompute: empty image");
        }

        // 1) Detect keypoints using SIFT
        sift_->detect(image, out.keypoints);

        if (out.keypoints.empty()) {
            // No keypoints: descriptors remain empty, this is fine.
            return;
        }

        // 2) Compute ASV descriptors for those keypoints
        cv::Mat realDescriptors;
        cv::Mat binaryDescriptors; // currently unused in evaluation
        asv_->compute(image, out.keypoints, realDescriptors, binaryDescriptors);

        // We assume ASV either preserves alignment or drops invalid keypoints consistently.
        out.descriptors = realDescriptors;
    }

private:
    cv::Ptr<cv::SIFT> sift_;
    cv::Ptr<cv::ASV> asv_;
};

// Factory
std::unique_ptr<IDescriptorExtractor> createExtractor(DescriptorKind kind) {
    switch (kind) {
        case DescriptorKind::SIFT:
            return std::make_unique<SiftDescriptor>();
        case DescriptorKind::ASV_SIFT:
            return std::make_unique<AsvSiftDescriptor>();
        default:
            throw std::invalid_argument("Unknown DescriptorKind");
    }
}
