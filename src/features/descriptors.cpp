// src/features/descriptors.cpp
#include "features/descriptors.hpp"
#include "asv/asv.hpp"

#include <stdexcept>

// ----------------- SIFT Descriptor Extractor ----------------
class SiftDescriptor : public IDescriptor {
public:
  SiftDescriptor() {
    sift_ = cv::SIFT::create();
  }

  DescriptorType type() const override {
    return DescriptorType::SIFT;
  }

  void detectAndCompute(const cv::Mat& image, 
                        DescriptorSet& out) override {
    out.keypoints.clear();
    out.descriptors.release();

    if (image.empty()) {
      throw std::invalid_argument("SIFTDescriptor::detectAndCompute: Input image is empty.");
    }
    sift_->detectAndCompute(image, cv::noArray(), out.keypoints, out.descriptors);
  }
private:
  cv::Ptr<cv::SIFT> sift_;
};

// ----------------- ASV Descriptor Extractor ----------------
class AsvSiftDescriptor : public IDescriptor {
public:
  AsvSiftDescriptor() {
    sift_ = cv::SIFT::create();

    int nScales = 5;
    float scale_min = 0.7f;
    float scale_max = 1.4f;
    double nThreshold1 = 1.0;
    double nThreshold2 = 1.0;
    bool isInter = false;

    asv_ = cv::ASV::create(/*detectorType*/ 0, 
                           nScales, 
                           scale_min, 
                           scale_max, 
                           nThreshold1, 
                           nThreshold2, 
                           isInter);
  }

  DescriptorType type() const override {
    return DescriptorType::ASV_SIFT;
  }

  void detectAndCompute(const cv::Mat& image,
                        DescriptorSet& out) override {
    out.keypoints.clear();
    out.descriptors.release();

    if (image.empty()) {
      throw std::invalid_argument("AsvSiftDescriptor::detectAndCompute: Input image is empty.");
    }

    sift_->detect(image, out.keypoints);
    if (out.keypoints.empty()) {
      return;
    }

    cv::Mat realDescriptors, binaryDescriptors;
    asv_->compute(image, out.keypoints, realDescriptors, binaryDescriptors);

    out.descriptors = realDescriptors;
  }
private:
  cv::Ptr<cv::SIFT> sift_;
  cv::Ptr<cv::ASV> asv_;
};

// Factory function to create descriptor extractors
std::unique_ptr<IDescriptor> createDescriptor(DescriptorType type) {
  switch (type) {
    case DescriptorType::SIFT:
      return std::make_unique<SiftDescriptor>();
    case DescriptorType::ASV_SIFT:
      return std::make_unique<AsvSiftDescriptor>();
    default:
      throw std::invalid_argument("createDescriptor: Unsupported descriptor type.");
  }
}