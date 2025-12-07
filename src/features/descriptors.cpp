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
class AsvDescriptor : public IDescriptor {
public:
  explicit AsvDescriptor(DescriptorType t, const ASVConfig& cfg) : type_(t) {
    asv_ = cv::ASV::create(cfg.detectorType,
                           cfg.nScales,
                           cfg.scale_min,
                           cfg.scale_max,
                           cfg.nThreshold1,
                           cfg.nThreshold2);

    if (type_ == DescriptorType::ASV_BINARY) {
      asv_->setASVType(cv::ASV::ASVType::Binary);
    }
    else {
      asv_->setASVType(cv::ASV::ASVType::Real);
    }
  }

  DescriptorType type() const override {
    return type_;
  }

  void detectAndCompute(const cv::Mat& image,
                        DescriptorSet& out) override {
    out.keypoints.clear();
    out.descriptors.release();

    if (image.empty()) {
      throw std::invalid_argument("ASVDescriptor::detectAndCompute: Input image is empty.");
    }
    asv_->detectAndCompute(image, cv::noArray(), out.keypoints, out.descriptors);
  }
private:
  DescriptorType type_;
  cv::Ptr<cv::ASV> asv_;
};

// Factory function to create descriptor extractors
std::unique_ptr<IDescriptor> createDescriptor(DescriptorType type, const ASVConfig& asvConfig) {
  switch (type) {
  case DescriptorType::SIFT:
    return std::make_unique<SiftDescriptor>();
  case DescriptorType::ASV_REAL:
    return std::make_unique<AsvDescriptor>(type, asvConfig);
  case DescriptorType::ASV_BINARY:
    return std::make_unique<AsvDescriptor>(type, asvConfig);
  default:
    throw std::invalid_argument("createDescriptor: Unsupported descriptor type.");
  }
}
