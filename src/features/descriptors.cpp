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

// ----------------- BRISK Descriptor Extractor ----------------
class BriskDescriptor : public IDescriptor {
public:
  BriskDescriptor() {
    brisk_ = cv::BRISK::create();
  }

  DescriptorType type() const override {
    return DescriptorType::BRISK;
  }

  void detectAndCompute(const cv::Mat& image,
                        DescriptorSet& out) override {
    out.keypoints.clear();
    out.descriptors.release();

    if (image.empty()) {
      throw std::invalid_argument("BRISKDescriptor::detectAndCompute: Input image is empty.");
    }
    brisk_->detectAndCompute(image, cv::noArray(), out.keypoints, out.descriptors);
  }
private:
  cv::Ptr<cv::BRISK> brisk_;
};

// ----------------- ORB Descriptor Extractor ----------------
class OrbDescriptor : public IDescriptor {
public:
  OrbDescriptor() {
    orb_ = cv::ORB::create();
  }

  DescriptorType type() const override {
    return DescriptorType::ORB;
  }

  void detectAndCompute(const cv::Mat& image,
                        DescriptorSet& out) override {
    out.keypoints.clear();
    out.descriptors.release();

    if (image.empty()) {
      throw std::invalid_argument("ORBDescriptor::detectAndCompute: Input image is empty.");
    }
    orb_->detectAndCompute(image, cv::noArray(), out.keypoints, out.descriptors);
  }
private:
  cv::Ptr<cv::ORB> orb_;
};

// ----------------- SURF Descriptor Extractor ----------------
class SurfDescriptor : public IDescriptor {
public:
  SurfDescriptor() {
    surf_ = cv::xfeatures2d::SURF::create();
  }

  DescriptorType type() const override {
    return DescriptorType::SURF;
  }

  void detectAndCompute(const cv::Mat& image,
                        DescriptorSet& out) override {
    out.keypoints.clear();
    out.descriptors.release();

    if (image.empty()) {
      throw std::invalid_argument("SIFTDescriptor::detectAndCompute: Input image is empty.");
    }
    surf_->detectAndCompute(image, cv::noArray(), out.keypoints, out.descriptors);
  }
private:
  cv::Ptr<cv::xfeatures2d::SURF> surf_;
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

// ----------------- Descriptor Factory Function ----------------
std::unique_ptr<IDescriptor> createDescriptor(DescriptorType type, const ASVConfig& asvConfig) {
  switch (type) {
  case DescriptorType::SIFT:
    return std::make_unique<SiftDescriptor>();
  case DescriptorType::BRISK:
    return std::make_unique<BriskDescriptor>();
  case DescriptorType::ORB:
    return std::make_unique<OrbDescriptor>();
  case DescriptorType::SURF:
    return std::make_unique<SurfDescriptor>();
  case DescriptorType::ASV_REAL:
    return std::make_unique<AsvDescriptor>(type, asvConfig);
  case DescriptorType::ASV_BINARY:
    return std::make_unique<AsvDescriptor>(type, asvConfig);
  default:
    throw std::invalid_argument("createDescriptor: Unsupported descriptor type.");
  }
}
