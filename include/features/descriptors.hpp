// include/features/descriptors.hpp
#pragma once

#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

enum class DescriptorType {
	SIFT,
	ASV_SIFT, 
	ORB,
	BRISK,
	CUSTOM
};

struct DescriptorSet {
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

class IDescriptor {
public:
	virtual ~IDescriptor() = default;
	virtual DescriptorType type() const = 0;

	virtual void detectAndCompute(const cv::Mat& image, 
																DescriptorSet& out) = 0;
};

// Factory function to create descriptor for given type
std::unique_ptr<IDescriptor> createDescriptor(DescriptorType type);