// include/util/eval_utils.hpp
#ifndef EVAL_UTILS_HPP
#define EVAL_UTILS_HPP

#include <opencv2/features2d.hpp>
#include <cmath>
#include <limits>

/**
 * @brief Project a 2D point using a 3x3 homography H.
 * 
 * @return Point2f, invalid point if W is near zero.
 */
cv::Point2f projectPoint(const cv::Point2f& ptA, const cv::Mat& H_AtoB);

/**
 * @brief Infer the appropriate distance norm for a descriptor matrix.
 * - Binary descriptors are matched with NORM_HAMMING.
 * - Real-valued descriptors are matched with NORM_L2.
 */
int inferNormType(const cv::Mat& descriptors);

#endif // !EVAL_UTILS_HPP
