#include "util/eval_utils.hpp"

#include <opencv2/features2d.hpp>
#include <cmath>
#include <limits>

cv::Point2f projectPoint(const cv::Point2f& ptA, const cv::Mat& H_AtoB) {
  CV_Assert(H_AtoB.rows == 3 && H_AtoB.cols == 3);
  CV_Assert(H_AtoB.type() == CV_64F);

  double x = ptA.x;
  double y = ptA.y;

  const double* h = H_AtoB.ptr<double>(0);
  double X = h[0] * x + h[1] * y + h[2];
  double Y = h[3] * x + h[4] * y + h[5];
  double W = h[6] * x + h[7] * y + h[8];

  if (std::abs(W) < 1e-9) {
    return cv::Point2f(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
  }

  return cv::Point2f(static_cast<float>(X / W),
                     static_cast<float>(Y / W));
}

int inferNormType(const cv::Mat& descriptors) {
  int depth = descriptors.depth();

  if (depth == CV_8U) {
    return cv::NORM_HAMMING;
  }

  return cv::NORM_L2;
}
