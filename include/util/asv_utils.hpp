// include/util/asv_utils.hpp
#ifndef ASV_UTILS_HPP
#define ASV_UTILS_HPP

#include <vector>
#include <opencv2/core.hpp>

namespace asv {
  namespace util {
    /**
     * @brief Compute evenly spaced scale factors between [scale_min, scale_max].
     *
     * @param nScales Number of scales.
     * @param scale_min Minimum scale factor.
     * @param scale_max Maximum scale factor.
     * @param scaleFactors (out) Vector to store computed scale factors.
     *
     * If nScales == 1, scaleFactors will contain a single value of 1.0.
     */
    void computeScaleFactors(const int nScales,
                             const float scale_min,
                             const float scale_max,
                             std::vector<double>& scaleFactors);

    /**
    * @brief Compute n choose k (binomial coefficient).
    *
    * @param n Total number of items.
    * @param k Subset size.
    * @return int The binomial coefficient C(n, k), or 0 if k > n.
    */
    int nChooseK(int n, int k);

    /**
    * @brief Local multi-thresholding
    *
    * inMat is dim x K, where each column i a vector of absolute differences
    * for one scale pair. This function:
    *   - sort each column,
    *   - partition indices into num_q quantization levels,
    *   - assign votes from smallest to largest level,
    *   - sum votes across columns to get outVec (dim x 1).
    *
    * @param inMat Input matrix of size dim x K (CV_32F).
    * @param num_q Number of quantization levels (num_q = nThreshold1 + 1).
    * @param outVec (out) Output vector of size dim x 1 (CV_32F).
    */
    void multiThresholdMatrix(const cv::Mat& inMat, int num_q, cv::Mat& outVec);

    /**
    * @brief Ensure a 1 x dim CV_32F "row descriptor" view.
    *
    * If input is already CV_32F, dst references it directly. Otherwise, convert
    * to CV_32F. Asserts that the final size is 1 x dim.
    *
    * @param src Input descriptor matrix.
    * @param dst (out) Output 1 x dim CV_32F descriptor matrix.
    * @param dim Expected descriptor dimension.
    */
    inline void ensureRowFloat(const cv::Mat& src, cv::Mat& dst, const int dim) {
      if (src.type() != CV_32F) {
        src.convertTo(dst, CV_32F);
      }
      else {
        dst = src;
      }
      CV_Assert(dst.rows == 1 && dst.cols == dim);
    }

  } // namespace util
} // namespace asv

#endif // ASV_UTILS_HPP