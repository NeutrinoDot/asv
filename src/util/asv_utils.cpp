// src/util/asv_utils.cpp

#include "util/asv_utils.hpp"

#include <algorithm>
#include <utility>
#include <cmath>

namespace asv {
  namespace util {

    void computeScaleFactors(const int nScales,
                             const float scale_min,
                             const float scale_max,
                             std::vector<double>& scaleFactors) {
      if (nScales <= 1) {
        scaleFactors.clear();
        scaleFactors.push_back(1.0);
        return;
      }

      const double step = (scale_max - scale_min) / (nScales - 1);
      scaleFactors.resize(nScales);

      for (int i = 0; i < nScales; ++i) {
        scaleFactors[i] = scale_min + i * step;
      }
    }

    int nChooseK(int n, int k) {
      if (k > n) return 0;
      if (k == 0 || k == n) return 1;

      k = std::min(k, n - k);
      int c = 1;

      for (int i = 0; i < k; ++i) {
        c = c * (n - i) / (i + 1);
      }
      return c;
    }

    void multiThresholdMatrix(const cv::Mat& inMat, int num_q, cv::Mat& outVec) {
      CV_Assert(inMat.type() == CV_32F);

      const int dim = inMat.rows;
      const int cols = inMat.cols;

      if (dim == 0 || cols == 0) return;

      outVec = cv::Mat::zeros(dim, 1, CV_32F);

      int segment = dim / num_q;
      if (segment <= 0) segment = 1;

      // buffer for (value, index) pairs per column
      std::vector<std::pair<float, int>> buf(dim);

      for (int c = 0; c < cols; ++c) {
        // Fill and sort by value ascending
        for (int r = 0; r < dim; ++r) {
          buf[r].first = inMat.at<float>(r, c);
          buf[r].second = r;
        }
        std::sort(buf.begin(), buf.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first < b.first;
                  });

        // Assign quantized votes: smallest differences get highest vote
        for (int lvl = 0; lvl < num_q; ++lvl) {
          int aa = lvl * segment;
          int bb = (lvl + 1) * segment - 1;

          if (aa >= dim) break;
          if (bb >= dim) bb = dim - 1;

          float voteVal = static_cast<float>(num_q - lvl - 1);
          for (int idx = aa; idx <= bb; ++idx) {
            int r = buf[idx].second;
            outVec.at<float>(r, 0) += voteVal;
          }
        }
      }
    }

  } // namespace util
} // namespace asv
