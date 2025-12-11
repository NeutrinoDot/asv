// include/asv/asv.hpp
#ifndef ASV_HPP
#define ASV_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv {

  /**
   * @class ASV
   * @brief Accumulated Stability Voting (ASV) descriptor extractor.
   *
   * This class implements the descriptor described in:
   *
   *  Yang, T.-Y., Lin, Y.-Y. & Chuang, Y.-Y.,
   *  "Accumulated Stability Voting: A Robust Descriptor from Descriptors of
   *  Multiple Scales," CVPR 2016.
   *
   * The goal of ASV is to compute a more robust local feature descriptor by:
   *  1. Extracting descriptors for each keypoint at multiple scales.
   *  2. Computing element-wise "stability votes" measuring how descriptor
   *     dimensions change across scale pairs.
   *  3. Accumulating these votes into a final real-valued descriptor.
   *  4. Optionally converting the real descriptor into a binary descriptor.
   *
   * This class wraps around any OpenCV Feature2D descriptor and applies the
   * ASV algorithm.
   */
  class CV_EXPORTS_W ASV : public Feature2D {
  public:

    /**
    * @brief Slects the output ASV descriptor type.
    */
    enum ASVType {
      Real,
      Binary
    };

    /**
     * @brief Constructor.
     *
     * @param detectorType Integer representing which OpenCV Feature2D to use.
     *        0 (default) or other -> SIFT,
     *        1 -> ORB,
     *        2 -> BRISK,
     *        4 -> SURF.
     * @param nScales Number of sampled scales around each keypoint.
     * @param scale_min Minimum relative scale factor.
     * @param scale_max Maximum relative scale factor (>= scale_min).
     * @param nThreshold1 Number of thresholds for 1st-stage thresholding
     * @param nThreshold2 Number of thresholds for 2nd-stage thresholding
     *
     * The constructor initializes the ASV sampler, scale factors, and allocates
     * per-feature descriptor buffers.
     */
    ASV(const int detectorType, const int nScales,
        const float scale_min, const float scale_max,
        const int nThreshold1, const int nThreshold2);

    /**
     * @brief Factory function for creating an ASV instance with OpenCV-style Ptr..
     */
    CV_WRAP static Ptr<ASV> create(const int detectorType = 0,
                                   const int nScales = 5,
                                   const float scale_min = 0.7f,
                                   const float scale_max = 1.4f,
                                   const int nThreshold1 = 1,
                                   const int nThreshold2 = 1);

    using Feature2D::compute;
    using Feature2D::detectAndCompute;

    /**
     * @brief Set/get which ASV type detectAndCompute returns
     */
    void setASVType(const ASVType type) { asvType = type; }

    /**
     * @brief Get current ASV type
     */
    ASVType getASVType() const { return asvType; }

    /**
     * @brief Detects keypoints and computes ASV descriptors
     *
     * @param image Input image.
     * @param mask Optional mask.
     * @param keypoints Input/output keypoints. If useProvidedKeypoints is false
     *        or keypoints is empty, detection is performed internally.
     * @param descriptor Output Mat of ASV descriptors.
     * @param useProvidedKeypoints If true, detections is skipped and the
     *        provided keypoints are used.
     */
    virtual void detectAndCompute(InputArray image,
                                  InputArray mask,
                                  std::vector<KeyPoint>& keypoints,
                                  OutputArray descriptors,
                                  bool useProvidedKeypoints = false) CV_OVERRIDE;

    /**
     * @brief Computes ASV descriptors for given keypoints.
     *
     * @param image Input image.
     * @param keypoints Input keypoints.
     * @param realDescriptors Output real-valued ASV descriptors.
     * @param binaryDescriptors Output binary ASV descriptors.
     */
    virtual void compute(InputArray image,
                         const std::vector<KeyPoint>& keypoints,
                         OutputArray realDescriptors,
                         OutputArray binaryDescriptors);

  protected:
    /** @brief Underlying OpenCV Feature2D detector/descriptor. */
    Ptr<Feature2D> detector;
    /** @brief Number of scales sampled per keypoint. */
    int nScales;
    /** @brief Relativev scale factors used for sampling. */
    std::vector<double> scaleFactors;
    /** @brief Number of thresholds in the 1st stage voting.*/
    int nThreshold1;
    /** @brief Number of thresholds in the 2nd stage voting.*/
    int nThreshold2;
    /** @brief Length of the base descriptor returned by the underlying detector.*/
    int descriptorSize;
    /** @brief OpenCV type of the base descriptor. */
    int descriptorType;
    /** @brief Length of the binary ASV descriptor (descriptorSize * nThreshold2). */
    int binaryDescriptorSize;
    /** @brief Per-keypoint real-valued ASV descriptors. */
    std::vector<Mat> realDescriptors;
    /** @brief Per-keypoint binary ASV descriptors. */
    std::vector<Mat> binaryDescriptors;
    /** @brief Current ASV descriptor type returned by detectAndCompute(). */
    ASVType asvType = ASVType::Real;

    /**
     * @brief Extract descriptors at multiple scales for each keypoint
     *
     * @param image Input image.
     * @param keypoints Input keypoints in the detected scale.
     * @param multiScaleDescriptors Output: for each keypoint i and scale s,
     *        multiScaleDescriptors[i][s] stores the descriptor at that scale,
     *        or an empty Mat if extraction failed for that scale.
     */
    void extractMultiScaleDescriptors(const Mat& image,
                                      const std::vector<KeyPoint>& keypoints,
                                      std::vector<std::vector<Mat>>& multiScaleDescriptors);

    /**
     * @brief Compute real-valued ASV descriptors from multi-scale descriptors.
     *
     * @param multiScaleDescriptors Multi-scale descriptors for each keypoint.
     */
    void computeRealASV(const std::vector<std::vector<Mat>>& multiScaleDescriptors);

    /**
     * @brief Comptue 1st stage stability votes for a single keypoint.
     *
     * @param keypointDescriptors Descriptors for a single keypoint across scales.
     * @param votes Output: matrix of stability votes.
     */
    void computeFeatureASV(const std::vector<Mat>& keypointDescriptors, Mat& votes) const;

    /**
     * @brief Compute element-wise absolute difference between two descriptors.
     * @param desc1 First descriptor.
     * @param desc2 Second descriptor.
     * @param votes Output: column vector with one entry per descriptor dimension.
     */
    void computeAbsDiff(const Mat& desc1, const Mat& desc2, Mat& votes) const;

    /**
     * @brief Compute binary ASV descriptors from realDescriptors.
     *
     * Uses 2nd stage thresholding to generate a binary code for each keypoint.
     */
    void computeBinaryASV();
  };
} // namespace cv
#endif // ASV_HPP
