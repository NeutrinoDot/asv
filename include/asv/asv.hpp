// include/asv/asv.hpp
#ifndef ASV_HPP
#define ASV_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv {

  /** 
   * @brief Class for computing descriptors using Accumulated Stability Voting (ASV).
   * 
   * ASV calculates decriptors at multiple scales and uses stability voting to
   * create a more robust descriptor. The ASV descriptor can work with any 
   * Feature2D detector/describer (SIFT, ORB, BRISK).
   * 
   * ASV algorithm is based on Yang, T.-Y., Lin, Y.-Y. & Chuang, Y.-Y.,
   * "Accumulated Stability Voting: A Robust Descriptor from Descriptors of
   * Multiple Scales," CVPR 2016.
   */
  class CV_EXPORTS_W ASV : public Feature2D {
  public:
    enum ASVType {
      Real,
      Binary
    };

    /** Constructor
     * @param detectorType The feature detector used by ASV.
     * @param nScales Number of scales to sample for each keypoint.
     * @param scale_min Minimum scale factor.
     * @param scale_max Maximum scale factor.
     * @param nThreshold1 Number of thresholds for 1st-stage thresholding
     * @param nThreshold2 Number of thresholds for 2nd-stage thresholding
     */
    ASV(const int detectorType, const int nScales,
        const float scale_min, const float scale_max,
        const int nThreshold1, const int nThreshold2);

    /** Factory */
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
     * @brief Detects keypoints with base detector and computes real ASV descriptors
     * 
     * 
     * 
     * @param image The input image.
     * @param mask The optional mask.
     * @param keypoints The detected keypoints.
     * @param descriptor The output ASV descriptors (real or binary based on asvType).
     * @param useProvidedKeypoints If true, the provided keypoints are used without running detection
     */
    virtual void detectAndCompute(InputArray image,
                                  InputArray mask,
                                  std::vector<KeyPoint>& keypoints,
                                  OutputArray descriptors,
                                  bool useProvidedKeypoints = false) CV_OVERRIDE;

    /**
     * @brief Computes ASV descriptors
     * @param image The input image.
     * @param keypoints The input keypoints.
     * @param realDescriptors The output real-valued ASV descriptors.
     * @param binaryDescriptors The output binary ASV descriptors.
     */
    virtual void compute(InputArray image,
                         const std::vector<KeyPoint>& keypoints,
                         OutputArray realDescriptors,
                         OutputArray binaryDescriptors);

  protected:
    Ptr<Feature2D> detector;
    int nScales;
    std::vector<double> scaleFactors;
    int nThreshold1;
    int nThreshold2;
    int descriptorSize;
    int descriptorType;
    int binaryDescriptorSize;
    std::vector<Mat> realDescriptors;
    std::vector<Mat> binaryDescriptors;
    ASVType asvType = ASVType::Real;

    /** 
     * @brief extract an array of descriptors at multiple scales per keypoint
     * @param image The input image.
     * @param keypoints The input keypoints.
     * @param multiScaleDescriptors Output array where multiScaleDescriptors[i][s] 
     * is the descriptor for keypoint i at scale s.
     */
    void extractMultiScaleDescriptors(const Mat& image,
                                      const std::vector<KeyPoint>& keypoints,
                                      std::vector<std::vector<Mat>>& multiScaleDescriptors);

    /**
     * @brief perform stability voting given multi-scale descriptors
     * @param multiScaleDescriptors The multi-scale descriptors for each keypoint.
     */
    void computeRealASV(const std::vector<std::vector<Mat>>& multiScaleDescriptors);

    /**
     * @brief calculate real stability vote of a multi-scale feature descriptor
     * @param keypointDescriptors The descriptors for a single keypoint across scales.
     * @param votes Output matrix of stability votes.
     */
    void computeFeatureASV(const std::vector<Mat>& keypointDescriptors, Mat& votes) const;

    /**
     * @brief calculate absolute difference between two descriptors across all dimensions
     * @param desc1 The first descriptor.
     * @param desc2 The second descriptor.
     * @param votes Output matrix of absolute differences.
     */
    void computeAbsDiff(const Mat& desc1, const Mat& desc2, Mat& votes) const;

    /**
     * @brief Generate binary descriptor from real-valued descriptor
     */
    void computeBinaryASV();
  };
} // namespace cv
#endif // ASV_HPP
