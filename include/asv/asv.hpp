// include/asv/asv.hpp
#ifndef ASV_HPP
#define ASV_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv {

  /** @brief Class for computing descriptors using Accumulated Stability Voting
   (ASV) algorithm based on: Yang, T.-Y., Lin, Y.-Y. & Chuang, Y.-Y.,
   "Accumulated Stability Voting: A Robust Descriptor from Descriptors of
   Multiple Scales," CVPR 2016.

   ASV calculates decriptors at multiple scales and uses stability voting to
   create a more robust descriptor. The ASV descriptor can work with any Feature2D
   detector/describer (SIFT, ORB, BRISK, etc.).
  */
  class CV_EXPORTS_W ASV : public Feature2D {
  public:
    enum ASVType {
      Real,
      Binary
    };

    /** Constructor
    @param detectorType The feature detector used by ASV.
    @param nScales Number of scales to sample for each keypoint.
    @param scale_min Minimum scale factor.
    @param scale_max Maximum scale factor.
    @param nThreshold1 Number of thresholds for 1st-stage thresholding
    @param nThreshold2 Number of thresholds for 2nd-stage thresholding
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

    // Set/get which ASV type detectAndCompute returns
    void setASVType(const ASVType type) { asvType = type; }
    ASVType getASVType() const { return asvType; }

    /** Detects keypoints with base detector and computes real ASV descriptors
    */
    virtual void detectAndCompute(InputArray image,
                                  InputArray mask,
                                  std::vector<KeyPoint>& keypoints,
                                  OutputArray descriptors,
                                  bool useProvidedKeypoints = false) CV_OVERRIDE;

    /** Computes ASV descriptors
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

    // extract an array of descriptors at multiple scales per keypoint
    void extractMultiScaleDescriptors(const Mat& image,
                                      const std::vector<KeyPoint>& keypoints,
                                      std::vector<std::vector<Mat>>& multiScaleDescriptors);

    // perform stability voting given multi-scale descriptors
    void computeRealASV(const std::vector<std::vector<Mat>>& multiScaleDescriptors);

    // calculate real stability vote of a multi-scale feature descriptor
    void computeFeatureASV(const std::vector<Mat>& keypointDescriptors, Mat& votes) const;

    // calculate absolute difference between two descriptors across all dimensions
    void computeAbsDiff(const Mat& desc1, const Mat& desc2, Mat& votes) const;

    // convert real-valued descriptors to binary descriptor
    void computeBinaryASV();
  };
} // namespace cv
#endif // ASV_HPP