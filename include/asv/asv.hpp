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
		/** Constructor
		@param detector The feature detector used by ASV.
		@param nScales Number of scales to sample for each keypoint.
		@param scaleStep Step size between scales.
		@param nThreshold1 Number of thresholds for each bin for 1st-stage
		thresholding
		@param nThreshold2 Number of thresholds for 2nd-stage thresholding
		@param isInter Flag to interpolate features between scales
		*/
		ASV(const int detectorType, const int nScales, 
				const float scale_min, const float scale_max, 
				const double nThreshold1, const double nThreshold2, 
				const bool isInter);

		/** Factory */
		CV_WRAP static Ptr<ASV> create(const int detectorType   = 0, 
																	 const int nScales        = 5, 
																	 const float scale_min    = 0.7f, 
                                   const float scale_max    = 1.4f,
																	 const double nThreshold1 = 1.0, 
																	 const double nThreshold2 = 1.0, 
																	 const bool isInter       = false);

		/** Computes ASV descriptors
		 */
		virtual void compute(const InputArray image, 
												 const std::vector<KeyPoint>& keypoints, 
												 OutputArray realDescriptors, 
												 OutputArray binaryDescriptors);

	protected:
		Ptr<Feature2D> detector;
		int nScales;
    std::vector<double> scaleFactors;
		double nThreshold1;
		double nThreshold2;
		bool isInter;
		int descriptorSize;
		int descriptorType;
		std::vector<Mat> realDescriptors;
    std::vector<Mat> binaryDescriptors;

		// extract an array of descriptors at multiple scales per keypoint
		void extractMultiScaleDescriptors(const Mat& image, 
																			const std::vector<KeyPoint>& keypoints,
																			std::vector<std::vector<Mat>>& multiScaleDescriptors);

		// perform stability voting given multi-scale descriptors
		void computeRealASV(const std::vector<std::vector<Mat>>& multiScaleDescriptors);

		// calculate real stability vote of a multi-scale feature descriptor
		void computeFeatureASV(const std::vector<Mat>& keypointDescriptors, Mat& votes);

		// calculate stability votes between two descriptors
		void computeStabilityVote(const Mat& desc1, const Mat& desc2, Mat& votes) const;

		// convert real-valued descriptors to binary descriptor
		void computeBinaryASV();

    // compute scale factors
    void computeScaleFactors(const float scale_min, const float scale_max);

		int nChooseK(int n, int k);
	};
} // namespace cv
#endif // ASV_HPP