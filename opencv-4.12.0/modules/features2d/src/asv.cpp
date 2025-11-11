/** @brief Class for computing descriptors using Accumulated Stability Voting 
 (ASV) algorithm based on: Yang, T.-Y., Lin, Y.-Y. & Chuang, Y.-Y., 
 "Accumulated Stability Voting: A Robust Descriptor from Descriptors of 
 Multiple Scales," CVPR 2016.

ASV calculates descriptors at multiple scales and uses stability voting to 
create more better descriptors. The ASV descriptor can work with any Feature2D 
detector (SIFT, ORB, BRISK, ...)
*/

#include "precomp.hpp"
#include "opencv2/features2d.hpp"
#include <numeric>
#include <iostream>

namespace cv {

    // constructor
    ASV::ASV(const Ptr<Feature2D>& _detector,
            int _nScales,
            double _scaleStep,
            double _voteThreshold)
        : detector(_detector),
        nScales(_nScales),
        scaleStep(_scaleStep),
        voteThreshold(_voteThreshold)
    {
    }

    Ptr<ASV> ASV::create(const Ptr<Feature2D>& detector,
                        int nScales,
                        double scaleStep,
                        double votingThreshold)
    {
        return makePtr<ASV>(detector, nScales, scaleStep, votingThreshold);
    }

    // compute descriptors
    void ASV::compute(InputArray _image,
                    std::vector<KeyPoint>& keypoints,
                    OutputArray _descriptors)
    {
        CV_Assert(!detector.empty());
        
        Mat image = _image.getMat();

        if (keypoints.empty())
        {
            _descriptors.release();
            CV_Error( Error::StsBadArg, "No keypoints detected." );
        }

        // Store descriptors at each scale for each keypoint
        int nKeypoints = (int)keypoints.size();
        std::vector<std::vector<Mat>> multiScaleDescriptors(nKeypoints);
        
        // Extract descriptors at multiple scales around each keypoint
        for (int scaleIdx = 0; scaleIdx < nScales; scaleIdx++)
        {
            // get and store descriptors at a specific scale
        }
        
        // Apply stability voting
    }
}
