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
            double _nThreshold1,
            double _nThreshold2,
            bool _isInter)
        : detector(_detector),
        nScales(_nScales),
        scaleStep(_scaleStep),
        nThreshold1(_nThreshold1),
        nThreshold2(_nThreshold2),
        isInter(_isInter)
    {
    }

    Ptr<ASV> ASV::create(const Ptr<Feature2D>& detector,
                        int nScales,
                        double scaleStep,
                        double nThreshold1,
                        double nThreshold2,
                        bool isInter)
    {
        return makePtr<ASV>(detector, nScales, scaleStep, nThreshold1, nThreshold2, isInter);
    }

    // compute descriptors
    void ASV::compute(InputArray _image, std::vector<KeyPoint>& keypoints,
                    OutputArray _descriptors, OutputArray _binaryDescriptors)
    {
        CV_Assert(!detector.empty());
        
        Mat image = _image.getMat();

        if (keypoints.empty())
        {
            _descriptors.release();
            CV_Error( Error::StsBadArg, "No keypoints detected." );
        }

        // store multiscale descriptors as vector [keypoint][scale][Mat]
        int nKeypoints = (int)keypoints.size();
        std::vector<std::vector<Mat>> multiScaleDescriptors(nKeypoints);
        
        // extract descriptors at multiple scales around each keypoint
        for (int scaleIdx = 0; scaleIdx < nScales; scaleIdx++)
        {
            // create scaled keypoints copy
            double scaleFactor = std::pow(scaleStep, scaleIdx - nScales / 2.0);            
            std::vector<KeyPoint> scaledKeypoints = keypoints; // deep copy keypoints
            for (size_t i = 0; i < nKeypoints; i++)
            {
                scaledKeypoints[i].size *= static_cast<float>(scaleFactor);
            }

            // get descriptors at this scale
            Mat scaledDescriptor; // [keypoint][descriptor]
            detector->compute(image, scaledKeypoints, scaledDescriptor);

            if (nKeypoints > scaledDescriptor.rows) {
                std::cerr << "nKeypoints > scaledDescriptor.rows" << std::endl;
                exit(1);
            }

            // store descriptors for each keypoint
            for (int i = 0; i < nKeypoints; i++)
            {                
                multiScaleDescriptors[i].push_back(scaledDescriptor.row(i).clone());
            }
        }
        
        // compute stability voting on multiScaleDescriptors
    }
}
