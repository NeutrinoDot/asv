// dataset.h
// -------------------------------------------------------------------------------------------------
// Defines data structures and loader for Oxford / HPatches-style image pairs with ground-truth
// homography.
//
// INPUT / FORMAT ASSUMPTIONS:
// - Each "pair" consists of:
//     * Image A path (grayscale or color; we convert to grayscale for descriptors)
//     * Image B path
//     * 3x3 homography text file H_AtoB:
//           h11 h12 h13
//           h21 h22 h23
//           h31 h32 h33
//   This is consistent with HPatches/Oxford typical homography format (row-major).
//
// OUTPUT STRUCTURE:
// - A vector of ImagePair structures, each with:
//     id       : unique identifier (sequence_name or pair name)
//     imgA     : cv::Mat (CV_8U, grayscale)
//     imgB     : cv::Mat (CV_8U, grayscale)
//     H_AtoB   : cv::Mat (3x3, CV_64F) mapping points in A to B (homogeneous coordinates)
//
// ASSUMPTIONS:
// - Paths and file naming conventions are provided by higher-level config; this loader does
//   not enforce a single canonical dataset layout. Instead, we accept lists of explicit paths.
// - Validation of "correctness" of homography (e.g., whether it really maps A→B) is not done
//   here; we trust dataset ground truth.
//
#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct ImagePair {
    // Human-readable identifier, e.g., "graf_1_2" or "sequence_01_view_1_2"
    std::string id;

    // Grayscale images used for keypoint detection + descriptor computation.
    // Type: CV_8U, single-channel.
    cv::Mat imgA;
    cv::Mat imgB;

    // Ground-truth homography mapping points from image A to image B.
    // 3x3 matrix, CV_64F.
    cv::Mat H_AtoB;
};

struct ImagePairSpec {
    // Paths on disk to load one pair and its homography.
    std::string id;
    std::string pathImageA;
    std::string pathImageB;
    std::string pathHomography; // text file with 3x3 matrix
};

// Lightweight loader that accepts explicit specs rather than assuming a specific folder layout.
class DatasetLoader {
public:
    // Constructor
    // specs: list of (imgA, imgB, H) triplets.
    explicit DatasetLoader(const std::vector<ImagePairSpec>& specs);

    // Load all pairs into memory (images converted to grayscale, homography parsed).
    // Throws cv::Exception or std::runtime_error on IO errors.
    std::vector<ImagePair> loadAll() const;

private:
    std::vector<ImagePairSpec> specs_;

    // Helper to load one homography 3x3 from a text file.
    // Expected format: 9 numbers (float or double), typically row-major.
    cv::Mat loadHomographyFromTxt(const std::string& path) const;
};

