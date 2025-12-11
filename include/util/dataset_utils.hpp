#pragma once

#include <string>
#include <vector>
#include "dataset/oxford_dataset.hpp"

/** 
 * @brief Discover image-homography pairs inside the Oxford dataset.
 * 
 * The Oxford dataset stores one reference image (image1.ppm) per sequence,
 * and several transformed images, each with an associated ground-truth
 * homography file H1toNp.
 * 
 * This function: 
 *    - Scans each subdirectory inside `datasetPath`.
 *    - Detects all homography filenames.
 *    - Builds an ImagePairSpec {id, img1, imgN, H1toNp} for each pair found.
 *    - Optionally applies a percentage filter to subsample the dataset.
 * 
 * @param datasetPath Root directory of the Oxford dataset.
 * @param percentage Fraction of discovered pairs to keep.
 * 
 * @return Vector of ImagePairSpec entries describing all usable pairs.
 */
std::vector<ImagePairSpec> discoverOxfordPairs(const std::string& datasetPath, float percentage = 1.0f);
