
// eval_main.cpp
// -------------------------------------------------------------------------------------------------
// Entry point for running the evaluation pipeline described in the slides.
//
// Pipeline:
//   1. Load image pairs + homographies (Oxford / HPatches).
//   2. For each pair:
//        a. Detect keypoints + compute descriptors with chosen method:
//             - SIFT (baseline)
//             - ASV-SIFT
//        b. Match descriptors using BFMatcher + (optional) ratio test.
//        c. Label matches as correct/incorrect via homography reprojection error.
//        d. Compute PR curve + AP for the pair.
//   3. Aggregate and compute mAP across all pairs.
//
// ASSUMPTIONS:
//   - Paths/specs for dataset pairs are supplied from some configuration source. For now, this
//     example uses a small hard-coded vector to illustrate usage.
//
#include <iostream>
#include <vector>

#include "dataset.h"
#include "descriptors.h"
#include "matching.h"

// TODO: Adding computing metrics" 

struct EvalConfig {
    DescriptorKind descriptorKind = DescriptorKind::SIFT;
    MatchingConfig matchingConfig;
};

int main(int argc, char** argv) {
    // If we want to setup more arguments, parse argv for:
    //   - dataset root folder
    //   - which descriptor (sift / asv)
    //   - epsilon, ratio threshold, etc.
    //
    // Here we set up a placeholder example config.

    EvalConfig cfg;
    cfg.descriptorKind = DescriptorKind::ASV_SIFT; // or DescriptorKind::SIFT
    cfg.matchingConfig.useRatioTest = true;
    cfg.matchingConfig.ratioThreshold = 0.8f; // Lowe's suggestion
    cfg.matchingConfig.epsilonPx = 3.0f;      // common default value

    // Example: hard-coded ImagePairSpec list.
    // In practice, we would load this from a file or generate from directory structure.
    std::vector<ImagePairSpec> specs = {
        // { id, pathImageA, pathImageB, pathHomography }
        // Example (dummy paths):
        // { "graf_1_2", "/path/to/graf1.png", "/path/to/graf2.png", "/path/to/H1to2.txt" }
    };

    if (specs.empty()) {
        std::cerr << "No image pair specs configured. Please populate 'specs'." << std::endl;
        return 1;
    }

    try {
        // 1. Load dataset
        DatasetLoader loader(specs);
        std::vector<ImagePair> pairs = loader.loadAll();

        // 2. Create descriptor extractor (SIFT or ASV-SIFT)
        auto extractor = createExtractor(cfg.descriptorKind);

        for (const auto& pair : pairs) {
            std::cout << "Evaluating pair: " << pair.id << std::endl;

            // 2a. Compute descriptors for image A
            DescriptorSet descA;
            extractor->detectAndCompute(pair.imgA, descA);

            // 2b. Compute descriptors for image B
            DescriptorSet descB;
            extractor->detectAndCompute(pair.imgB, descB);

            // 2c. Match descriptors
            std::vector<MatchWithLabel> matches =
                matchDescriptors(descA, descB, cfg.matchingConfig);

            // 2d. Label matches with homography-based correctness
            labelMatchesWithHomography(descA, descB, pair.H_AtoB,
                                       cfg.matchingConfig.epsilonPx,
                                       matches);

            //TODO: 2e. Compute PR + AP for this pair

        }

        //TODO: 3. Aggregate mAP

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Std exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
