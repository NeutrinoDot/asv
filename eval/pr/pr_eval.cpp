// unittest/asv_test.cpp

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include "asv/asv.hpp"

using namespace cv;
using namespace std;

// ------------------------ Constants ------------------------
constexpr int   SCALE_VIEW = 8;          // divisor to shrink large windows
constexpr float RATIO_TEST_THRESH = 0.75f;      // Lowe ratio test threshold
constexpr auto  DRAW_FLAGS = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;

// ------------------------ Utility helpers ------------------------

/**
 * @brief Display an image in a normal, resizable window.
 * @pre  img is a valid Mat, return if empty img.
 * @post A window is shown; does not block.
 */
static void ShowImageScaled(const string& windowTitle, const Mat& img)
{
  if (img.empty()) return;
  namedWindow(windowTitle, WINDOW_NORMAL);
  resizeWindow(windowTitle,
               max(1, img.cols / SCALE_VIEW),
               max(1, img.rows / SCALE_VIEW));
  imshow(windowTitle, img);
}

/**
 * @brief Write image to disk; logs result to stdout/stderr.
 * @pre  img is a valid Mat.
 * @post Returns true on success; false on failure.
 */
static bool SaveImage(const string& path, const Mat& img)
{
  if (!imwrite(path, img)) {
    cerr << "Warning: failed to write output to " << path << '\n';
    return false;
  }
  cout << "Saved: " << path << '\n';
  return true;
}

/**
 * @brief Load image (BGR). Throws if load fails.
 * @pre  File exists and is a readable image format for OpenCV.
 * @post Returns a non-empty cv::Mat or throws.
 */
static Mat LoadImageOrThrow(const string& path)
{
  Mat img = imread(path);
  if (img.empty()) {
    throw runtime_error("Could not load image: " + path);
  }
  return img;
}

/**
 * @brief Detect keypoints (only) with a given detector, grayscale conversion inside.
 *
 * @param detector  (in)  A Feature2D pointer.
 * @param img       (in)  Input BGR image.
 * @param kpts      (out) Detected keypoints.
 *
 * @pre  detector is non-null; img is non-empty.
 * @post kpts.size() >= 0
 */
static void DetectAndDescribe(const Ptr<Feature2D>& detector, const Mat& img,
                              vector<KeyPoint>& kpts, Mat& desc,
                              const bool useProvidedKeypoints = false)
{
  CV_Assert(!img.empty() && detector);
  Mat gray;
  if (img.channels() == 3) {
    cvtColor(img, gray, COLOR_BGR2GRAY);
  }
  else {
    gray = img;
  }
  detector->detectAndCompute(gray, noArray(), kpts, desc, useProvidedKeypoints);
}

/**
 * @brief Match descriptors using BFMatcher with Lowe ratio test.
 *
 * @param normType  (in)  NORM_L2 for float; NORM_HAMMING for binary.
 * @param desc1     (in)  Descriptors from image 1.
 * @param desc2     (in)  Descriptors from image 2.
 * @param good      (out) Filtered good matches (after ratio test).
 *
 * @pre  desc1/desc2 are valid descriptor Mats (may be empty).
 * @post good contains 0..N matches that passed the ratio test.
 */
static void MatchWithRatioTest(int normType,
                               const Mat& desc1,
                               const Mat& desc2,
                               vector<DMatch>& good)
{
  good.clear();
  if (desc1.empty() || desc2.empty()) return;

  Ptr<BFMatcher> matcher = BFMatcher::create(normType);
  vector<vector<DMatch>> knn;
  matcher->knnMatch(desc1, desc2, knn, 2);

  for (const auto& nbrs : knn) {
    if (nbrs.size() == 2 &&
        nbrs[0].distance < RATIO_TEST_THRESH * nbrs[1].distance)
    {
      good.push_back(nbrs[0]);
    }
  }
}

// ------------------------ Precision-Recall Curve ------------------------

struct PRCurve {
    vector<double> recalls;
    vector<double> precisions;
    double ap;  // Average precision
};

Mat loadHomography(const string& filepath) {
    Mat H = Mat::eye(3, 3, CV_64F);
    ifstream file(filepath);
    if (!file.is_open()) {
        throw runtime_error("Cannot open " + filepath);
    }
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            file >> H.at<double>(i, j);
        }
    }
    return H;
}

vector<bool> verifyMatches(const vector<KeyPoint>& kpts1,
                           const vector<KeyPoint>& kpts2,
                           const vector<DMatch>& matches,
                           const Mat& H,
                           double threshold = 5.0) {
    vector<bool> is_correct(matches.size());
    
    for (size_t i = 0; i < matches.size(); i++) {
        Point2f pt1 = kpts1[matches[i].queryIdx].pt;
        Point2f pt2 = kpts2[matches[i].trainIdx].pt;
        
        // Project pt1 to img2 coordinates
        vector<Point2f> src = {pt1};
        vector<Point2f> dst;
        perspectiveTransform(src, dst, H);
        
        // Compute reprojection error
        double error = norm(dst[0] - pt2);
        is_correct[i] = (error < threshold);
    }
    
    return is_correct;
}

PRCurve computePRCurve(const vector<DMatch>& matches,
                       const vector<bool>& is_correct) {
    PRCurve curve;
    
    // Sort matches by distance (ascending)
    vector<size_t> indices(matches.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
         [&matches](size_t a, size_t b) {
             return matches[a].distance < matches[b].distance;
         });
    
    int total_correct = count(is_correct.begin(), is_correct.end(), true);
    if (total_correct == 0) {
        curve.ap = 0.0;
        return curve;
    }
    
    // Compute precision/recall at each rank
    int true_positives = 0;
    for (size_t k = 0; k < indices.size(); k++) {
        if (is_correct[indices[k]]) {
            true_positives++;
        }
        
        double precision = (double)true_positives / (k + 1);
        double recall = (double)true_positives / total_correct;
        
        curve.precisions.push_back(precision);
        curve.recalls.push_back(recall);
    }
    
    // Compute 11-point interpolated AP
    curve.ap = 0.0;
    for (double r = 0.0; r <= 1.0; r += 0.1) {
        double max_p = 0.0;
        for (size_t i = 0; i < curve.recalls.size(); i++) {
            if (curve.recalls[i] >= r) {
                max_p = max(max_p, curve.precisions[i]);
            }
        }
        curve.ap += max_p;
    }
    curve.ap /= 11.0;
    
    return curve;
}

void savePRCurve(const PRCurve& curve, const string& filename) {
    ofstream out(filename);
    out << "Recall Precision\n";
    out << "# AP = " << curve.ap << "\n";
    
    size_t n = curve.recalls.size();
    if (n == 0) {
        out.close();
        return;
    }
    
    // Sample 15 points evenly distributed
    const int target_points = 15;
    if (n <= target_points) {
        // If we have fewer than target points, output all
        for (size_t i = 0; i < n; i++) {
            out << curve.recalls[i] << " " << curve.precisions[i] << "\n";
        }
    } else {
        // Sample evenly across the range
        for (int i = 0; i < target_points; i++) {
            size_t idx = (i * (n - 1)) / (target_points - 1);
            out << curve.recalls[idx] << " " << curve.precisions[idx] << "\n";
        }
    }
    out.close();
}

// ------------------------ Draw Matches ------------------------

/**
 * @brief Compose a match visualization image, display and save it.
 *
 * @param img1         (in)  Image 1 (BGR).
 * @param img2         (in)  Image 2 (BGR).
 * @param kpts1        (in)  Keypoints for img1.
 * @param kpts2        (in)  Keypoints for img2.
 * @param matches      (in)  Good matches to draw.
 * @param title        (in)  Window title (for display).
 * @param outPath      (in)  If non-empty, save montage to this path.
 * @param outMontage   (out) The composed montage image.
 *
 * @pre  Images and keypoints/vectors correspond to each other.
 * @post outMontage is set; image displayed and optionally saved.
 */
static void DrawMatchesAndSave(const Mat& img1, const Mat& img2,
                               const vector<KeyPoint>& kpts1,
                               const vector<KeyPoint>& kpts2,
                               const vector<DMatch>& matches,
                               const string& title,
                               const string& outPath,
                               Mat& outMontage)
{
  drawMatches(img1, kpts1, img2, kpts2,
              matches,
              outMontage,
              Scalar::all(-1),
              Scalar::all(-1),
              std::vector<char>(),
              DRAW_FLAGS);

  ShowImageScaled(title, outMontage);

  if (!outPath.empty()) {
    SaveImage(outPath, outMontage);
  }
}

// ------------------------------ main ------------------------------

/**
 * @brief Main: Load images, detect SIFT keypoints, compute ASV descriptors,
 *        match & visualize.
 *
 * @pre  "kittens1.jpg" and "kittens2.jpg" exist in working directory.
 * @post "output_ASV.jpg" is written; window shown until key press.
 */
int main(int argc, char* argv[])
{
  try {
    // Parse command line arguments
    string img1_path = "data/Oxford_dataset/bark/img1.ppm";
    string img2_path = "data/Oxford_dataset/bark/img6.ppm";
    string homography_path = "data/Oxford_dataset/bark/H1to6p";
    
    if (argc >= 4) {
      img1_path = argv[1];
      img2_path = argv[2];
      homography_path = argv[3];
      cout << "Using provided paths:\n";
      cout << "  Image 1: " << img1_path << "\n";
      cout << "  Image 2: " << img2_path << "\n";
      cout << "  Homography: " << homography_path << "\n";
    } else {
      cout << "Usage: " << argv[0] << " <img1> <img2> <homography>\n";
      cout << "Using default: bark img1 to img6\n";
    }
    
    // 1. Load input images (BGR)
    Mat img1 = LoadImageOrThrow(img1_path);
    Mat img2 = LoadImageOrThrow(img2_path);

    cout << "Loaded img1.ppm and img6.ppm\n";

    // 2. Detect keypoints with SIFT (matches ASV detectorType=0)
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1_sift, desc2_sift;

    DetectAndDescribe(sift, img1, kpts1, desc1_sift);
    DetectAndDescribe(sift, img2, kpts2, desc2_sift);

    cout << "Image 1 keypoints: " << kpts1.size() << '\n';
    cout << "Image 2 keypoints: " << kpts2.size() << '\n';

    if (kpts1.empty() || kpts2.empty()) {
      cerr << "Not enough keypoints detected in one or both images.\n";
      return 1;
    }

    // 3. Prepare grayscale copies for ASV::compute
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // 4. Create ASV descriptor (SIFT baseline)
    int detectorType = 0;     // 0=SIFT, 1=ORB, 2=BRISK
    int nScales = 10;
    float scale_min = 1.0f / 6.0f;
    float scale_max = 3.0f;
    int nThreshold1 = 3;   // stage-1
    int nThreshold2 = 3;   // stage-2

    Ptr<ASV> asv = ASV::create(detectorType,
                               nScales,
                               scale_min,
                               scale_max,
                               nThreshold1,
                               nThreshold2);

    cout << "Created ASV with detectorType=" << detectorType
      << " nScales=" << nScales
      << " nThreshold1=" << nThreshold1
      << " nThreshold2=" << nThreshold2
      << endl;

    // 5. Compute ASV descriptors (real + binary) for both images
    Mat desc1_real, desc1_bin;
    Mat desc2_real, desc2_bin;
    kpts1.clear();
    kpts2.clear();

    cout << "desc1_sift size: " << desc1_sift.size() << endl;
    cout << "desc2_sift size: " << desc2_sift.size() << endl;

    asv->setASVType(ASV::ASVType::Real);
    DetectAndDescribe(asv, img1, kpts1, desc1_real, true);
    cout << "ASV real descriptors img1: " << desc1_real.rows
      << " x " << desc1_real.cols << " (type " << desc1_real.type() << ")\n";
    DetectAndDescribe(asv, img2, kpts2, desc2_real, true);
    cout << "ASV real descriptors img2: " << desc2_real.rows
      << " x " << desc2_real.cols << " (type " << desc2_real.type() << ")\n";
    cout << "desc1_real size: " << desc1_real.size() << endl;
    cout << "desc2_real size: " << desc2_real.size() << endl;

    asv->setASVType(ASV::ASVType::Binary);
    DetectAndDescribe(asv, img1, kpts1, desc1_bin, true);
    cout << "ASV binary descriptors img1: " << desc1_bin.rows
      << " x " << desc1_bin.cols << " (type " << desc1_bin.type() << ")\n";
    DetectAndDescribe(asv, img2, kpts2, desc2_bin, true);
    cout << "ASV binary descriptors img2: " << desc2_bin.rows
      << " x " << desc2_bin.cols << " (type " << desc2_bin.type() << ")\n";
    cout << "desc1_bin size: " << desc1_bin.size() << endl;
    cout << "desc2_bin size: " << desc2_bin.size() << endl;

    if (desc1_real.empty() || desc2_real.empty()) {
      cerr << "ASV real descriptors are empty. Check ASV implementation.\n";
      return 1;
    }
    if (desc1_bin.empty() || desc2_bin.empty()) {
      cerr << "ASV binary descriptors are empty. Check ASV implementation.\n";
      return 1;
    }

    // 6. Match SIFT descriptors using BFMatcher + ratio test (NORM_L2 for CV_32F)
    vector<DMatch> good_sift;
    MatchWithRatioTest(NORM_L2, desc1_sift, desc2_sift, good_sift);

    cout << "Good SIFT matches after ratio test: " << good_sift.size() << '\n';

    // 6. Match ASV descriptors using BFMatcher + ratio test (NORM_L2 for CV_32F)
    vector<DMatch> good_asv_real;
    MatchWithRatioTest(NORM_L2, desc1_real, desc2_real, good_asv_real);

    cout << "Good real ASV matches after ratio test: " << good_asv_real.size() << '\n';

    vector<DMatch> good_asv_bin;
    MatchWithRatioTest(NORM_HAMMING, desc1_bin, desc2_bin, good_asv_bin);

    cout << "Good binary real ASV matches after ratio test: " << good_asv_bin.size() << '\n';

    // 8. Load ground-truth homography
    Mat H = loadHomography(homography_path);

    // Extract category name from path
    string category = "unknown";
    size_t last_slash = img1_path.find_last_of("/\\");
    if (last_slash != string::npos) {
      size_t second_last_slash = img1_path.find_last_of("/\\", last_slash - 1);
      if (second_last_slash != string::npos) {
        category = img1_path.substr(second_last_slash + 1, last_slash - second_last_slash - 1);
      }
    }
    cout << "Category: " << category << endl;

    vector<bool> is_sift_correct = verifyMatches(kpts1, kpts2, good_sift, H);
    PRCurve sift_curve = computePRCurve(good_sift, is_sift_correct);
    cout << "SIFT AP = " << sift_curve.ap << endl;
    savePRCurve(sift_curve, "eval/pr/output/pr_" + category + "_sift.txt");

    vector<bool> is_asv_real_correct = verifyMatches(kpts1, kpts2, good_asv_real, H);
    PRCurve asv_real_curve = computePRCurve(good_asv_real, is_asv_real_correct);
    cout << "ASV Real AP = " << asv_real_curve.ap << endl;
    savePRCurve(asv_real_curve, "eval/pr/output/pr_" + category + "_asv_real.txt");

    vector<bool> is_asv_bin_correct = verifyMatches(kpts1, kpts2, good_asv_bin, H);
    PRCurve asv_bin_curve = computePRCurve(good_asv_bin, is_asv_bin_correct);
    cout << "ASV Binary AP = " << asv_bin_curve.ap << endl;
    savePRCurve(asv_bin_curve, "eval/pr/output/pr_" + category + "_asv_bin.txt");
    
    // 7. Draw matches and save montage (commented out for faster execution)
    /*
    Mat montage_sift;
    DrawMatchesAndSave(img1, img2,
                       kpts1, kpts2,
                       good_sift,
                       "SIFT Matches",
                       "eval/pr/output/output_" + category + "_SIFT.jpg",
                       montage_sift);
    Mat montage_asv_real;
    DrawMatchesAndSave(img1, img2,
                       kpts1, kpts2,
                       good_asv_real,
                       "ASV Real Matches",
                       "eval/pr/output/output_" + category + "_ASV_Real.jpg",
                       montage_asv_real);
    Mat montage_asv_bin;
    DrawMatchesAndSave(img1, img2,
                       kpts1, kpts2,
                       good_asv_bin,
                       "ASV Binary Matches",
                       "eval/pr/output/output_" + category + "_ASV_Binary.jpg",
                       montage_asv_bin);

    cout << "Press any key to exit..." << endl;
    waitKey(0);
    destroyAllWindows();
    */
    return 0;
  }
  catch (const exception& ex) {
    cerr << "Fatal error: " << ex.what() << '\n';
    return 1;
  }
}
