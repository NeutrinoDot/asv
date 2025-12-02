// unittest/asv_test.cpp

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

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
                              vector<KeyPoint>& kpts, Mat& desc)
{
  CV_Assert(!img.empty() && detector);
  Mat gray;
  if (img.channels() == 3) {
    cvtColor(img, gray, COLOR_BGR2GRAY);
  }
  else {
    gray = img;
  }
  detector->detectAndCompute(gray, noArray(), kpts, desc);
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
int main(int /*argc*/, char* /*argv*/[])
{
  try {
    // 1. Load input images (BGR)
    Mat img1 = LoadImageOrThrow("unittest/kittens1.jpg");
    Mat img2 = LoadImageOrThrow("unittest/kittens2.jpg");

    cout << "Loaded kittens1.jpg and kittens2.jpg\n";

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
    int nScales = 5;
    float scale_min = 0.7f;
    float scale_max = 1.4f;
    int nThreshold1 = 3;   // stage-1
    int nThreshold2 = 1;   // stage-2
    bool isInter = false;

    Ptr<ASV> asv = ASV::create(detectorType,
                               nScales,
                               scale_min,
                               scale_max,
                               nThreshold1,
                               nThreshold2,
                               isInter);

    cout << "Created ASV with detectorType=" << detectorType
      << " nScales=" << nScales
      << " nThreshold1=" << nThreshold1
      << " nThreshold2=" << nThreshold2
      << endl;

    // 5. Compute ASV descriptors (real + binary) for both images
    Mat desc1_real, desc1_bin;
    Mat desc2_real, desc2_bin;

    asv->compute(gray1, kpts1, desc1_real, desc1_bin);
    asv->compute(gray2, kpts2, desc2_real, desc2_bin);

    cout << "ASV real descriptors img1: " << desc1_real.rows
      << " x " << desc1_real.cols << " (type " << desc1_real.type() << ")\n";
    cout << "ASV real descriptors img2: " << desc2_real.rows
      << " x " << desc2_real.cols << " (type " << desc2_real.type() << ")\n";

    if (desc1_real.empty() || desc2_real.empty()) {
      cerr << "ASV descriptors are empty. Check ASV implementation.\n";
      return 1;
    }

    // 6. Match SIFT descriptors using BFMatcher + ratio test (NORM_L2 for CV_32F)
    vector<DMatch> good_sift;
    MatchWithRatioTest(NORM_L2, desc1_sift, desc2_sift, good_sift);

    cout << "Good SIFT matches after ratio test: " << good_sift.size() << '\n';

    // 6. Match ASV descriptors using BFMatcher + ratio test (NORM_L2 for CV_32F)
    vector<DMatch> good_asv;
    MatchWithRatioTest(NORM_L2, desc1_real, desc2_real, good_asv);

    cout << "Good ASV matches after ratio test: " << good_asv.size() << '\n';

    // 7. Draw matches and save montage

    Mat montage_sift;
    DrawMatchesAndSave(img1, img2,
                       kpts1, kpts2,
                       good_sift,
                       "SIFT Matches",
                       "output_SIFT.jpg",
                       montage_sift);
    Mat montage_asv;
    DrawMatchesAndSave(img1, img2,
                       kpts1, kpts2,
                       good_asv,
                       "ASV Matches",
                       "output_ASV.jpg",
                       montage_asv);

    cout << "Press any key to exit..." << endl;
    waitKey(0);
    destroyAllWindows();
    return 0;
  }
  catch (const exception& ex) {
    cerr << "Fatal error: " << ex.what() << '\n';
    return 1;
  }
}
