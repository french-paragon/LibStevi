#include <QtTest/QtTest>

#include "sparseMatching/cornerDetectors.h"
#include "sparseMatching/nonLocalMaximumPointSelection.h"
#include "sparseMatching/pointsOrientation.h"
#include "sparseMatching/pointsDescriptors.h"

#include "correlation/matching_costs.h"

#include <iostream>
#include <random>
#include <limits>

using namespace StereoVision::SparseMatching;

Multidim::Array<float,2> generateSampleProblem(std::default_random_engine & re, int radius = 3) {

    int s = 2*radius+1;

    float max = -std::numeric_limits<float>::infinity();


    std::uniform_real_distribution<float> dist(-1, 1);
    Multidim::Array<float,2> ret(s,s);

    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            float val = dist(re);
            ret.atUnchecked(i,j) = val;

            if (max < val) {
                max = val;
            }
        }
    }

    ret.atUnchecked(radius, radius) = max+1;

    return ret;
}

Multidim::Array<float,3> generateMultiProblem(std::default_random_engine & re, int radius = 3) {

    int s = 2*radius+1;
    int c = 3;

    float max = -std::numeric_limits<float>::infinity();


    std::uniform_real_distribution<float> dist(-1, 1);
    Multidim::Array<float,3> ret(s,s,c);

    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            for (int k = 0; k < c; k++) {
                float val = dist(re);
                ret.atUnchecked(i,j,k) = val;

                if (max < val) {
                    max = val;
                }
            }
        }
    }

    for (int k = 0; k < c; k++) {
        ret.atUnchecked(radius, radius, k) = max+1+k;
    }

    return ret;
}

std::tuple<Multidim::Array<float,2>, std::array<std::array<int,2>,4>> generateSquare(int width) {

    int squareWidth = 2*width;

    int b1 = width/2;
    int b2 = b1+width;

    std::array<std::array<int,2>,4> coords = {std::array<int,2>({b1,b1}),
                                              std::array<int,2>({b1,b2}),
                                              std::array<int,2>({b2,b1}),
                                              std::array<int,2>({b2,b2})};

    Multidim::Array<float,2> ret(squareWidth, squareWidth);

    for (int i = 0; i < squareWidth; i++) {
        for (int j = 0; j < squareWidth; j++) {

            bool iInRange = i >= b1 and i <= b2;
            bool jInRange = j >= b1 and j <= b2;

            bool iOnRange = i == b1 or i == b2;
            bool jOnRange = j == b1 or j == b2;

            float color = 0;

            if (iOnRange and jInRange) {
                color = 0.5;
            } else if (jOnRange and iInRange) {
                color = 0.5;
            } else if (iInRange and jInRange) {
                color = 1;
            }

            ret.atUnchecked(i,j) = color;
        }
    }

    return std::make_tuple(ret, coords);

}

class TestSparseMatchingUtils: public QObject{

    Q_OBJECT
private Q_SLOTS:

    void initTestCase();

    void testHarrisCornerDetector();

    void testMaskedHarrisCornerMMat();

    void testFastCornerDetector();

    void testNonMaximumPointSelection();

    void testIntensityOrientedCoordinates();

    void testBriefDescriptor();


private:
    std::default_random_engine re;
};


void TestSparseMatchingUtils::initTestCase() {
    srand((unsigned int) time(nullptr));
    std::random_device rd;
    re.seed(rd());
}

void TestSparseMatchingUtils::testHarrisCornerDetector() {

    constexpr int squareWidth = 30;
    constexpr int lp_radius = 2;
    constexpr int nm_radius = 4;

    auto [img, gt_points] = generateSquare(squareWidth);

    Multidim::Array<float, 2> score = HarrisCornerScore(Multidim::Array<float, 2, Multidim::ConstView>(img), lp_radius);

    std::vector<std::array<float, 2>> results = nonLocalMaximumPointSelection(Multidim::Array<float, 2, Multidim::ConstView>(score), nm_radius, 0.f);

    QCOMPARE(results.size(), gt_points.size());

    for (int i = 0; i < results.size(); i++) {
        float minDist = std::numeric_limits<float>::infinity();

        for (int j = 0; j < gt_points.size(); j++) {

            float d1 = gt_points[j][0] - results[i][0];
            float d2 = gt_points[j][1] - results[i][1];
            float dist = sqrt(d1*d1 + d2*d2);

            if (dist < minDist) {
                minDist = dist;
            }
        }

        QVERIFY2(minDist < sqrt(2), qPrintable(QString("Min distance too large for pt %1").arg(i)));
    }

    constexpr int w_radius = 0;

    Multidim::Array<float, 2> score_w = windowedHarrisCornerScore(Multidim::Array<float, 2, Multidim::ConstView>(img), w_radius, lp_radius);

    results = nonLocalMaximumPointSelection(Multidim::Array<float, 2, Multidim::ConstView>(score_w), nm_radius, 0.f);

    QCOMPARE(results.size(), gt_points.size());

    for (int i = 0; i < results.size(); i++) {
        float minDist = std::numeric_limits<float>::infinity();

        for (int j = 0; j < gt_points.size(); j++) {
            float d1 = gt_points[j][0] - results[i][0];
            float d2 = gt_points[j][1] - results[i][1];
            float dist = sqrt(d1*d1 + d2*d2);

            if (dist < minDist) {
                minDist = dist;
            }
        }

        QVERIFY2(minDist < sqrt(2), qPrintable(QString("Min distance too large for pt %1").arg(i)));
    }

}



void TestSparseMatchingUtils::testMaskedHarrisCornerMMat() {

    constexpr int size = 30;
    constexpr int lp_radius = 2;
    constexpr int nm_radius = 4;

    Multidim::Array<float,2> img(size, size);
    Multidim::Array<bool,2> mask(size, size);

    std::uniform_real_distribution<float> imgDist(-1,1);
    std::uniform_int_distribution<uint8_t> maskDist(0,1);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

            img.atUnchecked(i,j) = imgDist(re);
            mask.atUnchecked(i,j) = maskDist(re);

        }
    }

    Multidim::Array<float, 3> base = HarrisCornerMMat(Multidim::Array<float, 2, Multidim::ConstView>(img),
                                                      lp_radius);
    Multidim::Array<float, 3> masked = maskedHarrisCornerMMat(Multidim::Array<float, 2, Multidim::ConstView>(img),
                                                              Multidim::Array<bool, 2, Multidim::ConstView>(mask),
                                                              lp_radius);

    QCOMPARE(masked.shape(), base.shape());

    for (int i = lp_radius+1; i < size-lp_radius-1; i++) {
        for (int j = lp_radius+1; j < size-lp_radius-1; j++) {

            if (mask.valueUnchecked(i,j)) {

                for (int c = 0; c < 3; c++) {
                    float tol = 1e-4;
                    float delta = std::abs(base.valueUnchecked(i,j, c) - masked.valueUnchecked(i,j, c));

                    QVERIFY2(delta < tol, qPrintable(QString("Different M matrix estimates results, error = %1, i = %2, j = %3.").arg(delta).arg(i).arg(j)));
                }
            } else {

                for (int c = 0; c < 3; c++) {
                    QCOMPARE(masked.valueUnchecked(i,j,c), 0);
                }
            }

        }
    }

}

void TestSparseMatchingUtils::testFastCornerDetector() {

    constexpr int squareWidth = 30;
    constexpr int thresh = 10;
    constexpr float fastThres = 0.05;
    constexpr int lp_radius = 2;
    constexpr int w_radius = 1; //windows radius need to be 0, else the corner position of windowed harris corner detector can be unpredictable.
    constexpr int nm_radius = 4;

    auto [img, gt_points] = generateSquare(squareWidth);

    Multidim::Array<uint8_t, 2> fast_score = FASTCornerDetection(Multidim::Array<float, 2, Multidim::ConstView>(img), fastThres);

    auto m_shape = fast_score.shape();
    Multidim::Array<bool, 2> mask(m_shape);

    for(int i = 0; i < m_shape[0]; i++) {
        for(int j = 0; j < m_shape[1]; j++) {
            mask.atUnchecked(i,j) = fast_score.valueUnchecked(i,j) >= thresh;
        }
    }

    Multidim::Array<float, 2> score = maskedWindowedHarrisCornerScore(Multidim::Array<float, 2, Multidim::ConstView>(img),
                                                                      Multidim::Array<bool, 2, Multidim::ConstView>(mask),
                                                                      w_radius);

    std::vector<std::array<float, 2>> results = nonLocalMaximumPointSelection(Multidim::Array<float, 2, Multidim::ConstView>(score), nm_radius, 0.f);

    QCOMPARE(results.size(), gt_points.size());

    for (int i = 0; i < results.size(); i++) {
        float minDist = std::numeric_limits<float>::infinity();

        for (int j = 0; j < gt_points.size(); j++) {
            float d1 = gt_points[j][0] - results[i][0];
            float d2 = gt_points[j][1] - results[i][1];
            float dist = sqrt(d1*d1 + d2*d2);

            if (dist < minDist) {
                minDist = dist;
            }
        }

        QVERIFY2(minDist < sqrt(2), qPrintable(QString("Min distance too large for pt %1").arg(i)));
    }

}

void TestSparseMatchingUtils::testNonMaximumPointSelection() {

    constexpr int radius = 3;

    Multidim::Array<float,2> test = generateSampleProblem(re, radius);

    std::vector<std::array<float, 2>> result = nonLocalMaximumPointSelection(Multidim::Array<float, 2, Multidim::ConstView>(test), radius, 0.f);

    QCOMPARE(result.size(), 1);
    QCOMPARE(result[0][0], radius);
    QCOMPARE(result[0][1], radius);

}

void TestSparseMatchingUtils::testIntensityOrientedCoordinates() {

    constexpr int squareWidth = 30;
    constexpr int searchRadius = 3;

    auto [img, gt_points] = generateSquare(squareWidth);

    std::vector<std::array<int, 2>> points(gt_points.begin(), gt_points.end());

    std::vector<orientedCoordinate<2>> orientedPoints = intensityOrientedCoordinates<false>(points, img, searchRadius);

    QCOMPARE(orientedPoints.size(), points.size());

    for (orientedCoordinate<2> & ocoord : orientedPoints) {
        float c1 = (ocoord[0] < squareWidth) ? 1 : -1;
        float c2 = (ocoord[1] < squareWidth) ? 1 : -1;

        c1 /= std::sqrt(2);
        c2 /= std::sqrt(2);

        QCOMPARE(ocoord.main_dir[0], c1);
        QCOMPARE(ocoord.main_dir[1], c2);
    }

}

void TestSparseMatchingUtils::testBriefDescriptor() {

    constexpr int squareWidth = 30;
    constexpr int searchRadius = 3;

    constexpr int windowsRadius = searchRadius + 2;
    constexpr int nSamples = 50;
    constexpr int tol = 5;

    int nNWords = nSamples/32;
    if (nSamples%32 != 0) {
        nNWords += 1;
    }

    auto tuple = generateSquare(squareWidth);
    Multidim::Array<float,2>& img = std::get<0>(tuple);
    std::array<std::array<int,2>,4>& gt_points = std::get<1>(tuple);

    std::vector<std::array<int, 2>> points(gt_points.begin(), gt_points.end());

    std::vector<orientedCoordinate<2>> orientedPoints = intensityOrientedCoordinates<false>(points, img, searchRadius);

    std::vector<ComparisonPair<2>> comparisonPairs = generateRandomComparisonPairs<2>(nSamples, windowsRadius);

    std::vector<pointFeatures<2, std::vector<uint32_t>>> descriptors = BriefDescriptor<false>(orientedPoints, img, comparisonPairs);

    QCOMPARE(descriptors.size(), orientedPoints.size());

    //feature vectors are supposed to be the same.
    std::vector<uint32_t> vec0 = descriptors[0].features;

    QCOMPARE(vec0.size(), nNWords);

    for (int i = 1; i < 4; i++) {
        std::vector<uint32_t> veci = descriptors[i].features;

        QCOMPARE(veci.size(), vec0.size());

        int unMatched = StereoVision::Correlation::hammingDistance(vec0, veci);

        QVERIFY2(unMatched <= tol, qPrintable(QString("Corners do not have the same oriented briefs features (%1 unmatched pixels)").arg(unMatched)));
    }

    //test if the system can hand multi-channels images.
    Multidim::Array<float,3> test3Channels = generateMultiProblem(re, searchRadius);

    std::vector<std::array<int, 2>> centralPoint = {{searchRadius, searchRadius}};

    std::vector<orientedCoordinate<2>> orientedCentralPoint = intensityOrientedCoordinates<true>(centralPoint, test3Channels, searchRadius);

    std::vector<ComparisonPair<3>> comparisonPairsMultiChannels = generateRandomComparisonPairs<2>(nSamples, windowsRadius, test3Channels.shape()[2]);

    std::vector<pointFeatures<2, std::vector<uint32_t>>> descriptors3Channels = BriefDescriptor<true>(orientedCentralPoint, test3Channels, comparisonPairsMultiChannels);

    QCOMPARE(descriptors3Channels.size(), 1);



}

QTEST_MAIN(TestSparseMatchingUtils)
#include "testSparseMatchingUtils.moc"
