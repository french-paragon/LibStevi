#include <QtTest/QtTest>

#include "imageProcessing/pixelsLines.h"
#include "imageProcessing/pixelsTriangles.h"

class TestPixelsSets: public QObject
{
    Q_OBJECT
private Q_SLOTS:
    void initTestCase();

    void testFlatLine();

    void testSimpleTriangles();

};

void TestPixelsSets::initTestCase() {

}

void TestPixelsSets::testFlatLine() {

    Eigen::Vector2f v0(0,0);
    Eigen::Vector2f v1(5,0);

    Eigen::Array<float, 2, Eigen::Dynamic> coords = StereoVision::ImageProcessing::listPixPointsOnLine(v0, v1);

    QCOMPARE(coords.cols(),6);

    for (int i = 0; i < 6; i++) {

        QCOMPARE(coords(0,i), i);
        QCOMPARE(coords(1,i), 0);
    }

    coords = StereoVision::ImageProcessing::listPixPointsOnLine(v1, v0);

    QCOMPARE(coords.cols(),6);

    for (int i = 0; i < 6; i++) {

        QCOMPARE(coords(0,i), 5-i);
        QCOMPARE(coords(1,i), 0);
    }

    Eigen::Vector2f v2(5,5);

    coords = StereoVision::ImageProcessing::listPixPointsOnLine(v0, v2);

    QCOMPARE(coords.cols(),11);

    for (int i = 0; i < 11; i += 2) {

        QCOMPARE(coords(0,i), i/2);
        QCOMPARE(coords(1,i), i/2);

        if (i < 10) {
            QCOMPARE(coords(0,i+1), i/2+0.5);
            QCOMPARE(coords(1,i+1), i/2+0.5);
        }
    }

}

void TestPixelsSets::testSimpleTriangles() {

    std::vector<std::array<int,2>> expectedPresentCoords = {{0,0}, {-1,1}, {1,-1}, {0,1}, {1,0}, {1,1}};
    std::vector<bool> foundGtCoord(expectedPresentCoords.size());

    std::fill(foundGtCoord.begin(), foundGtCoord.end(), false);

    Eigen::Matrix<float,2,1> pt1(-0.5,0.5);
    Eigen::Matrix<float,2,1> pt2(0.5,-0.5);
    Eigen::Matrix<float,2,1> pt3(0.5,0.5);

    std::vector<StereoVision::ImageProcessing::WeigthedPixCoord<float>> result =
            StereoVision::ImageProcessing::listPixPointsInTriangle(pt1, pt2, pt3);

    QCOMPARE(result.size(), expectedPresentCoords.size());

    for (StereoVision::ImageProcessing::WeigthedPixCoord<float> const& pix : result) {

        std::array<int,2> coord = {pix.pixCoord.x(), pix.pixCoord.y()};

        auto found = std::find(expectedPresentCoords.begin(), expectedPresentCoords.end(), coord);

        QVERIFY(found != expectedPresentCoords.end());

        QVERIFY(!foundGtCoord[std::distance(expectedPresentCoords.begin(), found)]);
        foundGtCoord[std::distance(expectedPresentCoords.begin(), found)] = true;

    }

}

QTEST_MAIN(TestPixelsSets)
#include "testPixelsSets.moc"
