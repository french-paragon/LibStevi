#include <QtTest/QtTest>

#include "imageProcessing/pixelsLines.h"

class TestPixelsLines: public QObject
{
    Q_OBJECT
private Q_SLOTS:
    void initTestCase();

    void testFlatLine();

};

void TestPixelsLines::initTestCase() {

}

void TestPixelsLines::testFlatLine() {

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

QTEST_MAIN(TestPixelsLines)
#include "testPixelsLines.moc"
