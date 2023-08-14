#include <QtTest/QtTest>

#include "sparseMatching/nonLocalMaximumPointSelection.h"

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

class TestSparseMatchingUtils: public QObject{

    Q_OBJECT
private Q_SLOTS:

    void initTestCase();

    void testNonMaximumPointSelection();

private:
    std::default_random_engine re;
};


void TestSparseMatchingUtils::initTestCase() {
    srand((unsigned int) time(nullptr));
    std::random_device rd;
    re.seed(rd());
}

void TestSparseMatchingUtils::testNonMaximumPointSelection() {

    constexpr int radius = 3;

    Multidim::Array<float,2> test = generateSampleProblem(re, radius);

    std::vector<std::array<float, 2>> result = nonLocalMaximumPointSelection(Multidim::Array<float, 2, Multidim::ConstView>(test), radius, 0.f);

    QCOMPARE(result.size(), 1);
    QCOMPARE(result[0][0], radius);
    QCOMPARE(result[0][1], radius);

}

QTEST_MAIN(TestSparseMatchingUtils)
#include "testSparseMatchingUtils.moc"
