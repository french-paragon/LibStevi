#include <QtTest/QtTest>

#include "optimization/principalComponentsAnalysis.h"

#include <iostream>
#include <random>

using namespace StereoVision::Optimization;

template<typename T, int nDims, int nData>
Eigen::Matrix<T, nDims, nData> generateSampleProblem() {

    return Eigen::Matrix<T, nDims, nData>::Random();
}

class TestPCA: public QObject{

    Q_OBJECT
private Q_SLOTS:

    void initTestCase();

    void testBasicPCA();

private:
    std::default_random_engine re;
};


void TestPCA::initTestCase() {
    srand((unsigned int) time(nullptr));
    std::random_device rd;
    re.seed(rd());
}

void TestPCA::testBasicPCA() {

    using T = float;

    constexpr int nDims = 4;
    constexpr int nData = 30;
    constexpr int nComps = 3;

    Eigen::Matrix<T, nDims, nData> data = generateSampleProblem<T, nDims, nData>();

    Eigen::Matrix<T, nComps, nDims> pc = StereoVision::Optimization::principalComponents<T, nComps>(data);

    for (int i = 0; i < nComps; i++) {
        for (int j = i+1; j < nComps; j++) {
            QVERIFY2(pc.row(i).dot(pc.row(j)) <= 1e-4, "Non orthogonal vector in the projection space!");
        }
    }

}

QTEST_MAIN(TestPCA)
#include "testPCA.moc"
