#include <QtTest/QtTest>

#include "optimization/affine_utils.h"
#include "optimization/huber_kernel.h"

#include <random>
#include <eigen3/Eigen/Core>


using namespace StereoVision::Optimization;

class TestOptimizationUtils: public QObject
{
	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

    void testAffineFullCoefficients();
    void testHuberLossDiffs();
    void testPseudoHuberLossDiffs();
    void testSqrtHuberLossDiffs();

private:

    std::default_random_engine re;
};

void TestOptimizationUtils::initTestCase() {
	srand((unsigned int) time(nullptr));
    std::random_device rd;
    re.seed(rd());
}

void TestOptimizationUtils::testAffineFullCoefficients() {
	typedef StereoVision::Optimization::AffineSpace<4,Eigen::Dynamic,float> AffSpace;

	AffSpace::TypeVectorCoeffs coeffs;

	for (int i = 0; i < 100; i++) {
		coeffs = AffSpace::TypeVectorCoeffs::Random();
		AffSpace::TypeVectorCoeffsFull coeffsFull = AffSpace::fullCoeffs(coeffs);
		QCOMPARE(coeffsFull.sum(), 1.f);
	}
}

void TestOptimizationUtils::testHuberLossDiffs() {

    constexpr double delta = 1e-7;

    std::uniform_real_distribution<double> randomDist(-100, 100);

    for (int i = 0; i < 100; i++) {
        double val = randomDist(re);

        double diffA = diffHuberLoss(val);
        double diff2A = diff2HuberLoss(val);

        double diffN = (huberLoss(val+delta) - huberLoss(val-delta)) / (2*delta);
        double diff2N = (diffHuberLoss(val+delta) - diffHuberLoss(val-delta)) / (2*delta);

        QVERIFY2(std::abs(diffN - diffA) < 10*delta, "Wrong derivative");
        QVERIFY2(std::abs(diff2N - diff2A) < 10*delta, "Wrong second derivative");
    }

}
void TestOptimizationUtils::testPseudoHuberLossDiffs() {

    constexpr double delta = 1e-7;

    std::uniform_real_distribution<double> randomDist(-100, 100);

    for (int i = 0; i < 100; i++) {
        double val = randomDist(re);

        double diffA = diffPseudoHuberLoss(val);
        double diff2A = diff2PseudoHuberLoss(val);

        double diffN = (pseudoHuberLoss(val+delta) - pseudoHuberLoss(val-delta)) / (2*delta);
        double diff2N = (diffPseudoHuberLoss(val+delta) - diffPseudoHuberLoss(val-delta)) / (2*delta);

        QVERIFY2(std::abs(diffN - diffA) < 10*delta, "Wrong derivative");
        QVERIFY2(std::abs(diff2N - diff2A) < 10*delta, "Wrong second derivative");
    }

}
void TestOptimizationUtils::testSqrtHuberLossDiffs() {

    constexpr double delta = 1e-7;

    std::uniform_real_distribution<double> randomDist(-100, 100);

    for (int i = 0; i < 100; i++) {
        double val = randomDist(re);

        double v = huberLoss(val);
        double sqrtV = sqrtHuberLoss(val);

        double diffA = diffSqrtHuberLoss(val);

        double diffN = (sqrtHuberLoss(val+delta) - sqrtHuberLoss(val-delta)) / (2*delta);

        QVERIFY2(std::abs(v - sqrtV*sqrtV) < 10*delta, "Wrong value of loss");
        QVERIFY2(std::abs(diffN - diffA) < 10*delta, "Wrong derivative");
    }

}

QTEST_MAIN(TestOptimizationUtils);
#include "testOptimizationUtils.moc"
