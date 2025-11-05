#include <QtTest/QtTest>

#include "optimization/affine_utils.h"
#include "optimization/huber_kernel.h"
#include "optimization/l2optimization.h"
#include "optimization/generic_ransac.h"

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
    void testGenericRansac();

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


void TestOptimizationUtils::testGenericRansac() {

    using Observation = std::array<float,2>; //x, y pair

    struct Model {
        Model() :
            _a(0),
            _b(0)
        {

        }
        Model(std::vector<Observation> const& observations) {

            Eigen::Matrix<float, Eigen::Dynamic, 2> A;
            A.resize(observations.size(), 2);

            Eigen::VectorXf b;
            b.resize(observations.size());

            for (int i = 0; i < observations.size(); i++) {
                A(i,0) = observations[i][0];
                A(i,1) = 1;
                b[i] = observations[i][1];
            }

            Eigen::Vector2f x = StereoVision::Optimization::leastSquares(A,b);

            _a = x[0];
            _b = x[1];

        }

        float error(Observation const& observation) {
            float const& x = observation[0];
            float const& y = observation[1];
            float res = y - (_a*x + _b);
            return std::fabs(res);
        }

        float _a;
        float _b;
    };

    std::uniform_real_distribution<float> params_dist(-27,33);
    std::uniform_real_distribution<float> x_dist(-42,69);

    float a_true = params_dist(re);
    float b_true = params_dist(re);

    constexpr int nCorrect = 100;

    std::vector<Observation> corrects(nCorrect);

    for (int i = 0; i < nCorrect; i++) {
        float x = x_dist(re);
        corrects[i][0] = x;
        corrects[i][1] = a_true*x + b_true;
    }
    //we use only correct values to ensure the tested ransac converge correctly (e.g. unitest will not fail due to sheer lack of luck)

    constexpr float threshold = 1e-3;

    GenericRansac<Observation, Model> ransac(corrects, 2, threshold);

    constexpr int nIterations = 5;

    ransac.ransacIterations(nIterations);

    Model finalModel = ransac.currentModel();

    QVERIFY(std::abs(finalModel._a - a_true) < 1e-3);
    QVERIFY(std::abs(finalModel._b - b_true) < 1e-3);

    QCOMPARE(ransac.currentInliers().size(), corrects.size());

    for (int i = 0; i < corrects.size(); i++) {
        QVERIFY(ransac.currentInliers()[i]);
    }

}

QTEST_MAIN(TestOptimizationUtils);
#include "testOptimizationUtils.moc"
