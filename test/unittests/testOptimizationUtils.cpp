#include <QtTest/QtTest>

#include "optimization/affine_utils.h"

#include <random>
#include <eigen3/Eigen/Core>


using namespace StereoVision::Optimization;

class TestOptimizationUtils: public QObject
{
	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testAffineFullCoefficients();
};

void TestOptimizationUtils::initTestCase() {
	srand((unsigned int) time(nullptr));
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

QTEST_MAIN(TestOptimizationUtils);
#include "testOptimizationUtils.moc"
