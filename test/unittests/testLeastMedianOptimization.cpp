#include <QtTest/QtTest>

#include "optimization/leastmedianoptimization.h"

#include <iostream>
#include <random>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>

using namespace StereoVision::Optimization;

class TestLeastMedianOptimizationMethods: public QObject
{
	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testLeastMedianAbsoluteDifferences_data();
	void testLeastMedianAbsoluteDifferences();
};


void TestLeastMedianOptimizationMethods::initTestCase() {
	srand((unsigned int) time(nullptr));
}

void TestLeastMedianOptimizationMethods::testLeastMedianAbsoluteDifferences_data() {

	QTest::addColumn<int>("inputs");

	QTest::newRow("Minimal") << 3;
	QTest::newRow("small") << 7;
	QTest::newRow("big") << 32;
	QTest::newRow("large") << 50;

}

void TestLeastMedianOptimizationMethods::testLeastMedianAbsoluteDifferences() {

	typedef Eigen::Matrix<float,Eigen::Dynamic,2> TypeMatrixA;
	typedef Eigen::Matrix<float,2,1> TypeVectorX;

	QFETCH(int, inputs);
	constexpr float tol = 1e-6;
	constexpr float probOut = 0.1;

	TypeMatrixA A = TypeMatrixA::Random(inputs,2);
	Eigen::Vector2f sol = Eigen::Vector2f::Random();
	Eigen::VectorXf b = A*sol;
	Eigen::VectorXf randv = Eigen::VectorXf::Random(inputs);
	Eigen::VectorXf thresh = Eigen::VectorXf::Random(inputs);

	int c = 0;

	for (int i = 0; i < inputs; i++) {
		c++; // <3

		if (c > probOut*inputs) { //do not corrupt more than the proportion.
			break;
		}

		if ((thresh(i) + 1.)/2. < probOut) {
			b(i) += randv(i);
		}
	}
	TypeVectorX eX = leastAbsoluteMedian(A,b,0.99,3*probOut, 1000);

	auto diff = sol - eX;

	float missalignement = diff.norm();
	QVERIFY2(missalignement < inputs*tol, qPrintable(QString("Reconstructed vector x not correct (norm (x^check - x^hat) = %1)").arg(missalignement)));


}

QTEST_MAIN(TestLeastMedianOptimizationMethods)
#include "testLeastMedianOptimization.moc"
