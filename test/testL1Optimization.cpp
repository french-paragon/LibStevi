#include <QtTest/QtTest>

#include "optimization/l1optimization.h"

#include <iostream>
#include <random>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>

using namespace StereoVision::Optimization;

class TestL1OptimizationMethods: public QObject
{
	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testLeastAbsoluteDifferences_data();
	void testLeastAbsoluteDifferences();

	void benchmarkLeastAbsoluteDifferences_data();
	void benchmarkLeastAbsoluteDifferences();
};


void TestL1OptimizationMethods::initTestCase() {
	srand((unsigned int) time(nullptr));
}

void TestL1OptimizationMethods::testLeastAbsoluteDifferences_data() {

	QTest::addColumn<int>("inputs");

	QTest::newRow("Minimal") << 3;
	QTest::newRow("small") << 7;
	QTest::newRow("big") << 32;
	QTest::newRow("large") << 50;

}

void TestL1OptimizationMethods::testLeastAbsoluteDifferences() {

	typedef Eigen::Matrix<float,Eigen::Dynamic,2> TypeMatrixA;
	typedef Eigen::Matrix<float,2,2> TypeMatrixRA;
	typedef Eigen::Matrix<float,2,1> TypeVectorX;

	QFETCH(int, inputs);
	constexpr float tol = 1e-6;

	TypeMatrixA A = TypeMatrixA::Random(inputs,2);
	Eigen::VectorXf b = Eigen::VectorXf::Random(inputs);

	TypeVectorX trueX = TypeVectorX::Zero();
	float func = std::numeric_limits<float>::infinity();

	for (int i = 0; i < inputs-1; i++) {
		for (int j = i+1; j < inputs; j++) {
			TypeMatrixRA RA;
			RA.row(0) = A.row(i);
			RA.row(1) = A.row(j);

			Eigen::Vector2f Rb;
			Rb(0) = b(i);
			Rb(1) = b(j);

			Eigen::Vector2f intersection = RA.colPivHouseholderQr().solve(Rb);

			auto vec = A*intersection - b;
			float n = vec.lpNorm<1>();

			if (n < func) {
				func = n;
				trueX = intersection;
			}
		}
	}

	TypeVectorX eX = leastAbsoluteDifferences(A,b,tol,inputs*100);

	auto diff = trueX - eX;

	float missalignement = diff.norm();
	QVERIFY2(missalignement < inputs*tol, qPrintable(QString("Reconstructed vector x not correct (norm (x^check - x^hat) = %1)").arg(missalignement)));


}

void TestL1OptimizationMethods::benchmarkLeastAbsoluteDifferences_data() {

	QTest::addColumn<int>("inputs");

	QTest::newRow("Minimal") << 3;
	QTest::newRow("small") << 7;
	QTest::newRow("big") << 32;
	QTest::newRow("large") << 50;
}
void TestL1OptimizationMethods::benchmarkLeastAbsoluteDifferences() {

	typedef Eigen::Matrix<float,Eigen::Dynamic,2> TypeMatrixA;
	typedef Eigen::Matrix<float,2,1> TypeVectorX;

	QFETCH(int, inputs);
	constexpr float tol = 1e-6;

	TypeMatrixA A = TypeMatrixA::Random(inputs,2);
	Eigen::VectorXf b = Eigen::VectorXf::Random(inputs);

	TypeVectorX eX;
	QBENCHMARK(eX = leastAbsoluteDifferences(A,b,tol,100));
}

QTEST_MAIN(TestL1OptimizationMethods)
#include "testL1Optimization.moc"
