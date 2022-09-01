#include <QtTest/QtTest>

#include "optimization/l1optimization.h"

#include <random>

class BenchmarkNormSolverAlgorithms: public QObject
{
	Q_OBJECT

private Q_SLOTS:
	void initTestCase();

	void benchmarkLeastAbsoluteDifferences_data();
	void benchmarkLeastAbsoluteDifferences();

private:
	std::default_random_engine re;

};
void BenchmarkNormSolverAlgorithms::initTestCase() {
	srand((unsigned int) time(nullptr));
}

void BenchmarkNormSolverAlgorithms::benchmarkLeastAbsoluteDifferences_data() {

	QTest::addColumn<int>("inputs");

	QTest::newRow("3 coefficients") << 3;
	QTest::newRow("7 coefficients") << 7;
	QTest::newRow("32 coefficients") << 32;
	QTest::newRow("50 coefficients") << 50;
}
void BenchmarkNormSolverAlgorithms::benchmarkLeastAbsoluteDifferences() {

	typedef Eigen::Matrix<float,Eigen::Dynamic,2> TypeMatrixA;
	typedef Eigen::Matrix<float,2,1> TypeVectorX;

	QFETCH(int, inputs);
	constexpr float tol = 1e-6;

	TypeMatrixA A = TypeMatrixA::Random(inputs,2);
	Eigen::VectorXf b = Eigen::VectorXf::Random(inputs);

	TypeVectorX eX;
	QBENCHMARK(eX = StereoVision::Optimization::leastAbsoluteDifferences(A,b,tol,100));
}

QTEST_MAIN(BenchmarkNormSolverAlgorithms)
#include "benchmarkNormSolverAlgorithms.moc"
