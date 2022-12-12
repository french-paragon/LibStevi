#include <QtTest/QtTest>

#include "geometry/rotations.h"

#include <random>

using namespace StereoVision::Geometry;

Q_DECLARE_METATYPE(Eigen::Matrix3f)

class TestGeometryLibRotation: public QObject
{
	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testRodriguez_data();
	void testRodriguez();

	void testInverseRodriguez_data();
	void testInverseRodriguez();

	void testDiffRodriguez_data();
	void testDiffRodriguez();

	void testDiffRigidTransformInverse();

private:
	std::default_random_engine re;
};


void TestGeometryLibRotation::initTestCase() {
	//srand((unsigned int) time(nullptr));
	std::random_device rd;
	re.seed(rd());
}

void TestGeometryLibRotation::testRodriguez_data() {

	QTest::addColumn<float>("rx");
	QTest::addColumn<float>("ry");
	QTest::addColumn<float>("rz");
	QTest::addColumn<Eigen::Matrix3f>("M");

	Eigen::Matrix3f M;

	M.setIdentity();
	QTest::newRow("Identity") << 0.0f << 0.0f << 0.0f << M;

	M << 1, 0, 0,
		 0, 0,-1,
		 0, 1, 0;
	QTest::newRow("90deg x axis") << static_cast<float>(M_PI/2) << 0.0f << 0.0f << M;

	M << 0, 0, 1,
		 0, 1, 0,
		-1, 0, 0;
	QTest::newRow("90deg y axis") << 0.0f << static_cast<float>(M_PI/2) << 0.0f << M;

	M << 0,-1, 0,
		 1, 0, 0,
		 0, 0, 1;
	QTest::newRow("90deg z axis") << 0.0f << 0.0f << static_cast<float>(M_PI/2) << M;

	M << 1, 0, 0,
		 0,-1, 0,
		 0, 0,-1;
	QTest::newRow("180deg x axis") << static_cast<float>(M_PI) << 0.0f << 0.0f << M;

	M <<-1, 0, 0,
		 0, 1, 0,
		 0, 0,-1;
	QTest::newRow("180deg y axis") << 0.0f << static_cast<float>(M_PI) << 0.0f << M;

	M <<-1, 0, 0,
		 0,-1, 0,
		 0, 0, 1;
	QTest::newRow("180deg z axis") << 0.0f << 0.0f << static_cast<float>(M_PI) << M;
}

void TestGeometryLibRotation::testRodriguez() {

	QFETCH(float, rx);
	QFETCH(float, ry);
	QFETCH(float, rz);
	QFETCH(Eigen::Matrix3f, M);

	Eigen::Matrix3f S = rodriguezFormula(Eigen::Vector3f(rx, ry, rz));

	float mismatch = (S.transpose()*M - Eigen::Matrix3f::Identity()).norm();

	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Reconstructed rotation not correct (norm (RgtxRrc - I) = %1)").arg(mismatch)));

}

void TestGeometryLibRotation::testInverseRodriguez_data() {

	QTest::addColumn<float>("rx");
	QTest::addColumn<float>("ry");
	QTest::addColumn<float>("rz");

	QTest::newRow("Identity") << 0.0f << 0.0f << 0.0f;

	QTest::newRow("x axis one") << 1.0f << 0.0f << 0.0f;
	QTest::newRow("y axis one") << 0.0f << 1.0f << 0.0f;
	QTest::newRow("z axis one") << 0.0f << 0.0f << 1.0f;

	QTest::newRow("x axis pi") << static_cast<float>(M_PI) << 0.0f << 0.0f;
	QTest::newRow("y axis pi") << 0.0f << static_cast<float>(M_PI) << 0.0f;
	QTest::newRow("z axis pi") << 0.0f << 0.0f << static_cast<float>(M_PI);

	QTest::newRow("pseudo random 1") << 0.2f << 0.5f << -1.3f;
	QTest::newRow("pseudo random 2") << 0.8f << 1.1f << 0.3f;
	QTest::newRow("pseudo random 3") << -0.5f << -0.4f << 1.2f;

}
void TestGeometryLibRotation::testInverseRodriguez() {

	QFETCH(float, rx);
	QFETCH(float, ry);
	QFETCH(float, rz);

	Eigen::Vector3f vec(rx, ry, rz);

	Eigen::Matrix3f S = rodriguezFormula(vec);

	Eigen::Vector3f recons = inverseRodriguezFormula(S);

	float mismatch = (vec - recons).norm();

	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Reconstructed rotation axis not correct (norm (rgt - rrc) = %1)").arg(mismatch)));

}

void TestGeometryLibRotation::testDiffRodriguez_data() {

	QTest::addColumn<float>("rx");
	QTest::addColumn<float>("ry");
	QTest::addColumn<float>("rz");

	QTest::newRow("Identity") << 0.0f << 0.0f << 0.0f;

	QTest::newRow("x axis one") << 1.0f << 0.0f << 0.0f;
	QTest::newRow("y axis one") << 0.0f << 1.0f << 0.0f;
	QTest::newRow("z axis one") << 0.0f << 0.0f << 1.0f;

	QTest::newRow("x axis pi") << static_cast<float>(M_PI) << 0.0f << 0.0f;
	QTest::newRow("y axis pi") << 0.0f << static_cast<float>(M_PI) << 0.0f;
	QTest::newRow("z axis pi") << 0.0f << 0.0f << static_cast<float>(M_PI);

}
void TestGeometryLibRotation::testDiffRodriguez() {

	QFETCH(float, rx);
	QFETCH(float, ry);
	QFETCH(float, rz);

	float epsilon = 1e-4;

	Eigen::Vector3f vec(rx, ry, rz);
	Eigen::Vector3f drx(epsilon, 0, 0);
	Eigen::Vector3f dry(0, epsilon, 0);
	Eigen::Vector3f drz(0, 0, epsilon);

	Eigen::Matrix3f S = rodriguezFormula(vec);

	Eigen::Matrix3f dX_numeric = (rodriguezFormula<float>(vec + drx) - S)/epsilon;
	Eigen::Matrix3f dY_numeric = (rodriguezFormula<float>(vec + dry) - S)/epsilon;
	Eigen::Matrix3f dZ_numeric = (rodriguezFormula<float>(vec + drz) - S)/epsilon;

	Eigen::Matrix3f dX_analytic = diffRodriguez(vec, Axis::X);
	Eigen::Matrix3f dY_analytic = diffRodriguez(vec, Axis::Y);
	Eigen::Matrix3f dZ_analytic = diffRodriguez(vec, Axis::Z);

	float mismatchX = (dX_numeric - dX_analytic).norm();
	QVERIFY2(mismatchX < 2e1*epsilon, qPrintable(QString("Reconstructed diff rotation axis not correct (norm (dx_ranalitic - dx_rnumeric) = %1)").arg(mismatchX)));

	float mismatchY = (dY_numeric - dY_analytic).norm();
	QVERIFY2(mismatchY < 2e1*epsilon, qPrintable(QString("Reconstructed diff rotation axis not correct (norm (dy_ranalitic - dy_rnumeric) = %1)").arg(mismatchY)));

	float mismatchZ = (dZ_numeric - dZ_analytic).norm();
	QVERIFY2(mismatchZ < 2e1*epsilon, qPrintable(QString("Reconstructed diff rotation axis not correct (norm (dz_ranalitic - dz_rnumeric) = %1)").arg(mismatchZ)));

}

void TestGeometryLibRotation::testDiffRigidTransformInverse() {

	constexpr int nTest = 100;

	float epsilon = 1e-4;

	std::uniform_real_distribution<float> dataGen(-10, 10);
	std::uniform_real_distribution<float> translateGen(-10, 10);
	std::uniform_real_distribution<float> rotGen(-5, 5);
	std::uniform_real_distribution<float> scaleGen(0.2, 3);
	std::uniform_int_distribution<int> scaleSign(0,1);

	for (int i = 0; i < nTest; i++) {
		Eigen::Vector3f r(rotGen(re),rotGen(re),rotGen(re));
		Eigen::Vector3f t(translateGen(re),translateGen(re),translateGen(re));
		float s = scaleGen(re);
		if (scaleSign(re)) {
			s = -s;
		}

		ShapePreservingTransform direct(r, t, s);
		ShapePreservingTransform inverse = direct.inverse();

		for (int j = 0; j < nTest; j++) {
			Eigen::Vector3f v(dataGen(re),dataGen(re),dataGen(re));
			Eigen::Vector3f tmp = direct*v;

			if (tmp.array().isInf().any() or tmp.array().isNaN().any()) {
				continue;
			}

			Eigen::Vector3f t = inverse*(tmp);

			float mismatch = (v - t).norm();
			QVERIFY2(mismatch <= epsilon, qPrintable(QString("Reconstructed vector is too different from original (mismatch = %1)").arg(mismatch)));
		}
	}

}

QTEST_MAIN(TestGeometryLibRotation)
#include "testRotations.moc"
