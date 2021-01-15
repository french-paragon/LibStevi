#include <QtTest/QtTest>

#include "geometry/core.h"

using namespace StereoVision::Geometry;


class TestGeometryLibCore: public QObject
{
	Q_OBJECT
private Q_SLOTS:
	void testAffineTransform();
};

void TestGeometryLibCore::testAffineTransform() {

	const int s = 42;

	Eigen::Vector3f tmp;

	Eigen::Array3Xf array;
	array.setRandom(3,s);

	Eigen::Array3Xf matrix;
	matrix.setRandom(3,s);

	Eigen::Matrix3f M;
	M.setRandom();

	Eigen::Vector3f t;
	t.setRandom();

	AffineTransform T(M,t);

	Eigen::Array3Xf t_array = T*array;
	Eigen::Array3Xf t_matrix = T*matrix;

	for (int i = 0; i < s; i++) {

		tmp = array.col(i);

		float e = (T*tmp - t_array.col(i).matrix()).norm();
		QVERIFY2(e < 1e-3, qPrintable(QString("Unexpected error (%1)").arg(e)));

		tmp = matrix.col(i);

		e = (T*tmp - t_matrix.col(i).matrix()).norm();
		QVERIFY2(e < 1e-3, qPrintable(QString("Unexpected error (%1)").arg(e)));

	}

}

QTEST_MAIN(TestGeometryLibCore)
#include "testCore.moc"
