#include <QtTest/QtTest>

#include "geometry/rotations.h"

#include <random>

using namespace StereoVision::Geometry;

Q_DECLARE_METATYPE(Eigen::Matrix3f)
Q_DECLARE_METATYPE(Eigen::Vector3f)

class TestGeometryLibRotation: public QObject
{
	Q_OBJECT
private Q_SLOTS:

    void initTestCase();

    void testRodriguez_data();
    void testRodriguez();

    void testAngleAxisRotate_data();
    void testAngleAxisRotate();

    void testDiffAngleAxisRotate();

	void testInverseRodriguez_data();
	void testInverseRodriguez();

	void testDiffRodriguez_data();
	void testDiffRodriguez();

    void testDiffRigidBodyTransform();
    void testDiffShapePreservingTransform();

	void testDiffRigidTransformInverse();

    void testEulerRad2RMat();

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

void TestGeometryLibRotation::testAngleAxisRotate_data() {

    QTest::addColumn<Eigen::Vector3f>("rAxis");

    QTest::newRow("90deg x axis") << Eigen::Vector3f(M_PI_2, 0, 0);
    QTest::newRow("90deg y axis") << Eigen::Vector3f(0, M_PI_2, 0);
    QTest::newRow("90deg z axis") << Eigen::Vector3f(0, 0, M_PI_2);

    QTest::newRow("180deg x axis") << Eigen::Vector3f(M_PI, 0, 0);
    QTest::newRow("180deg y axis") << Eigen::Vector3f(0, M_PI, 0);
    QTest::newRow("180deg z axis") << Eigen::Vector3f(0, 0, M_PI);

    QTest::newRow("Random axis 1") << Eigen::Vector3f(0.23, 2.14, -0.342);
    QTest::newRow("Random axis 2") << Eigen::Vector3f(-1.234, 0.963, -0.726);
    QTest::newRow("Random axis 3") << Eigen::Vector3f(7.562, -6.273, 21.23);

}
void TestGeometryLibRotation::testAngleAxisRotate() {

    QFETCH(Eigen::Vector3f, rAxis);

    Eigen::Matrix3f R = StereoVision::Geometry::rodriguezFormula(rAxis);

    std::uniform_real_distribution<float> rd(-10, 10);

    for (int i = 0; i < 42; i++) {
        Eigen::Vector3f vec(rd(re), rd(re), rd(re));

        Eigen::Vector3f Rvec = R*vec;
        Eigen::Vector3f rvec = angleAxisRotate(rAxis, vec);

        Eigen::Vector3f delta = Rvec - rvec;

        float mismatch = delta.norm();

        QVERIFY2(mismatch < 1e-5, qPrintable(QString("Rotated vector by angle axis is not the same as with rotation matrix! (mismatch = %1)").arg(mismatch)));
    }

}

void TestGeometryLibRotation::testDiffAngleAxisRotate() {

    std::uniform_real_distribution<double> rd(-10, 10);

    double delta = 1e-5;

    for (int i = 0; i < 42; i++) {
        Eigen::Vector3d vec(rd(re), rd(re), rd(re));
        Eigen::Vector3d randomRAxis(rd(re), rd(re), rd(re));

        for (int axis = int(Axis::X); axis <= int(Axis::Z); axis++) {

            Eigen::Vector3d deltaVec = Eigen::Vector3d::Zero();
            deltaVec[axis] = delta;

            Eigen::Vector3d numDiff = (angleAxisRotate<double>(randomRAxis + deltaVec, vec) - angleAxisRotate<double>(randomRAxis - deltaVec, vec))/(2*delta);
            Eigen::Vector3d analDiff = diffAngleAxisRotate(randomRAxis, vec, Axis(axis));

            for (int idx = 0; idx < 3; idx++) {
                QCOMPARE(float(analDiff[idx]), float(numDiff[idx]));
            }

        }
    }


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

void TestGeometryLibRotation::testDiffRigidBodyTransform() {

    constexpr int nRepeats = 42;

    std::uniform_real_distribution<double> rd(-10, 10);

    double delta = 1e-5;

    for (int i = 0; i < nRepeats; i++) {
        Eigen::Vector3d vec(rd(re), rd(re), rd(re));
        Eigen::Matrix<double,6,1> randomTransform(rd(re), rd(re), rd(re), rd(re), rd(re), rd(re));

        RigidBodyTransform transform(randomTransform);
        Eigen::Matrix<double,3,6> Jac = transform.Jacobian(vec);

        for (int axis = 0; axis < 6; axis++) {

            Eigen::Matrix<double,6,1> deltaVec = Eigen::Matrix<double,6,1>::Zero();
            deltaVec[axis] = delta;

            RigidBodyTransform<double> prev(randomTransform-deltaVec);
            RigidBodyTransform<double> next(randomTransform+deltaVec);

            Eigen::Vector3d numDiff = (next*vec - prev*vec)/(2*delta);

            for (int idx = 0; idx < 3; idx++) {
                QCOMPARE(float(Jac(idx,axis)), float(numDiff[idx]));
            }

        }

    }

}
void TestGeometryLibRotation::testDiffShapePreservingTransform() {

    constexpr int nRepeats = 42;

    std::uniform_real_distribution<double> rd(-10, 10);

    double delta = 1e-5;

    for (int i = 0; i < nRepeats; i++) {

        Eigen::Vector3d vec(rd(re), rd(re), rd(re));
        Eigen::Matrix<double,7,1> randomTransform(rd(re), rd(re), rd(re), rd(re), rd(re), rd(re), rd(re));

        ShapePreservingTransform transform(randomTransform);
        Eigen::Matrix<double,3,7> Jac = transform.Jacobian(vec);

        for (int axis = 0; axis < 7; axis++) {

            Eigen::Matrix<double,7,1> deltaVec = Eigen::Matrix<double,7,1>::Zero();
            deltaVec[axis] = delta;

            ShapePreservingTransform<double> prev(randomTransform-deltaVec);
            ShapePreservingTransform<double> next(randomTransform+deltaVec);

            Eigen::Vector3d numDiff = (next*vec - prev*vec)/(2*delta);

            for (int idx = 0; idx < 3; idx++) {
                QCOMPARE(float(Jac(idx,axis)), float(numDiff[idx]));
            }

        }

    }
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

void TestGeometryLibRotation::testEulerRad2RMat() {

    Eigen::Matrix3f R0 = eulerRadXYZToRotation<float>(0,0,0);

    float epsilon = 1e-5;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {

            float target = (i == j) ? 1 : 0;

            float mismatch = std::abs(R0(i,j) - target);
            QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a 0 angle rotation (mismatch = %1)").arg(mismatch)));
        }
    }

    Eigen::Matrix3f RX = eulerRadXYZToRotation<float>(M_PI_2,0,0);
    Eigen::Matrix3f RY = eulerRadXYZToRotation<float>(0,M_PI_2,0);
    Eigen::Matrix3f RZ = eulerRadXYZToRotation<float>(0,0,M_PI_2);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {

            float target = 0;

            if ((i == 0 and j == 0) or (i == 2 and j == 1)) {
                target = 1;
            }

            if (i == 1 and j == 2) {
                target = -1;
            }

            float mismatch = std::abs(RX(i,j) - target);
            QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a 90 angle rotation around the X axis (mismatch = %1)").arg(mismatch)));
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {

            float target = 0;

            if ((i == 1 and j == 1) or (i == 0 and j == 2)) {
                target = 1;
            }

            if (i == 2 and j == 0) {
                target = -1;
            }

            float mismatch = std::abs(RY(i,j) - target);
            QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a 90 angle rotation around the Y axis (mismatch = %1)").arg(mismatch)));
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {

            float target = 0;

            if ((i == 2 and j == 2) or (i == 1 and j == 0)) {
                target = 1;
            }

            if (i == 0 and j == 1) {
                target = -1;
            }

            float mismatch = std::abs(RZ(i,j) - target);
            QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a 90 angle rotation around the Z axis (mismatch = %1)").arg(mismatch)));
        }
    }

}

QTEST_MAIN(TestGeometryLibRotation)
#include "testRotations.moc"
