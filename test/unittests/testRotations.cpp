#include <QtTest/QtTest>

#include "geometry/rotations.h"
#include "geometry/sensorframesconvention.h"

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

    void testInverseRodriguezNumStable_data();
    void testInverseRodriguezNumStable();

    void testJacobianSO3_data();
    void testJacobianSO3();

	void testDiffRodriguez_data();
	void testDiffRodriguez();

    void testDiffRigidBodyTransform();
    void testDiffShapePreservingTransform();

    void testRigidTransformInverse();

    void testEulerRad2RMat();
    void testRMat2EulerRad();

    void testRAxis2Quat2RAxis_data();
    void testRAxis2Quat2RAxis();

    void testRAxis2QuatDiff();

    void testSensorFrameConversions();

    void testRigidBodyTransformInterpolationOnManifold();

private:
	std::default_random_engine re;
};


void TestGeometryLibRotation::initTestCase() {
    srand((unsigned int) time(nullptr));
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


    //failure cases that have been encountered during devellopement
    //TODO: check if we want to use a different test protocol for these ones

    /*Eigen::Matrix3d Md;
    Md << 0.55796742,  0.82778577,  0.05867651,
            0.82778694, -0.56017673,  0.03115715,
            0.05866066,  0.03118697, -0.99779071;
    M = Md.cast<float>();
    Eigen::Vector3f axis = inverseRodriguezFormula<double>(Md).cast<float>();

    QTest::newRow("Failure case 1") << axis.x() << axis.y() << axis.z() << M;*/
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

void TestGeometryLibRotation::testInverseRodriguezNumStable_data() {

    QTest::addColumn<double>("r00");
    QTest::addColumn<double>("r01");
    QTest::addColumn<double>("r02");
    QTest::addColumn<double>("r10");
    QTest::addColumn<double>("r11");
    QTest::addColumn<double>("r12");
    QTest::addColumn<double>("r20");
    QTest::addColumn<double>("r21");
    QTest::addColumn<double>("r22");

    QTest::newRow("Case 1") << 0.9976515596006329 << 0.0001402027638521819 << 0.0684932807438838 <<
                               0.00012947832197639572 << -0.9999999694524941 << 0.00016091083470215484 <<
                               0.06849330024634 << -0.00015174534422002228 << -0.9976515280311443;

}
void TestGeometryLibRotation::testInverseRodriguezNumStable() {

    QFETCH(double, r00);
    QFETCH(double, r01);
    QFETCH(double, r02);
    QFETCH(double, r10);
    QFETCH(double, r11);
    QFETCH(double, r12);
    QFETCH(double, r20);
    QFETCH(double, r21);
    QFETCH(double, r22);

    Eigen::Matrix3d R;
    R << r00, r01, r02, r10, r11, r12, r20, r21, r22;

    Eigen::Vector3d r = inverseRodriguezFormula(R);

    bool allFinite = r.array().allFinite();

    QVERIFY2(allFinite, "Expected coefficients are all supposed to be finite");

}

void TestGeometryLibRotation::testJacobianSO3_data() {

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

    Eigen::Vector3f random = Eigen::Vector3f::Random();
    QTest::newRow("Random 1") << random.x() << random.y() << random.z();
    random = Eigen::Vector3f::Random();
    QTest::newRow("Random 2") << random.x() << random.y() << random.z();
    random = Eigen::Vector3f::Random();
    QTest::newRow("Random 3") << random.x() << random.y() << random.z();

}

void TestGeometryLibRotation::testJacobianSO3() {

    QFETCH(float, rx);
    QFETCH(float, ry);
    QFETCH(float, rz);

    Eigen::Vector3f r(rx, ry, rz);

    Eigen::Matrix3f R = rodriguezFormula<float>(r);
    Eigen::Matrix3f Jr = diffRodriguezLieAlgebra<float>(r);

    constexpr int nRuns = 100;

    constexpr float epsilon = 1e-4;

    for (int i = 0; i < nRuns; i++) {

        Eigen::Vector3f dr = Eigen::Vector3f::Random();
        dr *= epsilon;

        Eigen::Matrix3f RJrdr = R * rodriguezFormula<float>(Jr*dr);

        Eigen::Matrix3f Rdr = rodriguezFormula<float>(r + dr);

        Eigen::Matrix3f delta = Rdr.transpose() * RJrdr;

        Eigen::Vector3f logDelta = inverseRodriguezFormula(delta);

        float error = logDelta.norm();

        QVERIFY2( error < epsilon*1e-2, qPrintable(QString("Large error (%1) detected").arg(error)));
    }

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
		Eigen::Matrix<double,6,1> randomTransform;
		randomTransform << rd(re), rd(re), rd(re), rd(re), rd(re), rd(re);

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
		Eigen::Matrix<double,7,1> randomTransform;
		randomTransform << rd(re), rd(re), rd(re), rd(re), rd(re), rd(re), rd(re);

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

void TestGeometryLibRotation::testRigidTransformInverse() {

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

        RigidBodyTransform directRigid(r, t);
        RigidBodyTransform inverseRigid = directRigid.inverse();

        for (int j = 0; j < nTest; j++) {
            Eigen::Vector3f v(dataGen(re),dataGen(re),dataGen(re));
            Eigen::Vector3f tmp = directRigid*v;

            if (tmp.array().isInf().any() or tmp.array().isNaN().any()) {
                continue;
            }

            Eigen::Vector3f t = inverseRigid*(tmp);

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


void TestGeometryLibRotation::testRMat2EulerRad() {

    Eigen::Vector3d r0xyz = rMat2eulerRadxyz<double>(Eigen::Matrix3d::Identity());
    Eigen::Vector3d r0zyx = rMat2eulerRadzyx<double>(Eigen::Matrix3d::Identity());

    float epsilon = 1e-5;

    for (int i = 0; i < 3; i++) {

        float target = 0;

        float mismatch = std::abs(r0xyz(i) - target);
        QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a 0 angle rotation order xyz (mismatch = %1)").arg(mismatch)));

        mismatch = std::abs(r0zyx(i) - target);
        QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a 0 angle rotation order zyx (mismatch = %1)").arg(mismatch)));
    }

    for (int axis = 0; axis < 3; axis++) {
        std::array<double,3> angles = {0,0,0};
        angles[axis] = M_PI_2;
        Eigen::Matrix3d R = eulerRadXYZToRotation<double>(angles[0],angles[1],angles[2]);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                QVERIFY2(std::isfinite(R(i,j)), "Error in input rotation matrix");
            }
        }

        Eigen::Vector3d rxyz = rMat2eulerRadxyz<double>(R);
        Eigen::Vector3d rzyx = rMat2eulerRadzyx<double>(R);

        for (int i = 0; i < 3; i++) {

            float target = angles[i];

            float mismatch = std::abs(rxyz(i) - target);
            QVERIFY2(mismatch <= epsilon,
                     qPrintable(QString("Error when reconstructing a PI/2 angle rotation order xyz (axis = %1, mismatch = %2)").arg(axis).arg(mismatch)));

            mismatch = std::abs(rzyx(i) - target);
            QVERIFY2(mismatch <= epsilon,
                     qPrintable(QString("Error when reconstructing a PI/2 angle rotation order zxy (axis = %1, mismatch = %2)").arg(axis).arg(mismatch)));
        }

    }

    std::array<double, 3> targets{M_PI_2/4,M_PI_2/3,M_PI_2/2};
    Eigen::Matrix3d RStrangeXYZ = eulerRadXYZToRotation<double>(targets[0], targets[1], targets[2]);
    Eigen::Matrix3d RStrangeZYX = eulerRadZYXToRotation<double>(targets[0], targets[1], targets[2]);

    Eigen::Vector3d rStrangeXYZ = rMat2eulerRadxyz<double>(RStrangeXYZ);
    Eigen::Vector3d rStrangeZYX = rMat2eulerRadzyx<double>(RStrangeZYX);

    for (int i = 0; i < 3; i++) {

        float target = targets[i];

        float mismatch = std::abs(rStrangeXYZ(i) - target);
        QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a 0 angle rotation order xyz (mismatch = %1)").arg(mismatch)));

        mismatch = std::abs(rStrangeZYX(i) - target);
        QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a 0 angle rotation order zyx (mismatch = %1)").arg(mismatch)));
    }

    constexpr int nRepeats = 100;

    std::uniform_real_distribution<double> rd(-M_PI, M_PI);

    for (int i = 0; i < nRepeats; i++) {

        Eigen::Matrix3d Rinitial = eulerRadXYZToRotation<double>(rd(re), rd(re), rd(re));

        Eigen::Vector3d rXYZ = rMat2eulerRadxyz<double>(Rinitial);
        Eigen::Vector3d rZYX = rMat2eulerRadzyx<double>(Rinitial);

        Eigen::Matrix3d Rxyz = eulerRadXYZToRotation<double>(rXYZ[0],rXYZ[1],rXYZ[2]);
        Eigen::Matrix3d Rzyx = eulerRadZYXToRotation<double>(rZYX[0],rZYX[1],rZYX[2]);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {


                float mismatch = std::abs(Rxyz(i,j) - Rinitial(i,j));
                QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a random rotation order xyz (mismatch = %1)").arg(mismatch)));

                mismatch = std::abs(Rzyx(i,j) - Rinitial(i,j));
                QVERIFY2(mismatch <= epsilon, qPrintable(QString("Error when reconstructing a random rotation order zyx (mismatch = %1)").arg(mismatch)));
            }
        }

    }
}

void TestGeometryLibRotation::testRAxis2Quat2RAxis_data() {

    QTest::addColumn<float>("rx");
    QTest::addColumn<float>("ry");
    QTest::addColumn<float>("rz");

    QTest::newRow("Identity") << 0.0f << 0.0f << 0.0f;

    QTest::newRow("Small") << 1.56e-8f << 5.32e-8f << -2.95e-8f;

    QTest::newRow("x axis one") << 1.0f << 0.0f << 0.0f;
    QTest::newRow("y axis one") << 0.0f << 1.0f << 0.0f;
    QTest::newRow("z axis one") << 0.0f << 0.0f << 1.0f;

    QTest::newRow("x axis pi") << static_cast<float>(M_PI) << 0.0f << 0.0f;
    QTest::newRow("y axis pi") << 0.0f << static_cast<float>(M_PI) << 0.0f;
    QTest::newRow("z axis pi") << 0.0f << 0.0f << static_cast<float>(M_PI);

    std::uniform_real_distribution<float> dataGen(-2, 2);

    QTest::newRow("Random 1") << dataGen(re) << dataGen(re) << dataGen(re);
    QTest::newRow("Random 2") << dataGen(re) << dataGen(re) << dataGen(re);
    QTest::newRow("Random 3") << dataGen(re) << dataGen(re) << dataGen(re);

}
void TestGeometryLibRotation::testRAxis2Quat2RAxis() {

    QFETCH(float, rx);
    QFETCH(float, ry);
    QFETCH(float, rz);

    Eigen::Vector3d rAxis(rx, ry, rz);

    Eigen::Matrix3f rot = rodriguezFormula(rAxis).cast<float>();

    Eigen::Quaterniond quat = axisAngleToQuaternion(rAxis);

    Eigen::Matrix3f rotBack  = rodriguezFormula(quaternionToAxisAngle(quat)).cast<float>();


    float epsilon = 1e-5;
    float meanError = (rot.transpose()*rotBack - Eigen::Matrix3f::Identity()).array().abs().mean();

    QVERIFY(meanError < epsilon);

    constexpr int nTest = 42;

    std::uniform_real_distribution<double> dataGen(-10, 10);

    for (int i = 0; i < nTest; i++) {
        Eigen::Vector3d vec(dataGen(re),dataGen(re),dataGen(re));

        Eigen::Vector3f rotatedAxis = (angleAxisRotate(rAxis, vec)).cast<float>();
        Eigen::Vector3f rotatedQuaternion = (quat*vec).cast<float>();

        for (int d = 0; d < 3; d++) {

            float misalignement = std::abs(rotatedQuaternion[d] - rotatedAxis[d]);
            QVERIFY(misalignement < epsilon);
        }
    }


}

void TestGeometryLibRotation::testRAxis2QuatDiff() {

    constexpr double delta = 1e-8;
    constexpr float tol = 1e-5;

    constexpr int nTest = 42;


    std::uniform_real_distribution<double> dataGen(-M_PI, M_PI);

    for (int i = 0; i < nTest; i++) {
        Eigen::Vector3d rAxis(dataGen(re),dataGen(re),dataGen(re));

        Eigen::Matrix<double, 4, 3> NumJac;

        for (int d = 0; d < 3; d++) {

            Eigen::Vector3d deltaVec = Eigen::Vector3d::Zero();
            deltaVec[d] = delta;

            NumJac.col(d) = (axisAngleToQuaternion<double>(rAxis+deltaVec).coeffs() - axisAngleToQuaternion<double>(rAxis-deltaVec).coeffs())/(2*delta);
        }


        Eigen::Matrix<double, 4, 3> AnalJac = diffAxisAngleToQuaternion(rAxis);

        Eigen::Array<double, 4, 3> error = (AnalJac - NumJac).array().abs();
        double maxError = error.maxCoeff();

        QVERIFY(maxError < tol);

    }

    constexpr float smoltol = 1e-7;


    std::uniform_real_distribution<double> smolDataGen(-3e-6, 3e-6);

    for (int i = 0; i < nTest; i++) {
        Eigen::Vector3d rAxis(smolDataGen(re),smolDataGen(re),smolDataGen(re));

        Eigen::Matrix<double, 4, 3> NumJac;

        for (int d = 0; d < 3; d++) {

            Eigen::Vector3d deltaVec = Eigen::Vector3d::Zero();
            deltaVec[d] = delta;

            NumJac.col(d) = (axisAngleToQuaternion<double>(rAxis+deltaVec).coeffs() - axisAngleToQuaternion<double>(rAxis-deltaVec).coeffs())/(2*delta);
        }

        if (NumJac(0,0) > 0.8) { //this mean numerical error
            continue;
        }

        Eigen::Matrix<double, 4, 3> AnalJac = diffAxisAngleToQuaternion(rAxis);

        Eigen::Array<double, 4, 3> error = (AnalJac - NumJac).array().abs();
        double maxError = error.maxCoeff();

        QVERIFY(maxError < smoltol);

    }

}

void TestGeometryLibRotation::testSensorFrameConversions() {

    using FRD = AxisSystemDefintion<Front, Right, Down>;
    using LFD = AxisSystemDefintion<Left, Front, Down>;

    Eigen::Matrix3i frame1toframe2 = getSensorFrameConversion<FRD,LFD>();

    QCOMPARE(frame1toframe2(0,0),0);
    QCOMPARE(frame1toframe2(1,0),1);
    QCOMPARE(frame1toframe2(2,0),0);
    QCOMPARE(frame1toframe2(0,1),-1);
    QCOMPARE(frame1toframe2(1,1),0);
    QCOMPARE(frame1toframe2(2,1),0);
    QCOMPARE(frame1toframe2(0,2),0);
    QCOMPARE(frame1toframe2(1,2),0);
    QCOMPARE(frame1toframe2(2,2),1);

    QCOMPARE(frame1toframe2.determinant(),1);

    frame1toframe2 = getSensorFrameConversion<LFD,FRD>();

    QCOMPARE(frame1toframe2(0,0),0);
    QCOMPARE(frame1toframe2(1,0),-1);
    QCOMPARE(frame1toframe2(2,0),0);
    QCOMPARE(frame1toframe2(0,1),1);
    QCOMPARE(frame1toframe2(1,1),0);
    QCOMPARE(frame1toframe2(2,1),0);
    QCOMPARE(frame1toframe2(0,2),0);
    QCOMPARE(frame1toframe2(1,2),0);
    QCOMPARE(frame1toframe2(2,2),1);

    QCOMPARE(frame1toframe2.determinant(),1);

}

void TestGeometryLibRotation::testRigidBodyTransformInterpolationOnManifold() {

    constexpr int nRuns = 10;

    std::uniform_real_distribution<float> dataGen(-2, 2);

    for (int i = 0; i < nRuns; i++) {

        Eigen::Vector3d r1;
        r1 << dataGen(re), dataGen(re), dataGen(re);
        Eigen::Vector3d r2;
        r2 << dataGen(re), dataGen(re), dataGen(re);

        Eigen::Vector3d t1;
        t1 << dataGen(re), dataGen(re), dataGen(re);
        Eigen::Vector3d t2;
        t2 << dataGen(re), dataGen(re), dataGen(re);

        RigidBodyTransform<double> T1(r1,t1);
        RigidBodyTransform<double> T2(r2,t2);

        RigidBodyTransform<double> T0(Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero());

        RigidBodyTransform<double> interpolated1 = interpolateRigidBodyTransformOnManifold<double>(1, T1, 0, T2);
        RigidBodyTransform<double> interpolated2 = interpolateRigidBodyTransformOnManifold<double>(0, T1, 1, T2);

        for (int i = 0; i < 3; i++) {

            QCOMPARE(interpolated1.r[i], T1.r[i]);
            QCOMPARE(interpolated1.t[i], T1.t[i]);

            QCOMPARE(interpolated2.r[i], T2.r[i]);
            QCOMPARE(interpolated2.t[i], T2.t[i]);
        }

        RigidBodyTransform<double> interpolated025 = interpolateRigidBodyTransformOnManifold<double>(0.25, T1, 0.75, T2);
        RigidBodyTransform<double> interpolated050 = interpolateRigidBodyTransformOnManifold<double>(0.5, T1, 0.5, T2);
        RigidBodyTransform<double> interpolated075 = interpolateRigidBodyTransformOnManifold<double>(0.75, T1, 0.25, T2);

        RigidBodyTransform<double> interpolated025d = interpolateRigidBodyTransformOnManifold<double>(0.25, T0, 0.75, T2*T1.inverse());
        RigidBodyTransform<double> interpolated050d = interpolateRigidBodyTransformOnManifold<double>(0.5, T0, 0.5, T2*T1.inverse());
        RigidBodyTransform<double> interpolated075d = interpolateRigidBodyTransformOnManifold<double>(0.75, T0, 0.25, T2*T1.inverse());

        RigidBodyTransform<double> interpolated025e = interpolated025d*T1;
        RigidBodyTransform<double> interpolated050e = interpolated050d*T1;
        RigidBodyTransform<double> interpolated075e = interpolated075d*T1;

        for (int i = 0; i < 3; i++) {

            QVERIFY(std::fabs(interpolated025e.r[i] - interpolated025.r[i]) < 1e-8);
            QVERIFY(std::fabs(interpolated050e.r[i] - interpolated050.r[i]) < 1e-8);
            QVERIFY(std::fabs(interpolated075e.r[i] - interpolated075.r[i]) < 1e-8);

            QVERIFY(std::fabs(interpolated025e.t[i] - interpolated025.t[i]) < 1e-8);
            QVERIFY(std::fabs(interpolated050e.t[i] - interpolated050.t[i]) < 1e-8);
            QVERIFY(std::fabs(interpolated075e.t[i] - interpolated075.t[i]) < 1e-8);
        }

    }

}

QTEST_MAIN(TestGeometryLibRotation)
#include "testRotations.moc"
