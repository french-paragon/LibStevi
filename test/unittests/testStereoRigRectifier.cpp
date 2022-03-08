#include <QtTest/QtTest>

#include "geometry/stereorigrectifier.h"

using namespace StereoVision::Geometry;

class TestStereoRigRectifier: public QObject
{
	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testAlignementEstimate();

private:

	std::default_random_engine re;
};

void TestStereoRigRectifier::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}


void TestStereoRigRectifier::testAlignementEstimate() {

	constexpr int nPointsReplicates = 50;

	std::uniform_real_distribution<float> rotDist(-0.1, 0.1);

	Eigen::Vector3f t(5, 0, 0);

	Eigen::Vector3f rCam1;
	rCam1 << rotDist(re), rotDist(re), rotDist(re);

	Eigen::Vector3f rCam2;
	rCam2 << rotDist(re), rotDist(re), rotDist(re);

	Eigen::Matrix3f RCam1 = rodriguezFormula(rCam1);
	Eigen::Matrix3f RCam2 = rodriguezFormula(rCam2);

	Eigen::Matrix3f RCam2toCam1 = RCam1.transpose()*RCam2;
	Eigen::Vector3f tCam2toCam1 = RCam1.transpose()*t;

	ShapePreservingTransform Cam2ToCam1(inverseRodriguezFormula(RCam2toCam1), tCam2toCam1, 1.);

	StereoRigRectifier rectifier(Cam2ToCam1,
								 1,
								 Eigen::Vector2f::Zero(),
								 Eigen::Vector2i::Zero(),
								 std::nullopt,
								 std::nullopt,
								 std::nullopt,
								 1.,
								 Eigen::Vector2f::Zero(),
								 Eigen::Vector2i::Zero(),
								 std::nullopt,
								 std::nullopt,
								 std::nullopt);

	bool ok = rectifier.computeOptimalCamsRots();

	QVERIFY2(ok, "Computation of optimal rotations failed for unknown reasons !");

	Eigen::Vector3f fowardCam1(0,0,1);

	Eigen::Vector3f corrFowardCam1 = rectifier.CorrRCam1()*fowardCam1;
	Eigen::Vector3f corrFowardCam2 = RCam2toCam1*rectifier.CorrRCam2()*fowardCam1;

	QVERIFY2(std::fabs(corrFowardCam1.x() - corrFowardCam2.x()) < 1e-5,
			 qPrintable(QString("Misaligned corrFoward x coord (%1 and %2)").arg(corrFowardCam1.x()).arg(corrFowardCam2.x())));
	QVERIFY2(std::fabs(corrFowardCam1.y() - corrFowardCam2.y()) < 1e-5,
			 qPrintable(QString("Misaligned corrFoward y coord (%1 and %2)").arg(corrFowardCam1.y()).arg(corrFowardCam2.y())));
	QVERIFY2(std::fabs(corrFowardCam1.z() - corrFowardCam2.z()) < 1e-5,
			 qPrintable(QString("Misaligned corrFoward z coord (%1 and %2)").arg(corrFowardCam1.z()).arg(corrFowardCam2.z())));

	Eigen::Vector3f yCam1(0,1,0);

	Eigen::Vector3f corrYCam1 = rectifier.CorrRCam1()*yCam1;
	Eigen::Vector3f corrYCam2 = RCam2toCam1*rectifier.CorrRCam2()*yCam1;

	QVERIFY2(std::fabs(corrYCam1.x() - corrYCam2.x()) < 1e-5,
			 qPrintable(QString("Misaligned corrY x coord (%1 and %2)").arg(corrYCam1.x()).arg(corrYCam2.x())));
	QVERIFY2(std::fabs(corrYCam1.y() - corrYCam2.y()) < 1e-5,
			 qPrintable(QString("Misaligned corrY y coord (%1 and %2)").arg(corrYCam1.y()).arg(corrYCam2.y())));
	QVERIFY2(std::fabs(corrYCam1.z() - corrYCam2.z()) < 1e-5,
			 qPrintable(QString("Misaligned corrY z coord (%1 and %2)").arg(corrYCam1.z()).arg(corrYCam2.z())));

	Eigen::Vector3f xCam1(1,0,0);

	Eigen::Vector3f corrXCam1 = rectifier.CorrRCam1()*xCam1;
	Eigen::Vector3f corrXCam2 = RCam2toCam1*rectifier.CorrRCam2()*xCam1;
	Eigen::Vector3f normalizedDir = tCam2toCam1;
	normalizedDir.normalize();

	QVERIFY2(std::fabs(corrXCam1.x() - corrXCam2.x()) < 1e-5,
			 qPrintable(QString("Misaligned corrX x coord (%1 and %2)").arg(corrXCam1.x()).arg(corrXCam2.x())));
	QVERIFY2(std::fabs(corrXCam1.y() - corrXCam2.y()) < 1e-5,
			 qPrintable(QString("Misaligned corrX y coord (%1 and %2)").arg(corrXCam1.y()).arg(corrXCam2.y())));
	QVERIFY2(std::fabs(corrXCam1.z() - corrXCam2.z()) < 1e-5,
			 qPrintable(QString("Misaligned corrX z coord (%1 and %2)").arg(corrXCam1.z()).arg(corrXCam2.z())));

	QVERIFY2(std::fabs(corrXCam1.x() - normalizedDir.x()) < 1e-5,
			 qPrintable(QString("Misaligned corrX x coord (%1 and %2)").arg(corrXCam1.x()).arg(normalizedDir.x())));
	QVERIFY2(std::fabs(corrXCam1.y() - normalizedDir.y()) < 1e-5,
			 qPrintable(QString("Misaligned corrX y coord (%1 and %2)").arg(corrXCam1.y()).arg(normalizedDir.y())));
	QVERIFY2(std::fabs(corrXCam1.z() - normalizedDir.z()) < 1e-5,
			 qPrintable(QString("Misaligned corrX z coord (%1 and %2)").arg(corrXCam1.z()).arg(normalizedDir.z())));

	for (int i = 0; i < nPointsReplicates; i++) {

		std::uniform_real_distribution<float> ptDist(-3, 3);

		Eigen::Vector3f ptCam2(0, -2.5, 20);
		ptCam2.x() += ptDist(re);
		ptCam2.y() += ptDist(re);
		ptCam2.z() += ptDist(re);

		Eigen::Vector2f ptNormCam2 = ptCam2.block<2,1>(0,0)/ptCam2.z();

		Eigen::Vector3f ptCam1 = RCam2toCam1*ptCam2 + tCam2toCam1;

		Eigen::Vector2f ptNormCam1 = ptCam1.block<2,1>(0,0)/ptCam1.z();

		Eigen::Vector2f ptCorrectedCam1 = rectifier.computeForwardVec(ptNormCam1,
																	  Eigen::Vector2f::Zero(),
																	  1,
																	  rectifier.CorrRCam1());

		Eigen::Vector2f ptCorrectedCam2 = rectifier.computeForwardVec(ptNormCam2,
																	  Eigen::Vector2f::Zero(),
																	  1,
																	  rectifier.CorrRCam2());

		QVERIFY2(std::fabs(ptCorrectedCam1.y() - ptCorrectedCam2.y()) < 1e-5,
				 qPrintable(QString("Misaligned corrected points y coord (%1 and %2)").arg(ptCorrectedCam1.y()).arg(ptCorrectedCam2.y())));

	}
}

QTEST_MAIN(TestStereoRigRectifier)
#include "testStereoRigRectifier.moc"
