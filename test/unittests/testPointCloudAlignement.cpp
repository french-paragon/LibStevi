#include <QtTest/QtTest>

#include "geometry/pointcloudalignment.h"

#include <random>
#include <optional>
#include <iostream>

using namespace StereoVision::Geometry;

std::default_random_engine rd;
std::default_random_engine engine(rd());
std::uniform_real_distribution<float> uDist(0.1, 10);

AffineTransform<float> generateRandomTransform() {
	AffineTransform<float> T;

	T.R.setRandom();
	T.t.setRandom();

	return T;
}

AffineTransform<float> generateShapePreservingRandomTransform() {
	ShapePreservingTransform<float> T;

	T.s = uDist(engine);
	T.r.setRandom();
	T.t.setRandom();

	return T.toAffineTransform();
}

AffineTransform<float> generateRigidRandomTransform() {
	ShapePreservingTransform<float> T;

	T.s = 1;
	T.r.setRandom();
	T.t.setRandom();

	return T.toAffineTransform();
}


AffineTransform<float> generateScalingRandomTransform() {
	ShapePreservingTransform<float> T;

	T.s = uDist(engine);
	T.r.setZero();
	T.t.setZero();

	return T.toAffineTransform();
}


AffineTransform<float> generateTranslateRandomTransform() {
	ShapePreservingTransform<float> T;

	T.s = 1.;
	T.r.setZero();
	T.t.setRandom();

	return T.toAffineTransform();
}


AffineTransform<float> generateRotationRandomTransform() {
	ShapePreservingTransform<float> T;

	T.s = 1.;
	T.r.setRandom();
	T.t.setZero();

	return T.toAffineTransform();
}

void generateObs(int nPts,
				 int nObsperPt,
				 Eigen::VectorXf & obs,
				 Eigen::Matrix3Xf & pts,
				 std::vector<int> & idxs,
				 std::vector<Axis> & coordinate,
				 AffineTransform<float> const& T) {

	std::random_device rd;
	std::default_random_engine engine(rd());
	std::uniform_int_distribution<int> uDist(0, 2);

	pts.setRandom(3, nPts);

	obs.resize(nPts*nObsperPt);
	idxs.clear();
	idxs.reserve(nPts*nObsperPt);
	coordinate.clear();
	coordinate.reserve(nPts*nObsperPt);

	Eigen::Matrix3Xf tPts = T*pts;

	for (int i = 0; i < nPts; i++) {

		int random_id = uDist(engine);

		if (nObsperPt == 1) {

			Axis cAx = (random_id == 0) ? Axis::X : ((random_id == 1) ? Axis::Y : Axis::Z);

			obs[i] = tPts(random_id, i);
			idxs.push_back(i);
			coordinate.push_back(cAx);
		} else if (nObsperPt == 2) {

			int rId1 = (random_id == 0) ? 1 : 0;
			int rId2 = (random_id == 2) ? 1 : 2;

			Axis cA1 = (random_id == 0) ? Axis::Y : Axis::X;
			Axis cA2 = (random_id == 2) ? Axis::Y : Axis::Z;

			obs[2*i] = tPts(rId1, i);
			idxs.push_back(i);
			coordinate.push_back(cA1);

			obs[2*i+1] = tPts(rId2, i);
			idxs.push_back(i);
			coordinate.push_back(cA2);
		} else {

			obs[3*i] = tPts(0, i);
			idxs.push_back(i);
			coordinate.push_back(Axis::X);

			obs[3*i+1] = tPts(1, i);
			idxs.push_back(i);
			coordinate.push_back(Axis::Y);

			obs[3*i+2] = tPts(2, i);
			idxs.push_back(i);
			coordinate.push_back(Axis::Z);
		}

	}

}

class TestPointCloudAlignement: public QObject
{
	Q_OBJECT
private Q_SLOTS:
	void initTestCase();
	void testAffineMap_data();
	void testAffineMap();

	void testQuasiShapePreservingMap_data();
	void testQuasiShapePreservingMap();

	void testQuasiRigidMap_data();
	void testQuasiRigidMap();

	void testInitShapePreservingMap_data();
	void testInitShapePreservingMap();

	void testShapePreservingMap_data();
	void testShapePreservingMap();

	void testExtractScaleMap_data();
	void testExtractScaleMap();

	void testExtractTranslationMap_data();
	void testExtractTranslationMap();

	void testExtractRotationMap_data();
	void testExtractRotationMap();

	void testExtractShapePreservingMap();
};

void TestPointCloudAlignement::initTestCase() {
	srand(time(nullptr));
}

void TestPointCloudAlignement::testAffineMap_data() {

	QTest::addColumn<int>("nPts");
	QTest::addColumn<int>("nObsPerPoints");

	QTest::newRow("Minimal") << 1 << 1;
	QTest::newRow("Underdetermined small") << 3 << 1;
	QTest::newRow("Underdetermined big") << 3 << 2;
	QTest::newRow("Just set dense") << 4 << 3;
	QTest::newRow("Just set sparse") << 12 << 1;
	QTest::newRow("Overdetermined") << 12 << 3;
}

void TestPointCloudAlignement::testAffineMap() {

	QFETCH(int, nPts);
	QFETCH(int, nObsPerPoints);

	Eigen::VectorXf obs;
	Eigen::Matrix3Xf pts;
	std::vector<int> idxs;
	std::vector<Axis> coordinate;
	AffineTransform<float> T = generateRandomTransform();

	generateObs(nPts, nObsPerPoints, obs, pts, idxs, coordinate, T);

	AffineTransform F = estimateAffineMap(obs, pts, idxs, coordinate);

	Eigen::Matrix3Xf tTrue = T*pts;
	Eigen::Matrix3Xf tFound = F*pts;

	for (int i = 0; i < nPts*nObsPerPoints; i++) {
		int row = (coordinate[i] == Axis::X) ? 0 : ((coordinate[i] == Axis::Y) ? 1 : 2);
		float vTrue = tTrue(row,idxs[i]);
		float vFound = tFound(row,idxs[i]);

		float error = fabs(vTrue - vFound);
		QVERIFY2(error < 1e-3, qPrintable(QString("Misaligned reconstructed coordinates (%1)").arg(error)));
	}

}

void TestPointCloudAlignement::testQuasiShapePreservingMap_data() {

	QTest::addColumn<int>("nPts");
	QTest::addColumn<int>("nObsPerPoints");

	QTest::newRow("Minimal") << 1 << 1;
	QTest::newRow("Underdetermined small") << 3 << 1;
	QTest::newRow("Underdetermined big") << 3 << 2;
	QTest::newRow("Just set dense") << 4 << 3;
	QTest::newRow("Just set sparse") << 12 << 1;
	QTest::newRow("Overdetermined") << 12 << 3;

}
void TestPointCloudAlignement::testQuasiShapePreservingMap() {

	QFETCH(int, nPts);
	QFETCH(int, nObsPerPoints);

	Eigen::VectorXf obs;
	Eigen::Matrix3Xf pts;
	std::vector<int> idxs;
	std::vector<Axis> coordinate;
	AffineTransform<float> T = generateShapePreservingRandomTransform();

	generateObs(nPts, nObsPerPoints, obs, pts, idxs, coordinate, T);

	AffineTransform F = estimateQuasiShapePreservingMap(obs, pts, idxs, coordinate, 2e-1, nullptr, 1e-7, 500, false);

	Eigen::Matrix3Xf tTrue = T*pts;
	Eigen::Matrix3Xf tFound = F*pts;

	for (int i = 0; i < nPts*nObsPerPoints; i++) {
		int row = (coordinate[i] == Axis::X) ? 0 : ((coordinate[i] == Axis::Y) ? 1 : 2);
		float vTrue = tTrue(row,idxs[i]);
		float vFound = tFound(row,idxs[i]);

		float error = fabs(vTrue - vFound);
		QVERIFY2(error < 1e-3, qPrintable(QString("Misaligned recontructed coordinates (%1)").arg(error)));
	}

	Eigen::Matrix3f delta = F.R.transpose()*F.R;

	float diagMismastch = (delta.diagonal().maxCoeff() - delta.diagonal().minCoeff())/delta.diagonal().minCoeff();
	QVERIFY2(diagMismastch < 1e-3, qPrintable(QString("Rt*R diagonal is not uniform (relative error %1)").arg(diagMismastch)));

	float mismatch = (delta/delta.diagonal().mean() - Eigen::Matrix3f::Identity()).norm();
	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Rt*R not a diagonal matrix (norm (R.t*R/mean(diag(R.t*R)) - I) = %1)").arg(mismatch)));

}

void TestPointCloudAlignement::testQuasiRigidMap_data() {

	QTest::addColumn<int>("nPts");
	QTest::addColumn<int>("nObsPerPoints");

	QTest::newRow("Minimal") << 1 << 1;
	QTest::newRow("Underdetermined small") << 3 << 1;
	QTest::newRow("Underdetermined big") << 3 << 2;
	QTest::newRow("Just set dense") << 4 << 3;
	QTest::newRow("Just set sparse") << 12 << 1;
	QTest::newRow("Overdetermined") << 12 << 3;

}
void TestPointCloudAlignement::testQuasiRigidMap() {

	QFETCH(int, nPts);
	QFETCH(int, nObsPerPoints);

	Eigen::VectorXf obs;
	Eigen::Matrix3Xf pts;
	std::vector<int> idxs;
	std::vector<Axis> coordinate;
	AffineTransform<float> T = generateRigidRandomTransform();

	generateObs(nPts, nObsPerPoints, obs, pts, idxs, coordinate, T);

	AffineTransform F = estimateQuasiRigidMap(obs, pts, idxs, coordinate, 2e-1, nullptr, 1e-7, 500, false);

	Eigen::Matrix3Xf tTrue = T*pts;
	Eigen::Matrix3Xf tFound = F*pts;

	for (int i = 0; i < nPts*nObsPerPoints; i++) {
		int row = (coordinate[i] == Axis::X) ? 0 : ((coordinate[i] == Axis::Y) ? 1 : 2);
		float vTrue = tTrue(row,idxs[i]);
		float vFound = tFound(row,idxs[i]);

		float error = fabs(vTrue - vFound);
		QVERIFY2(error < 1e-3, qPrintable(QString("Misaligned recontructed coordinates (%1)").arg(error)));
	}

	Eigen::Matrix3f delta = F.R.transpose()*F.R;

	float diagMismastch = (delta.diagonal().maxCoeff() - delta.diagonal().minCoeff())/delta.diagonal().minCoeff();
	QVERIFY2(diagMismastch < 1e-3, qPrintable(QString("Rt*R diagonal is not uniform (relative error %1)").arg(diagMismastch)));

	float mismatch = (delta/delta.diagonal().mean() - Eigen::Matrix3f::Identity()).norm();
	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Rt*R not a diagonal matrix (norm (R.t*R/mean(diag(R.t*R)) - I) = %1)").arg(mismatch)));

}

void TestPointCloudAlignement::testInitShapePreservingMap_data() {

	QTest::addColumn<int>("nPts");

	QTest::newRow("Just set") << 3;
	QTest::newRow("Overdetermined") << 12;
}
void TestPointCloudAlignement::testInitShapePreservingMap() {

	QFETCH(int, nPts);

	Eigen::VectorXf obs;
	Eigen::Matrix3Xf pts;
	std::vector<int> idxs;
	std::vector<Axis> coordinate;
	AffineTransform<float> T = generateShapePreservingRandomTransform();

	generateObs(nPts, 3, obs, pts, idxs, coordinate, T);

	std::optional<ShapePreservingTransform<float>> out = initShapePreservingMapEstimate(obs,
																				 pts,
																				 idxs,
																				 coordinate);

	QVERIFY2(out.has_value(), "Missing output value in initializer solution.");
	AffineTransform<float> F = out.value().toAffineTransform();

	Eigen::Matrix3Xf tTrue = T*pts;
	Eigen::Matrix3Xf tFound = F*pts;

	for (int i = 0; i < nPts*2; i++) {
		int row = (coordinate[i] == Axis::X) ? 0 : ((coordinate[i] == Axis::Y) ? 1 : 2);
		float vTrue = tTrue(row,idxs[i]);
		float vFound = tFound(row,idxs[i]);

		float error = fabs(vTrue - vFound);
		QVERIFY2(error < 1e-3, qPrintable(QString("Misaligned recontructed coordinates (%1)").arg(error)));
	}

	Eigen::Matrix3f delta = F.R.transpose()*F.R;

	float diagMismastch = (delta.diagonal().maxCoeff() - delta.diagonal().minCoeff())/delta.diagonal().minCoeff();
	QVERIFY2(diagMismastch < 1e-3, qPrintable(QString("Rt*R diagonal is not uniform (relative error %1)").arg(diagMismastch)));

	float mismatch = (delta/delta.diagonal().mean() - Eigen::Matrix3f::Identity()).norm();
	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Rt*R not a diagonal matrix (norm (R.t*R/mean(diag(R.t*R)) - I) = %1)").arg(mismatch)));
}

void TestPointCloudAlignement::testShapePreservingMap_data() {

	QTest::addColumn<int>("nPts");
	QTest::addColumn<int>("nObsPerPoints");

	QTest::newRow("Minimal") << 1 << 1;
	QTest::newRow("Underdetermined small") << 3 << 1;
	QTest::newRow("Underdetermined big") << 3 << 2;
	QTest::newRow("Just set dense") << 4 << 3;
	QTest::newRow("Just set sparse") << 12 << 1;
	QTest::newRow("Overdetermined") << 12 << 3;
}
void TestPointCloudAlignement::testShapePreservingMap() {

	QFETCH(int, nPts);
	QFETCH(int, nObsPerPoints);

	Eigen::VectorXf obs;
	Eigen::Matrix3Xf pts;
	std::vector<int> idxs;
	std::vector<Axis> coordinate;
	AffineTransform<float> T = generateShapePreservingRandomTransform();

	generateObs(nPts, nObsPerPoints, obs, pts, idxs, coordinate, T);

	IterativeTermination endStatus;
	AffineTransform F = estimateShapePreservingMap(obs,
												   pts,
												   idxs,
												   coordinate,
												   &endStatus,
												   5000,
												   1e-8,
												   3e-1,
												   1e-1).toAffineTransform();

	if (endStatus == IterativeTermination::Converged) {
		std::cout << "Estimation converged" << std::endl;
	} else if (endStatus == IterativeTermination::MaxStepReached) {
		std::cout << "Estimation reach max number of iterations" << std::endl;
	}
	QVERIFY2(endStatus == IterativeTermination::Converged or endStatus == IterativeTermination::MaxStepReached,
			 qPrintable(QString("Estimation failed without finishing !")));

	Eigen::Matrix3Xf tTrue = T*pts;
	Eigen::Matrix3Xf tFound = F*pts;

	for (int i = 0; i < nPts*nObsPerPoints; i++) {
		int row = (coordinate[i] == Axis::X) ? 0 : ((coordinate[i] == Axis::Y) ? 1 : 2);
		float vTrue = tTrue(row,idxs[i]);
		float vFound = tFound(row,idxs[i]);

		float error = fabs(vTrue - vFound);
		QVERIFY2(error < 1e-3, qPrintable(QString("Misaligned recontructed coordinates (%1)").arg(error)));
	}

	Eigen::Matrix3f delta = F.R.transpose()*F.R;

	float diagMismastch = (delta.diagonal().maxCoeff() - delta.diagonal().minCoeff())/delta.diagonal().minCoeff();
	QVERIFY2(diagMismastch < 1e-3, qPrintable(QString("Rt*R diagonal is not uniform (relative error %1)").arg(diagMismastch)));

	float mismatch = (delta/delta.diagonal().mean() - Eigen::Matrix3f::Identity()).norm();
	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Rt*R not a diagonal matrix (norm (R.t*R/mean(diag(R.t*R)) - I) = %1)").arg(mismatch)));
}

void TestPointCloudAlignement::testExtractScaleMap_data() {

	QTest::addColumn<int>("nPts");
	QTest::addColumn<int>("nObsPerPoints");

	QTest::newRow("Minimal") << 1 << 1;
	QTest::newRow("Underdetermined small") << 3 << 1;
	QTest::newRow("Underdetermined big") << 3 << 2;
	QTest::newRow("Just set dense") << 4 << 3;
	QTest::newRow("Just set sparse") << 12 << 1;
	QTest::newRow("Overdetermined") << 12 << 3;
}

void TestPointCloudAlignement::testExtractScaleMap() {

	QFETCH(int, nPts);
	QFETCH(int, nObsPerPoints);

	Eigen::VectorXf obs;
	Eigen::Matrix3Xf pts;
	std::vector<int> idxs;
	std::vector<Axis> coordinate;
	AffineTransform<float> T = generateScalingRandomTransform();

	generateObs(nPts, nObsPerPoints, obs, pts, idxs, coordinate, T);

	AffineTransform F = estimateScaleMap(obs, pts, idxs, coordinate, nullptr, false).toAffineTransform();

	Eigen::Matrix3Xf tTrue = T*pts;
	Eigen::Matrix3Xf tFound = F*pts;

	for (int i = 0; i < nPts*nObsPerPoints; i++) {
		int row = (coordinate[i] == Axis::X) ? 0 : ((coordinate[i] == Axis::Y) ? 1 : 2);
		float vTrue = tTrue(row,idxs[i]);
		float vFound = tFound(row,idxs[i]);

		float error = fabs(vTrue - vFound);
		QVERIFY2(error < 1e-3, qPrintable(QString("Misaligned recontructed coordinates (%1)").arg(error)));
	}

	Eigen::Matrix3f delta = F.R.transpose()*F.R;

	float diagMismastch = (delta.diagonal().maxCoeff() - delta.diagonal().minCoeff())/delta.diagonal().minCoeff();
	QVERIFY2(diagMismastch < 1e-3, qPrintable(QString("Rt*R diagonal is not uniform (relative error %1)").arg(diagMismastch)));

	float mismatch = (delta/delta.diagonal().mean() - Eigen::Matrix3f::Identity()).norm();
	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Rt*R not a diagonal matrix (norm (R.t*R/mean(diag(R.t*R)) - I) = %1)").arg(mismatch)));
}

void TestPointCloudAlignement::testExtractTranslationMap_data() {

	QTest::addColumn<int>("nPts");
	QTest::addColumn<int>("nObsPerPoints");

	QTest::newRow("Minimal") << 1 << 1;
	QTest::newRow("Underdetermined small") << 3 << 1;
	QTest::newRow("Underdetermined big") << 3 << 2;
	QTest::newRow("Just set dense") << 4 << 3;
	QTest::newRow("Just set sparse") << 12 << 1;
	QTest::newRow("Overdetermined") << 12 << 3;
}

void TestPointCloudAlignement::testExtractTranslationMap() {

	QFETCH(int, nPts);
	QFETCH(int, nObsPerPoints);

	Eigen::VectorXf obs;
	Eigen::Matrix3Xf pts;
	std::vector<int> idxs;
	std::vector<Axis> coordinate;
	AffineTransform<float> T = generateTranslateRandomTransform();

	generateObs(nPts, nObsPerPoints, obs, pts, idxs, coordinate, T);

	AffineTransform F = estimateTranslationMap(obs, pts, idxs, coordinate, nullptr, false).toAffineTransform();

	Eigen::Matrix3Xf tTrue = T*pts;
	Eigen::Matrix3Xf tFound = F*pts;

	for (int i = 0; i < nPts*nObsPerPoints; i++) {
		int row = (coordinate[i] == Axis::X) ? 0 : ((coordinate[i] == Axis::Y) ? 1 : 2);
		float vTrue = tTrue(row,idxs[i]);
		float vFound = tFound(row,idxs[i]);

		float error = fabs(vTrue - vFound);
		QVERIFY2(error < 1e-3, qPrintable(QString("Misaligned recontructed coordinates (%1)").arg(error)));
	}

	Eigen::Matrix3f delta = F.R.transpose()*F.R;

	float diagMismastch = (delta.diagonal().maxCoeff() - delta.diagonal().minCoeff())/delta.diagonal().minCoeff();
	QVERIFY2(diagMismastch < 1e-3, qPrintable(QString("Rt*R diagonal is not uniform (relative error %1)").arg(diagMismastch)));

	float mismatch = (delta/delta.diagonal().mean() - Eigen::Matrix3f::Identity()).norm();
	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Rt*R not a diagonal matrix (norm (R.t*R/mean(diag(R.t*R)) - I) = %1)").arg(mismatch)));
}

void TestPointCloudAlignement::testExtractRotationMap_data() {

	QTest::addColumn<int>("nPts");
	QTest::addColumn<int>("nObsPerPoints");

	QTest::newRow("Minimal") << 1 << 1;
	QTest::newRow("Underdetermined small") << 3 << 1;
	QTest::newRow("Underdetermined big") << 3 << 1;
	QTest::newRow("Just set dense") << 4 << 3;
	QTest::newRow("Just set sparse") << 12 << 1;
	QTest::newRow("Overdetermined") << 12 << 3;
}

void TestPointCloudAlignement::testExtractRotationMap() {

	QFETCH(int, nPts);
	QFETCH(int, nObsPerPoints);

	Eigen::VectorXf obs;
	Eigen::Matrix3Xf pts;
	std::vector<int> idxs;
	std::vector<Axis> coordinate;
	AffineTransform<float> T = generateRotationRandomTransform();

	generateObs(nPts, nObsPerPoints, obs, pts, idxs, coordinate, T);

	AffineTransform F = estimateRotationMap(obs,
											pts,
											idxs,
											coordinate,
											nullptr,
											nullptr,
											false,
											5000,
											1e-8).toAffineTransform();

	Eigen::Matrix3Xf tTrue = T*pts;
	Eigen::Matrix3Xf tFound = F*pts;

	for (int i = 0; i < nPts*nObsPerPoints; i++) {
		int row = (coordinate[i] == Axis::X) ? 0 : ((coordinate[i] == Axis::Y) ? 1 : 2);
		float vTrue = tTrue(row,idxs[i]);
		float vFound = tFound(row,idxs[i]);

		float error = fabs(vTrue - vFound);
		QVERIFY2(error < 1e-3, qPrintable(QString("Misaligned recontructed coordinates (%1)").arg(error)));
	}

	Eigen::Matrix3f delta = F.R.transpose()*F.R;

	float diagMismastch = (delta.diagonal().maxCoeff() - delta.diagonal().minCoeff())/delta.diagonal().minCoeff();
	QVERIFY2(diagMismastch < 1e-3, qPrintable(QString("Rt*R diagonal is not uniform (relative error %1)").arg(diagMismastch)));

	float mismatch = (delta/delta.diagonal().mean() - Eigen::Matrix3f::Identity()).norm();
	QVERIFY2(mismatch < 1e-5, qPrintable(QString("Rt*R not a diagonal matrix (norm (R.t*R/mean(diag(R.t*R)) - I) = %1)").arg(mismatch)));

}

void TestPointCloudAlignement::testExtractShapePreservingMap() {

	for (int i = 0; i < 10; i++) {

		AffineTransform<float> T = generateShapePreservingRandomTransform();
		ShapePreservingTransform F = affine2ShapePreservingMap(T);

		AffineTransform FT = F.toAffineTransform();

		Eigen::Matrix3f delta = FT.R.transpose()*T.R;

		float diagMismastch = (delta.diagonal().maxCoeff() - delta.diagonal().minCoeff())/delta.diagonal().minCoeff();
		QVERIFY2(diagMismastch < 1e-3, qPrintable(QString("FRt*R diagonal is not uniform (relative error %1)").arg(diagMismastch)));

		float mismatch = (delta/delta.diagonal().mean() - Eigen::Matrix3f::Identity()).norm();
		QVERIFY2(mismatch < 1e-5, qPrintable(QString("FRt*R not a diagonal matrix (norm (R.t*R/mean(diag(R.t*R)) - I) = %1)").arg(mismatch)));

	}

}

QTEST_MAIN(TestPointCloudAlignement)
#include "testPointCloudAlignement.moc"
