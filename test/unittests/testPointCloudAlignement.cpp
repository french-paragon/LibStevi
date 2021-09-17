#include <QtTest/QtTest>

#include "geometry/pointcloudalignment.h"

#include <random>

using namespace StereoVision::Geometry;

std::default_random_engine rd;
std::default_random_engine engine(rd());
std::uniform_real_distribution<float> uDist(0.1, 10);

AffineTransform generateRandomTransform() {
	AffineTransform T;

	T.R.setRandom();
	T.t.setRandom();

	return T;
}

AffineTransform generateShapePreservingRandomTransform() {
	ShapePreservingTransform T;

	T.s = uDist(engine);
	T.r.setRandom();
	T.t.setRandom();

	return T.toAffineTransform();
}

void generateObs(int nPts,
				 int nObsperPt,
				 Eigen::VectorXf & obs,
				 Eigen::Matrix3Xf & pts,
				 std::vector<int> & idxs,
				 std::vector<Axis> & coordinate,
				 AffineTransform const& T) {

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

			int rId1 = (random_id != 0) ? 0 : 1;
			int rId2 = (random_id != 1) ? 1 : 2;

			Axis cA1 = (random_id != 0) ? Axis::X : Axis::Y;
			Axis cA2 = (random_id != 1) ? Axis::Y : Axis::Z;

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
	AffineTransform T = generateRandomTransform();

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
	AffineTransform T = generateShapePreservingRandomTransform();

	generateObs(nPts, nObsPerPoints, obs, pts, idxs, coordinate, T);

	AffineTransform F = estimateQuasiShapePreservingMap(obs, pts, idxs, coordinate, 2e-1, nullptr, 1e-7, 500, true);

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

		AffineTransform T = generateShapePreservingRandomTransform();
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
