#include <QtTest/QtTest>

#include "geometry/lensdistortion.h"

#include <random>

using namespace StereoVision::Geometry;

Q_DECLARE_METATYPE(Eigen::Vector3f)
Q_DECLARE_METATYPE(Eigen::Vector2f)

class TestLenseDistortion: public QObject
{
	Q_OBJECT
private Q_SLOTS:
	void initTestCase();

	void testRadialInverse_data();
	void testRadialInverse();

	void testTangentialInverse_data();
	void testTangentialInverse();

	void testRadialTangentialInverse_data();
	void testRadialTangentialInverse();

	void testSkewInverse_data();
	void testSkewInverse();

	void testDistortionModelInverse_data();
	void testDistortionModelInverse();

private:
	std::default_random_engine re;
};

void TestLenseDistortion::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestLenseDistortion::testRadialInverse_data() {

	QTest::addColumn<Eigen::Vector3f>("k123");

	QTest::newRow("Zero distortion") << Eigen::Vector3f(0, 0, 0);

	QTest::newRow("Small distortion 1") << Eigen::Vector3f(0.01, 0, 0);
	QTest::newRow("Small distortion 2") << Eigen::Vector3f(0.02, -0.03, 0.005);
	QTest::newRow("Small distortion 3") << Eigen::Vector3f(-0.006, 0.01, 0.01);

}
void TestLenseDistortion::testRadialInverse() {

	static const float distLimit = 1.5;

	std::uniform_real_distribution<float> posDist(-1, 1);

	QFETCH(Eigen::Vector3f, k123);

	for (int i = 0; i < 100; i++) {
		float x = posDist(re);
		float y = posDist(re);

		Eigen::Vector2f pos;
		pos << x, y;

		Eigen::Vector2f drpos = radialDistortion(pos, k123);

		Eigen::Vector2f mpos = pos + drpos;

		float oNorm = pos.norm();
		float mNorm = mpos.norm();

		if (std::max(oNorm, mNorm)/std::min(oNorm, mNorm) > distLimit) {
			QSKIP("Distortion too important, passing test");
		}

		Eigen::Vector2f rpos = invertRadialDistorstion(mpos, k123);

		float missalignement = (pos - rpos).norm();
		QVERIFY2(missalignement < 1e-2, qPrintable(QString("Reconstructed initial image coordinates not correct (norm (pos - rpos) = %1)").arg(missalignement)));
	}

}

void TestLenseDistortion::testTangentialInverse_data() {

	QTest::addColumn<Eigen::Vector2f>("t12");

	QTest::newRow("Zero distortion") << Eigen::Vector2f(0, 0);

	QTest::newRow("Small distortion 1") << Eigen::Vector2f(0.01, 0);
	QTest::newRow("Small distortion 2") << Eigen::Vector2f(0.02, -0.03);
	QTest::newRow("Small distortion 3") << Eigen::Vector2f(-0.06, 0.01);

}
void TestLenseDistortion::testTangentialInverse() {

	static const float distLimit = 1.5;

	std::uniform_real_distribution<float> posDist(-1, 1);

	QFETCH(Eigen::Vector2f, t12);

	for (int i = 0; i < 100; i++) {
		float x = posDist(re);
		float y = posDist(re);

		Eigen::Vector2f pos;
		pos << x, y;

		Eigen::Vector2f dtpos = tangentialDistortion(pos, t12);

		Eigen::Vector2f mpos = pos + dtpos;

		float oNorm = pos.norm();
		float mNorm = mpos.norm();

		if (std::max(oNorm, mNorm)/std::min(oNorm, mNorm) > distLimit) {
			QSKIP("Distortion too important, passing test");
		}

		Eigen::Vector2f rpos = invertTangentialDistorstion(mpos, t12);

		float missalignement = (pos - rpos).norm();
		QVERIFY2(missalignement < 1e-2, qPrintable(QString("Reconstructed initial image coordinates not correct (norm (pos - rpos) = %1)").arg(missalignement)));
	}

}

void TestLenseDistortion::testRadialTangentialInverse_data() {

	QTest::addColumn<Eigen::Vector3f>("k123");
	QTest::addColumn<Eigen::Vector2f>("t12");

	QTest::newRow("Zero distortion") << Eigen::Vector3f(0, 0, 0) << Eigen::Vector2f(0, 0);

	QTest::newRow("Small distortion 1") << Eigen::Vector3f(0.01, 0, 0) << Eigen::Vector2f(0.01, 0);
	QTest::newRow("Small distortion 2") << Eigen::Vector3f(0.02, -0.03, 0.005) << Eigen::Vector2f(0.02, -0.03);
	QTest::newRow("Small distortion 3") << Eigen::Vector3f(-0.006, 0.01, 0.01) << Eigen::Vector2f(-0.06, 0.01);

}
void TestLenseDistortion::testRadialTangentialInverse() {

	static const float distLimit = 1.5;

	std::uniform_real_distribution<float> posDist(-1, 1);

	QFETCH(Eigen::Vector3f, k123);
	QFETCH(Eigen::Vector2f, t12);

	for (int i = 0; i < 100; i++) {
		float x = posDist(re);
		float y = posDist(re);

		Eigen::Vector2f pos(x, y);

		Eigen::Vector2f drpos = radialDistortion(pos, k123);
		Eigen::Vector2f dtpos = tangentialDistortion(pos, t12);

		Eigen::Vector2f mpos = pos + drpos + dtpos;

		float oNorm = pos.norm();
		float mNorm = mpos.norm();

		if (std::max(oNorm, mNorm)/std::min(oNorm, mNorm) > distLimit) {
			QSKIP("Distortion too important, passing test");
		}

		Eigen::Vector2f rpos = invertRadialTangentialDistorstion(mpos, k123, t12);

		float missalignement = (pos - rpos).norm();
		QVERIFY2(missalignement < 1e-2, qPrintable(QString("Reconstructed initial image coordinates not correct (norm (pos - rpos) = %1)").arg(missalignement)));
	}

}

void TestLenseDistortion::testSkewInverse_data() {

	QTest::addColumn<float>("f");
	QTest::addColumn<Eigen::Vector2f>("pp");
	QTest::addColumn<Eigen::Vector2f>("B12");

	QTest::newRow("Zero distortion") << 1.f << Eigen::Vector2f(0.5, 0.5) << Eigen::Vector2f(0, 0);

	QTest::newRow("Small distortion 1") << 10.f << Eigen::Vector2f(7.5, 5.5) << Eigen::Vector2f(0.1, -0.3);
	QTest::newRow("Small distortion 2") << 15.f << Eigen::Vector2f(4.5, 6.5) << Eigen::Vector2f(1, 1);
	QTest::newRow("Small distortion 3") << 12.f << Eigen::Vector2f(5.5, 5.5) << Eigen::Vector2f(-0.5, 0.8);
}
void TestLenseDistortion::testSkewInverse() {

	std::uniform_real_distribution<float> posDist(-1, 1);

	QFETCH(float, f);
	QFETCH(Eigen::Vector2f, pp);
	QFETCH(Eigen::Vector2f, B12);

	for (int i = 0; i < 100; i++) {
		float x = posDist(re);
		float y = posDist(re);

		Eigen::Vector2f pos(x, y);

		Eigen::Vector2f mpos = skewDistortion(pos, B12, f, pp);

		Eigen::Vector2f rpos = inverseSkewDistortion(mpos, B12, f, pp);

		float missalignement = (pos - rpos).norm();
		QVERIFY2(missalignement < 1e-2, qPrintable(QString("Reconstructed initial image coordinates not correct (norm (pos - rpos) = %1)").arg(missalignement)));
	}

}

void TestLenseDistortion::testDistortionModelInverse_data() {

	QTest::addColumn<float>("f");
	QTest::addColumn<Eigen::Vector2f>("pp");

	QTest::addColumn<Eigen::Vector3f>("k123");
	QTest::addColumn<Eigen::Vector2f>("t12");
	QTest::addColumn<Eigen::Vector2f>("B12");

	QTest::newRow("Zero distortion") << 1.f << Eigen::Vector2f(0.5, 0.5) << Eigen::Vector3f(0, 0, 0) << Eigen::Vector2f(0, 0) << Eigen::Vector2f(0, 0);

	QTest::newRow("Small distortion 1") << 10.f << Eigen::Vector2f(7.5, 5.5) << Eigen::Vector3f(0.01, 0, 0) << Eigen::Vector2f(0.01, 0) << Eigen::Vector2f(0.1, -0.3);
	QTest::newRow("Small distortion 2") << 15.f << Eigen::Vector2f(4.5, 6.5) << Eigen::Vector3f(0.02, -0.03, 0.005) << Eigen::Vector2f(0.02, -0.03) << Eigen::Vector2f(1, 1);
	QTest::newRow("Small distortion 3") << 12.f << Eigen::Vector2f(5.5, 5.5) << Eigen::Vector3f(-0.006, 0.01, 0.01) << Eigen::Vector2f(-0.06, 0.01) << Eigen::Vector2f(-0.1, 0.07);

}
void TestLenseDistortion::testDistortionModelInverse() {

	static const float distLimit = 1.5;

	std::uniform_real_distribution<float> posDist(-1, 1);

	QFETCH(float, f);
	QFETCH(Eigen::Vector2f, pp);

	QFETCH(Eigen::Vector3f, k123);
	QFETCH(Eigen::Vector2f, t12);
	QFETCH(Eigen::Vector2f, B12);


	for (int i = 0; i < 100; i++) {
		float x = posDist(re);
		float y = posDist(re);

		Eigen::Vector2f pos(x, y);
		Eigen::Vector2f upos = f*pos + pp;

		Eigen::Vector2f mpos = fullLensDistortionHomogeneousCoordinates<float, float, float, float>(pos, f, pp, k123, t12, B12);

		float oNorm = upos.norm();
		float mNorm = mpos.norm();

		if (std::max(oNorm, mNorm)/std::min(oNorm, mNorm) > distLimit) {
			QSKIP("Distortion too important, passing test");
		}

		Eigen::Vector2f rpos = invertFullLensDistortionHomogeneousCoordinates<float, float, float, float>(mpos, f, pp, k123, t12, B12, 5);

		float missalignement = (pos - rpos).norm();
		QVERIFY2(missalignement < 1e-2, qPrintable(QString("Reconstructed initial image coordinates not correct (norm (pos - rpos) = %1)").arg(missalignement)));
	}
}

QTEST_MAIN(TestLenseDistortion)
#include "testLenseDistortion.moc"
