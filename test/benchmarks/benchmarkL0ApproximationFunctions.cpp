#include <QtTest/QtTest>

#include "optimization/l0optimization.h"

#include <iostream>
#include <random>

using namespace StereoVision::Optimization;

Multidim::Array<float, 3> getPiecewiseConstantImage(int w,
													int h,
													std::default_random_engine & re,
													float jumpScale = 1.0,
													float jumpVariability = 0.5) {

	std::uniform_real_distribution<float> valuesDist(jumpScale - jumpVariability, jumpScale + jumpVariability);

	std::array<float, 3> color1 = {valuesDist(re), valuesDist(re), valuesDist(re)};
	std::array<float, 3> color2 = {valuesDist(re), valuesDist(re), valuesDist(re)};

	Multidim::Array<float, 3> img(w, h, 3);

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			for (int c = 0; c < 3; c++) {
				img.atUnchecked(i,j,c) = (i < j) ? color1[c] : color2[c];
			}
		}
	}

	return img;
}

class BenchmarkL0ApproximationFunctions: public QObject{

	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testRegionFusionL0Approximation_data();
	void testRegionFusionL0Approximation();

private:
	std::default_random_engine re;
};


void BenchmarkL0ApproximationFunctions::initTestCase() {
	//srand((unsigned int) time(nullptr));
	std::random_device rd;
	re.seed(rd());
}

void BenchmarkL0ApproximationFunctions::testRegionFusionL0Approximation_data() {

	QTest::addColumn<int>("w");
	QTest::addColumn<int>("h");
	QTest::addColumn<float>("lambda");

	QTest::newRow("half defintion") << 320 << 240 << 3.0f;
	//QTest::newRow("high definition") << 1920 << 1080 << 3.0f;
	//QTest::newRow("4k") << 3840 << 2160 << 3.0f;


}
void BenchmarkL0ApproximationFunctions::testRegionFusionL0Approximation() {

	QFETCH(int, w);
	QFETCH(int, h);
	QFETCH(float, lambda);

	auto img = getPiecewiseConstantImage(w,h,re);

	Multidim::Array<float, 3> approx;

	QBENCHMARK_ONCE {
		approx = regionFusionL0Approximation(img, lambda, -1, 100);
	}

	auto imgShape = img.shape();
	auto approxShape = approx.shape();

	for (size_t i = 0; i < imgShape.size(); i++) {
		QCOMPARE(imgShape[i], approxShape[i]);
	}

	for (int i = 0; i < w; i++) {

		for (int j = 0; j < h; j++) {

			for (int c = 0; c < 3; c++) {
				QVERIFY2(std::isfinite(approx.valueUnchecked(i,j,c)),"nan detected in results");
			}

		}

	}

}

QTEST_MAIN(BenchmarkL0ApproximationFunctions)
#include "benchmarkL0ApproximationFunctions.moc"
