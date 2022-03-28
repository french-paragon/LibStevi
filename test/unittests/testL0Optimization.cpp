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

class TestL0OptimizationMethods: public QObject{

	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testRegionFusionL0Approximation_data();
	void testRegionFusionL0Approximation();

private:
	std::default_random_engine re;
};


void TestL0OptimizationMethods::initTestCase() {
	//srand((unsigned int) time(nullptr));
	std::random_device rd;
	re.seed(rd());
}

void TestL0OptimizationMethods::testRegionFusionL0Approximation_data() {

	QTest::addColumn<int>("w");
	QTest::addColumn<int>("h");
	QTest::addColumn<float>("lambda");

	QTest::newRow("minuscule") << 10 << 10 << 3.0f;
	QTest::newRow("small") << 50 << 50 << 3.0f;
	QTest::newRow("average") << 100 << 100 << 3.0f;


}
void TestL0OptimizationMethods::testRegionFusionL0Approximation() {

	QFETCH(int, w);
	QFETCH(int, h);
	QFETCH(float, lambda);

	auto img = getPiecewiseConstantImage(w,h,re);

	auto approx = regionFusionL0Approximation(img, lambda, -1, 100);

	auto imgShape = img.shape();
	auto approxShape = approx.shape();

	for (size_t i = 0; i < imgShape.size(); i++) {
		QCOMPARE(imgShape[i], approxShape[i]);
	}

	std::array<float, 3> avg_color = {0,0,0};

	for (int i = 0; i < w; i++) {

		for (int j = 0; j < w; j++) {

			for (int c = 0; c < 3; c++) {
				avg_color[c] += img.valueUnchecked(i,j,c);
			}

		}

	}

	for (int c = 0; c < 3; c++) {
		avg_color[c] /= w*h;
	}

	for (int i = 0; i < w; i++) {

		for (int j = 0; j < w; j++) {

			for (int c = 0; c < 3; c++) {
				float imgVal = img.valueUnchecked(i,j,c);
				float approxVal = approx.valueUnchecked(i,j,c);

				float errorVal = imgVal - approxVal;
				float errorAvg = avg_color[c] - approxVal;

				QVERIFY2(std::fabs(errorVal) < 1e-4 or std::fabs(errorAvg) < 1e-4,
						 "The approximated color is not either the original image value or the average value.");
			}

		}

	}

}

QTEST_MAIN(TestL0OptimizationMethods)
#include "testL0Optimization.moc"
