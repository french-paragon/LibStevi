#include <QtTest/QtTest>

#include "test_correlation_utils.h"
#include "correlation/cross_correlations.h"

class TestCorrelation2d: public QObject
{
	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testNCCMatching_data();
	void testNCCMatching();

	void testZNCCMatching_data();
	void testZNCCMatching();

	void testSSDMatching_data();
	void testSSDMatching();

	void testZSSDMatching_data();
	void testZSSDMatching();

	void testSADMatching_data();
	void testSADMatching();

	void testZSADMatching_data();
	void testZSADMatching();

private:

	void basic2dMatchingData();

	template<StereoVision::Correlation::matchingFunctions matchFunc>
	void test2dMatching() {

		constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

		QFETCH(int, h_radius);
		QFETCH(int, v_radius);
		QFETCH(int, disp_w);
		QFETCH(int, disp_h);

		int h = 2*v_radius + disp_h + 1;
		int w = 2*h_radius + disp_w + 1;

		Multidim::Array<float, 2> randLeft(h,w);
		Multidim::Array<float, 2> randRight(h,w);

		std::uniform_real_distribution<float> uniformDist(-1, 1);

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {
				randLeft.at<Nc>(i,j) = uniformDist(re);
				randRight.at<Nc>(i,j) = uniformDist(re);
			}
		}

		StereoVision::Correlation::searchOffset<2> searchRange(0,disp_h,0,disp_w);

		Multidim::Array<float, 4> CV = StereoVision::Correlation::unfoldBased2dDisparityCostVolume<matchFunc>(randLeft,
																											  randRight,
																											  h_radius,
																											  v_radius,
																											  searchRange);

		Multidim::Array<float, 2> window1 = randRight.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(0,2*h_radius+1));

		for (int i = 0; i < disp_h; i++) {
			for (int j = 0; j < disp_w; j++) {

				Multidim::Array<float, 2> window2 = randLeft.subView(Multidim::DimSlice(i,i+2*v_radius+1), Multidim::DimSlice(j,j+2*h_radius+1));

				float unnefectiveVal = InneficientMatchingFunction<matchFunc>(window1, window2);
				float effectiveVal = CV.value<Nc>(v_radius, h_radius, i, j);

				float missalignement = std::abs(unnefectiveVal - effectiveVal);
				QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed %0 is wrong (error = %1 at disp index %2,%3)")
														   .arg(QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::Name))
														   .arg(missalignement)
														   .arg(i)
														   .arg(j)));
			}
		}

	}


	std::default_random_engine re;

};

void TestCorrelation2d::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestCorrelation2d::basic2dMatchingData() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");
	QTest::addColumn<int>("disp_h");

	QTest::newRow("small") << 1 << 1 << 5 << 5;
	QTest::newRow("avg") << 3 << 3 << 5 << 5;
	QTest::newRow("wide") << 5 << 1 << 5 << 5;
	QTest::newRow("big") << 1 << 5 << 5 << 5;
	QTest::newRow("large") << 5 << 5 << 5 << 5;

}

void TestCorrelation2d::testNCCMatching_data() {
	basic2dMatchingData();
}
void TestCorrelation2d::testNCCMatching() {
	test2dMatching<StereoVision::Correlation::matchingFunctions::NCC>();
}

void TestCorrelation2d::testZNCCMatching_data() {
	basic2dMatchingData();
}
void TestCorrelation2d::testZNCCMatching() {
	test2dMatching<StereoVision::Correlation::matchingFunctions::ZNCC>();
}

void TestCorrelation2d::testSSDMatching_data() {
	basic2dMatchingData();
}
void TestCorrelation2d::testSSDMatching() {
	test2dMatching<StereoVision::Correlation::matchingFunctions::SSD>();
}

void TestCorrelation2d::testZSSDMatching_data() {
	basic2dMatchingData();
}
void TestCorrelation2d::testZSSDMatching() {
	test2dMatching<StereoVision::Correlation::matchingFunctions::ZSSD>();
}

void TestCorrelation2d::testSADMatching_data() {
	basic2dMatchingData();
}
void TestCorrelation2d::testSADMatching() {
	test2dMatching<StereoVision::Correlation::matchingFunctions::SAD>();
}

void TestCorrelation2d::testZSADMatching_data() {
	basic2dMatchingData();
}
void TestCorrelation2d::testZSADMatching() {
	test2dMatching<StereoVision::Correlation::matchingFunctions::ZSAD>();
}

QTEST_MAIN(TestCorrelation2d)
#include "testCorrelation2d.moc"
