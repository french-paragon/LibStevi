#include <QtTest/QtTest>

#include "test_correlation_utils.h"
#include "correlation/hierarchical.h"

Q_DECLARE_METATYPE(StereoVision::Correlation::matchingFunctions);

class TestCorrelationHierarchical: public QObject
{
	Q_OBJECT

private Q_SLOTS:
	void initTestCase();


	void testMatchingFilter_data();
	void testMatchingFilter();


private:
	std::default_random_engine re;

	template<StereoVision::Correlation::matchingFunctions matchFunc, int depth>
	void testMatchingFilterImpl() {
		QFETCH(int, img_height);
		QFETCH(int, img_width);
		QFETCH(int, h_radius);
		QFETCH(int, v_radius);
		QFETCH(int, disp_w);

		int disp_square = (disp_w & (~7))/2;
		int disp_bg = disp_square/2;

		int square_size = std::max(h_radius, v_radius)*(2<<std::max(3,depth))+1;

		int square_v_pos = img_height/2 - square_size/2;
		int square_h_pos = img_width/2 - square_size/2;

		auto imgPair = generateParallaxSquareImage(img_height,
												   img_width,
												   square_size,
												   square_v_pos,
												   square_h_pos,
												   disp_bg,
												   disp_square,
												   re);

		uint8_t h_r = h_radius;
		uint8_t v_r = v_radius;

		auto result = StereoVision::Correlation::hiearchicalTruncatedCostVolume<matchFunc, depth>
				(imgPair.target,
				 imgPair.source,
				 h_r,
				 v_r,
				 disp_w);

		std::cout << result.disp_estimate << std::endl;

		QVERIFY2(result.disp_estimate.shape()[0] == img_height and result.disp_estimate.shape()[1] == img_width,
				qPrintable(QString("result disparity has wrong shape (%1, %2), expected (%3, %4)")
						   .arg(result.disp_estimate.shape()[0])
							.arg(result.disp_estimate.shape()[1])
							.arg(img_height)
							.arg(img_width)
							)
				);

		int count = 0;
		int expected = 0;

		for (int i = square_v_pos + 2*depth*v_radius; i < square_v_pos + square_size - 3*depth*v_radius and i < img_height - 2*depth*v_radius; i++) {

			for (int j = square_h_pos + 2*depth*h_radius; j < square_h_pos + square_size - 3*depth*h_radius and j < img_width - 2*depth*h_radius; j++) {

				int disp_found = result.disp_estimate.value(i,j);

				expected += 1;
				count += (disp_found == disp_square);

			}
		}

		int diff = expected - count;

		if (expected > 0) {
			QVERIFY2(diff < 0.05*expected, qPrintable(QString("Not enough correct matches have been detected (detected: %1, expected: %2)")
													  .arg(count)
													  .arg(expected)
													  )
					 );
		}
	}


	template<int depth>
	void testMatchingFilterImplIntrm(StereoVision::Correlation::matchingFunctions matchFunc) {
		switch (matchFunc) {
		case StereoVision::Correlation::matchingFunctions::CC :
			testMatchingFilterImpl<StereoVision::Correlation::matchingFunctions::CC, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::NCC :
			testMatchingFilterImpl<StereoVision::Correlation::matchingFunctions::NCC, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZNCC :
			testMatchingFilterImpl<StereoVision::Correlation::matchingFunctions::ZNCC, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::SSD :
			testMatchingFilterImpl<StereoVision::Correlation::matchingFunctions::SSD, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSSD :
			testMatchingFilterImpl<StereoVision::Correlation::matchingFunctions::ZSSD, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::SAD :
			testMatchingFilterImpl<StereoVision::Correlation::matchingFunctions::SAD, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSAD :
			testMatchingFilterImpl<StereoVision::Correlation::matchingFunctions::ZSAD, depth>();
			break;
		default:
			QSKIP("Unsupported matching function for the test");
		}
	}

};

void TestCorrelationHierarchical::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestCorrelationHierarchical::testMatchingFilter_data() {

	QTest::addColumn<int>("depth");
	QTest::addColumn<StereoVision::Correlation::matchingFunctions>("matchFunc");
	QTest::addColumn<int>("img_height");
	QTest::addColumn<int>("img_width");
	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("small_lvl1_zncc") << 1 << StereoVision::Correlation::matchingFunctions::ZNCC << 48 << 64 << 2 << 2 << 16;
	QTest::newRow("small_lvl2_zncc") << 2 << StereoVision::Correlation::matchingFunctions::ZNCC << 48 << 64 << 2 << 2 << 16;
	QTest::newRow("small_lvl3_zncc") << 3 << StereoVision::Correlation::matchingFunctions::ZNCC << 48 << 64 << 2 << 2 << 16;

}
void TestCorrelationHierarchical::testMatchingFilter() {

	QFETCH(int, depth);
	QFETCH(StereoVision::Correlation::matchingFunctions, matchFunc);


	switch (depth) {
	case 1:
		testMatchingFilterImplIntrm<1>(matchFunc);
		break;
	case 2:
		testMatchingFilterImplIntrm<2>(matchFunc);
		break;
	case 3:
		testMatchingFilterImplIntrm<3>(matchFunc);
		break;
	default:
		QSKIP("This test suit support depth ranging from 1 to 3 for the hierarchical matching unit test");
	}

}

QTEST_MAIN(TestCorrelationHierarchical)
#include "testCorrelationHierarchical.moc"
