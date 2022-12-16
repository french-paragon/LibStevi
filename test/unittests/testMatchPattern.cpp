#include <QTest>

#include <MultidimArrays/MultidimArrays.h>

#include "correlation/matching_costs.h"
#include "correlation/unfold.h"
#include "correlation/cross_correlations.h"
#include "correlation/template_matching.h"

#include <random>

using namespace StereoVision;
using namespace StereoVision::Correlation;

class TestMatchPattern: public QObject
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
private:

	std::default_random_engine re;

	template<matchingFunctions matchFunc>
	void testMatching() {

		QFETCH(int, h_radius);
		QFETCH(int, v_radius);

		QFETCH(int, extend_h);
		QFETCH(int, extend_v);

		int boxSize = (2*h_radius+1)*(2*v_radius+1);

		int h = 2*v_radius + 1 + extend_v;
		int w = 2*h_radius + 1 + extend_h;

		int i = v_radius + extend_v;
		int j = h_radius + extend_h;

		Multidim::Array<float, 2> rand1(h,w);
		Multidim::Array<float, 2> rand2(h,w);

		std::uniform_real_distribution<float> uniformDist(-1, 1);

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {
				rand1.at(i,j) = uniformDist(re);
				rand2.at(i,j) = uniformDist(re);
			}
		}

		Multidim::Array<float, 3> fVol1 = unfold(h_radius, v_radius, rand1);
		Multidim::Array<float, 3> fVol2 = unfold(h_radius, v_radius, rand2);

		QCOMPARE(fVol1.shape()[0], h);
		QCOMPARE(fVol1.shape()[1], w);
		QCOMPARE(fVol1.shape()[2], boxSize);

		QCOMPARE(fVol2.shape()[0], h);
		QCOMPARE(fVol2.shape()[1], w);
		QCOMPARE(fVol2.shape()[2], boxSize);

		searchOffset<2> range(0, h-1, 0, w-1);

		Multidim::Array<float, 4> fullMatchCV =
				featureVolume2CostVolume<matchFunc, float, float, searchOffset<2>, dispDirection::RightToLeft, float>
				(fVol2, fVol1, range);

		QCOMPARE(fullMatchCV.shape()[0], h);
		QCOMPARE(fullMatchCV.shape()[1], w);
		QCOMPARE(fullMatchCV.shape()[2], h);
		QCOMPARE(fullMatchCV.shape()[3], w);

		Multidim::Array<float, 1> pattern = fVol1.indexDimView(2, {0,0});

		Multidim::Array<float, 2> templateMatchCost = matchPattern<matchFunc>(pattern, fVol2);
		Multidim::Array<float, 2> costVolSlice = fullMatchCV.subView(Multidim::DimIndex(0), Multidim::DimIndex(0), Multidim::DimSlice(), Multidim::DimSlice());

		QCOMPARE(templateMatchCost.shape()[0], h);
		QCOMPARE(templateMatchCost.shape()[1], w);

		QCOMPARE(costVolSlice.shape()[0], h);
		QCOMPARE(costVolSlice.shape()[1], w);

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				float tcost = templateMatchCost.atUnchecked(i,j);
				float cvcost = costVolSlice.atUnchecked(i,j);
				QCOMPARE(tcost, cvcost);
			}
		}
	}
};

void TestMatchPattern::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestMatchPattern::testNCCMatching_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::addColumn<int>("extend_h");
	QTest::addColumn<int>("extend_v");

	QTest::newRow("small") << 1 << 1 << 5 << 7;
	QTest::newRow("large") << 2 << 2 << 7 << 11;

}
void TestMatchPattern::testNCCMatching() {
	testMatching<matchingFunctions::NCC>();
}

void TestMatchPattern::testZNCCMatching_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::addColumn<int>("extend_h");
	QTest::addColumn<int>("extend_v");

	QTest::newRow("small") << 1 << 1 << 5 << 7;
	QTest::newRow("large") << 2 << 2 << 7 << 11;

}
void TestMatchPattern::testZNCCMatching() {
	testMatching<matchingFunctions::ZNCC>();
}

void TestMatchPattern::testSSDMatching_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::addColumn<int>("extend_h");
	QTest::addColumn<int>("extend_v");

	QTest::newRow("small") << 1 << 1 << 5 << 7;
	QTest::newRow("large") << 2 << 2 << 7 << 11;

}
void TestMatchPattern::testSSDMatching() {
	testMatching<matchingFunctions::SSD>();
}

QTEST_MAIN(TestMatchPattern)
#include "testMatchPattern.moc"
