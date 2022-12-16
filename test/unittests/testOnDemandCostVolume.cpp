#include <QtTest/QtTest>
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022  Paragon<french.paragon@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "correlation/on_demand_cost_volume.h"
#include "correlation/cross_correlations.h"

#include <MultidimArrays/MultidimArrays.h>

#include <random>

using namespace StereoVision;
using namespace StereoVision::Correlation;

Q_DECLARE_METATYPE(StereoVision::Correlation::matchingFunctions);

Multidim::Array<float, 3> buildRandomFeatureVolume(int h, int w, int c, std::function<float()> const& rg) {

	Multidim::Array<float, 3> feature_volume(h,w,c);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int f = 0; f < c; f++) {
				feature_volume.atUnchecked(i,j,f) = rg();
			}
		}
	}

	return feature_volume;

}

class TestOnDemandCostVolume: public QObject
{
	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testOnDemandCVStereo_data();
	void testOnDemandCVStereo();

	void testOnDemandCVOpticalFlow_data();
	void testOnDemandCVOpticalFlow();

private:

	std::default_random_engine re;

	template<StereoVision::Correlation::matchingFunctions matchFunc>
	void testStereoCV(int imgHeight, int imgWidth, int nFeatures, int maxDisp) {

		using OnDemandCostVolT = OnDemandStereoCostVolume<matchFunc, float, float, float, Multidim::NonConstView, Multidim::NonConstView>;
		using SearchSpaceT = typename OnDemandCostVolT::SearchSpaceType;

		static_assert (OnDemandCostVolT::nSearchDim == 1, "Error in cv type");

		if (maxDisp <= 0) {
			QSKIP("Disparity not large enough");
		}

		std::uniform_real_distribution<float> dist(-10, 10);
		Multidim::Array<float, 3> source = buildRandomFeatureVolume(imgHeight, imgWidth, nFeatures, [this, &dist] () {return dist(re);});
		Multidim::Array<float, 3> target = buildRandomFeatureVolume(imgHeight, imgWidth, nFeatures, [this, &dist] () {return dist(re);});

		SearchSpaceT searchSpace(SearchSpaceBase::IgnoredDim(), SearchSpaceBase::SearchDim(0, maxDisp), SearchSpaceBase::FeatureDim());

		OnDemandCostVolT on_demand_cv(source, target, searchSpace);

		disp_t disp_width = maxDisp+1;
		Multidim::Array<float, 3> cv = featureVolume2CostVolume<matchFunc, float, float, disp_t, dispDirection::RightToLeft, float>(target, source, disp_width);

		auto od_shape = on_demand_cv.shape();
		auto cv_shape = cv.shape();

		for (size_t i = 0; i < cv_shape.size(); i++) {
			QCOMPARE(od_shape[i], cv_shape[i]);
		}

		for (int i = 0; i < od_shape[0]; i++) {
			for (int j = 0; j < od_shape[1]; j++) {
				for (int d = 0; d < od_shape[2]; d++) {

					auto opt_cost = on_demand_cv.costValue({i,j},{d});

					if (j + d < 0 or j + d >= od_shape[1]) {
						QVERIFY2(!opt_cost.has_value(), "On demand cost volume has a value when it should not!");
						continue;
					}

					QVERIFY2(opt_cost.has_value(), "On demand cost volume has no value when it should !");
					float cost = opt_cost.value();

					QCOMPARE(cost, cv.atUnchecked(i,j,d));

				}
			}
		}

	}

	template<StereoVision::Correlation::matchingFunctions matchFunc>
	void testFlowCV(int imgHeight, int imgWidth, int nFeatures, int dispRadius) {

		using OnDemandCostVolT = OnDemandImageFlowVolume<matchFunc, float, float, float, Multidim::NonConstView, Multidim::NonConstView>;
		using SearchSpaceT = typename OnDemandCostVolT::SearchSpaceType;

		static_assert (OnDemandCostVolT::nSearchDim == 2, "Error in cv type");

		if (dispRadius <= 0) {
			QSKIP("Disparity not large enough");
		}

		std::uniform_real_distribution<float> dist(-10, 10);
		Multidim::Array<float, 3> source = buildRandomFeatureVolume(imgHeight, imgWidth, nFeatures, [this, &dist] () {return dist(re);});
		Multidim::Array<float, 3> target = buildRandomFeatureVolume(imgHeight, imgWidth, nFeatures, [this, &dist] () {return dist(re);});

		SearchSpaceT searchSpace(SearchSpaceBase::SearchDim(-dispRadius, dispRadius), SearchSpaceBase::SearchDim(-dispRadius, dispRadius), SearchSpaceBase::FeatureDim());
		searchOffset<2> searchOffsets(-dispRadius, dispRadius, -dispRadius, dispRadius);


		OnDemandCostVolT on_demand_cv(source, target, searchSpace);

		Multidim::Array<float, 4> cv = featureVolume2CostVolume<matchFunc, float, float, searchOffset<2>, dispDirection::RightToLeft, float>(target, source, searchOffsets);

		auto od_shape = on_demand_cv.shape();
		auto cv_shape = cv.shape();

		for (size_t i = 0; i < cv_shape.size(); i++) {
			QCOMPARE(od_shape[i], cv_shape[i]);
		}

		for (int i = 0; i < od_shape[0]; i++) {
			for (int j = 0; j < od_shape[1]; j++) {
				for (int d1 = -dispRadius; d1 <= dispRadius; d1++) {
					for (int d2 = -dispRadius; d2 <= dispRadius; d2++) {

						auto opt_cost = on_demand_cv.costValue({i,j},{d1, d2});

						if (i + d1 < 0 or i + d1 >= od_shape[0]) {
							QVERIFY2(!opt_cost.has_value(), "On demand cost volume has a value when it should not!");
							continue;
						}

						if (j + d2 < 0 or j + d2 >= od_shape[1]) {
							QVERIFY2(!opt_cost.has_value(), "On demand cost volume has a value when it should not!");
							continue;
						}

						QVERIFY2(opt_cost.has_value(), "On demand cost volume has no value when it should !");
						float cost = opt_cost.value();

						QCOMPARE(cost, cv.atUnchecked(i,j,searchSpace.disp2idx(0,d1), searchSpace.disp2idx(1,d2)));
					}
				}
			}
		}

	}
};


void TestOnDemandCostVolume::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}


void TestOnDemandCostVolume::testOnDemandCVStereo_data() {


	QTest::addColumn<matchingFunctions>("matchFunc");
	QTest::addColumn<int>("h");
	QTest::addColumn<int>("w");
	QTest::addColumn<int>("f");
	QTest::addColumn<int>("d");

	QTest::newRow("NCC small") << matchingFunctions::NCC << 1 << 5 << 2 << 2;
	QTest::newRow("NCC avg") << matchingFunctions::NCC << 16 << 32 << 9 << 15;

	QTest::newRow("ZNCC small") << matchingFunctions::ZNCC << 1 << 5 << 2 << 2;
	QTest::newRow("ZNCC avg") << matchingFunctions::ZNCC << 16 << 32 << 9 << 15;
}
void TestOnDemandCostVolume::testOnDemandCVStereo() {

	QFETCH(matchingFunctions, matchFunc);
	QFETCH(int, h);
	QFETCH(int, w);
	QFETCH(int, f);
	QFETCH(int, d);

	switch (matchFunc) {
	case matchingFunctions::NCC:
		testStereoCV<matchingFunctions::NCC>(h, w, f, d);
		break;
	case matchingFunctions::ZNCC:
		testStereoCV<matchingFunctions::ZNCC>(h, w, f, d);
		break;
	default:
		QSKIP("Unsupported cost function");
	}
}

void TestOnDemandCostVolume::testOnDemandCVOpticalFlow_data() {


	QTest::addColumn<matchingFunctions>("matchFunc");
	QTest::addColumn<int>("h");
	QTest::addColumn<int>("w");
	QTest::addColumn<int>("f");
	QTest::addColumn<int>("d");

	QTest::newRow("NCC small") << matchingFunctions::NCC << 1 << 5 << 2 << 2;
	QTest::newRow("NCC avg") << matchingFunctions::NCC << 16 << 32 << 9 << 15;

	QTest::newRow("ZNCC small") << matchingFunctions::ZNCC << 1 << 5 << 2 << 2;
	QTest::newRow("ZNCC avg") << matchingFunctions::ZNCC << 16 << 32 << 9 << 15;

}
void TestOnDemandCostVolume::testOnDemandCVOpticalFlow() {

	QFETCH(matchingFunctions, matchFunc);
	QFETCH(int, h);
	QFETCH(int, w);
	QFETCH(int, f);
	QFETCH(int, d);

	switch (matchFunc) {
	case matchingFunctions::NCC:
		testFlowCV<matchingFunctions::NCC>(h, w, f, d);
		break;
	case matchingFunctions::ZNCC:
		testFlowCV<matchingFunctions::ZNCC>(h, w, f, d);
		break;
	default:
		QSKIP("Unsupported cost function");
	}

}

QTEST_MAIN(TestOnDemandCostVolume)
#include "testOnDemandCostVolume.moc"
