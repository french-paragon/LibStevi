#include <QtTest/QtTest>

#include "../test_correlation_utils.h"
#include "correlation/cross_correlations.h"

typedef std::array<float,2> BidimensionalDirs;
typedef std::array<float,4> BarycentricWeightsPix;
typedef std::array<float,8> BarycentricSymmetricWeightsPix;

Q_DECLARE_METATYPE(BidimensionalDirs);
Q_DECLARE_METATYPE(BarycentricWeightsPix);
Q_DECLARE_METATYPE(BarycentricSymmetricWeightsPix);

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


	void testScoreDisp_data();
	void testScoreDisp();

	void testCostDisp_data();
	void testCostDisp();


	void testNCCBarycentricRefine_data();
	void testNCCBarycentricRefine();

	void testZNCCBarycentricRefine_data();
	void testZNCCBarycentricRefine();

	void testSSDBarycentricRefine_data();
	void testSSDBarycentricRefine();

	void testZSSDBarycentricRefine_data();
	void testZSSDBarycentricRefine();

	void testSADBarycentricRefine_data();
	void testSADBarycentricRefine();

	void testZSADBarycentricRefine_data();
	void testZSADBarycentricRefine();


	void testNCCBarycentricSymmetricRefine_data();
	void testNCCBarycentricSymmetricRefine();

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

	void basic2dDisparityData();

	template<StereoVision::Correlation::matchingFunctions matchFunc>
	void test2dDisparity() {

		typedef StereoVision::Correlation::MatchingFunctionTraits<matchFunc> mFTraits;
		constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

		QFETCH(int, h_radius);
		QFETCH(int, v_radius);
		int disp_w = 3;
		int disp_h = 3;

		int h = 2*v_radius + disp_h + 1;
		int w = 2*h_radius + disp_w + 1;

		Multidim::Array<float, 2> source(h,w);
		Multidim::Array<float, 2> target0(h,w);
		Multidim::Array<float, 2> targetmax(h,w);

		std::uniform_real_distribution<float> uniformDist(-1, 1);

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {
				source.at<Nc>(i,j) = uniformDist(re);
				target0.at<Nc>(i,j) = uniformDist(re);
				targetmax.at<Nc>(i,j) = uniformDist(re);
			}
		}

		for(int i = 0; i < 2*v_radius+1; i++) {
			for (int j = 0; j < 2*h_radius+1; j++) {
				target0.at<Nc>(i,j) = source.value<Nc>(i+1,j+1);
				targetmax.at<Nc>(i+disp_h,j+disp_w) = source.value<Nc>(i+1,j+1);
			}
		}

		StereoVision::Correlation::searchOffset<2> searchRange(-1,disp_h-1,-1,disp_w-1);

		Multidim::Array<float, 4> CV0 = StereoVision::Correlation::unfoldBased2dDisparityCostVolume<matchFunc>(target0,
																											   source,
																											   h_radius,
																											   v_radius,
																											   searchRange);

		Multidim::Array<StereoVision::Correlation::disp_t, 3> disp0 = StereoVision::Correlation::selected2dIndexToDisp(
					StereoVision::Correlation::extractSelected2dIndex<mFTraits::extractionStrategy>(CV0), searchRange);

		QCOMPARE(disp0.value<Nc>(v_radius+1,h_radius+1,0),-1);
		QCOMPARE(disp0.value<Nc>(v_radius+1,h_radius+1,1),-1);

		Multidim::Array<float, 4> CVmax = StereoVision::Correlation::unfoldBased2dDisparityCostVolume<matchFunc>(targetmax,
																												 source,
																												 h_radius,
																												 v_radius,
																												 searchRange);

		Multidim::Array<StereoVision::Correlation::disp_t, 3> dispMax = StereoVision::Correlation::selected2dIndexToDisp(
					StereoVision::Correlation::extractSelected2dIndex<mFTraits::extractionStrategy>(CVmax), searchRange);

		QCOMPARE(dispMax.value<Nc>(v_radius+1,h_radius+1,0),disp_h-1);
		QCOMPARE(dispMax.value<Nc>(v_radius+1,h_radius+1,1),disp_w-1);
	}

	void barycentricRefine2dMatchingData();

	template<StereoVision::Correlation::matchingFunctions matchFunc>
	void barycentricRefine2dMatchingTest() {

		constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

		QFETCH(int, h_radius);
		QFETCH(int, v_radius);
		QFETCH(BarycentricWeightsPix, weights);

		float sum = 0;
		bool allPos = true;
		bool anyBiggerThan07 = false;

		for (float w : weights) {
			sum += w;
			if (w < 0) {
				allPos = false;
			}
			if (w > 0.7) {
				anyBiggerThan07 = true;
			}
		}

		if (!allPos) {
			QSKIP("Malformed data case, the weights have to be positive !");
		}

		if (!anyBiggerThan07) {
			QSKIP("Malformed data case, one of the weight have to be greather than 0.7 !");
		}

		if (std::abs(sum-1) > 1e-8) {
			QSKIP("Malformed data case, the weights have to sum to 1 !");
		}

		constexpr int searchSpaceWidth = 10;

		int h = 2*v_radius + 2*searchSpaceWidth + 2;
		int w = 2*h_radius + 2*searchSpaceWidth + 2;

		Multidim::Array<float, 2> target(h,w);

		std::uniform_real_distribution<float> uniformDist(-1, 1);

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {
				target.at<Nc>(i,j) = uniformDist(re);
			}
		}

		float di = weights[1] + weights[3];
		float dj = weights[2] + weights[3];

		Multidim::Array<float, 2> source(h,w);
		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {

				float comp = 0;

				float v0 = target.valueOrAlt({i,j},0);
				float v1 = target.valueOrAlt({i+1,j},0);
				float v2 = target.valueOrAlt({i,j+1},0);
				float v3 = target.valueOrAlt({i+1,j+1},0);

				comp += weights[0]*v0;
				comp += weights[1]*v1;
				comp += weights[2]*v2;
				comp += weights[3]*v3;

				source.at<Nc>(i,j) = comp;
			}
		}

		StereoVision::Correlation::searchOffset<2> searchRange(0,searchSpaceWidth,0,searchSpaceWidth);

		Multidim::Array<float,3> refDisp = StereoVision::Correlation::refinedBarycentric2dDisp<matchFunc, float, float, 2, StereoVision::Contiguity::Queen>
				(target,
				 source,
				 h_radius,
				 v_radius,
				 searchRange);

		QCOMPARE(refDisp.shape()[2],2);

		constexpr float fTol = 1e-4;

		float count = 0;

		for (int i = 0; i < searchSpaceWidth; i++) {
			for (int j = 0; j < searchSpaceWidth; j++) {

				int id_i = i+v_radius+1;
				int id_j = j+h_radius+1;

				float sub_i = refDisp.value<Nc>(id_i,id_j,0);
				float sub_j = refDisp.value<Nc>(id_i,id_j,1);

				if (std::abs(sub_i - di) < fTol and std::abs(sub_j - dj) < fTol) {
					count += 1;
				}

			}
		}

		float prop = count/float(searchSpaceWidth*searchSpaceWidth);

		constexpr float propTol = 0.9;

		QVERIFY2(prop >= propTol, qPrintable(QString("Not enough points properly reconstructed (proportion of correct match = %1, minimum required = %2)").arg(prop).arg(propTol)));

	}

	void barycentricSymmetricRefine2dMatchingData();

	template<StereoVision::Correlation::matchingFunctions matchFunc>
	void barycentricSymmetricRefine2dMatchingTest() {

		constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

		QFETCH(int, h_radius);
		QFETCH(int, v_radius);
		QFETCH(BarycentricSymmetricWeightsPix, weights);

		float sum = 0;
		bool allPos = true;

		for (float w : weights) {
			sum += w;
			if (w < 0) {
				allPos = false;
			}
		}

		if (!allPos) {
			QSKIP("Malformed data case, the weights have to be positive !");
		}

		if (sum > 0.3) {
			QSKIP("Malformed data case, the outter weights have to sum to a value not greather than 0.3 !");
		}

		constexpr int searchSpaceWidth = 10;

		int h = 2*v_radius + 2*searchSpaceWidth + 2;
		int w = 2*h_radius + 2*searchSpaceWidth + 2;

		Multidim::Array<float, 2> target(h,w);

		std::uniform_real_distribution<float> uniformDist(-1, 1);

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {
				target.at<Nc>(i,j) = uniformDist(re);
			}
		}

		/* Weights order:
		 * 0 1 2
		 * 3 - 4
		 * 5 6 7
		 */

		float di = weights[5] + weights[6] + weights[7] - (weights[0] + weights[1] + weights[2]);
		float dj = weights[2] + weights[4] + weights[7] - (weights[0] + weights[3] + weights[5]);

		Multidim::Array<float, 2> source(h,w);
		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {

				float v0 = target.valueOrAlt({i-1,j-1},0);
				float v1 = target.valueOrAlt({i-1,j},0);
				float v2 = target.valueOrAlt({i-1,j+1},0);
				float v3 = target.valueOrAlt({i,j-1},0);
				float v4 = target.valueOrAlt({i,j+1},0);
				float v5 = target.valueOrAlt({i+1,j-1},0);
				float v6 = target.valueOrAlt({i+1,j},0);
				float v7 = target.valueOrAlt({i+1,j+1},0);
				float vc = target.valueOrAlt({i,j},0);

				float comp = (1-sum)*vc;

				comp += weights[0]*v0;
				comp += weights[1]*v1;
				comp += weights[2]*v2;
				comp += weights[3]*v3;
				comp += weights[4]*v4;
				comp += weights[5]*v5;
				comp += weights[6]*v6;
				comp += weights[7]*v7;

				source.at<Nc>(i,j) = comp;
			}
		}

		StereoVision::Correlation::searchOffset<2> searchRange(0,searchSpaceWidth,0,searchSpaceWidth);

		Multidim::Array<float,3> refDisp = StereoVision::Correlation::refinedBarycentricSymmetric2dDisp<matchFunc, float, float, 2, StereoVision::Contiguity::Queen>
				(target,
				 source,
				 h_radius,
				 v_radius,
				 searchRange);

		QCOMPARE(refDisp.shape()[2],2);

		constexpr float fTol = 1e-3;

		float count = 0;

		for (int i = 0; i < searchSpaceWidth; i++) {
			for (int j = 0; j < searchSpaceWidth; j++) {

				int id_i = i+v_radius+1;
				int id_j = j+h_radius+1;

				float sub_i = refDisp.value<Nc>(id_i,id_j,0);
				float sub_j = refDisp.value<Nc>(id_i,id_j,1);

				if (std::abs(sub_i - di) < fTol and std::abs(sub_j - dj) < fTol) {
					count += 1;
				}

			}
		}

		float prop = count/float(searchSpaceWidth*searchSpaceWidth);

		constexpr float propTol = 0.9;

		QVERIFY2(prop >= propTol, qPrintable(QString("Not enough points properly reconstructed (proportion of correct match = %1, minimum required = %2)").arg(prop).arg(propTol)));

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

void TestCorrelation2d::basic2dDisparityData() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::newRow("small") << 1 << 1;
	QTest::newRow("avg") << 3 << 3;
	QTest::newRow("wide") << 5 << 1;
	QTest::newRow("big") << 1 << 5;
	QTest::newRow("large") << 5 << 5;

}

void TestCorrelation2d::barycentricRefine2dMatchingData() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<BarycentricWeightsPix>("weights");

	QTest::newRow("small") << 1 << 1 << BarycentricWeightsPix({0.75, 0.05, 0.1, 0.1});
	QTest::newRow("avg") << 3 << 3 << BarycentricWeightsPix({0.05, 0.75, 0.05, 0.15});
	QTest::newRow("wide") << 5 << 1 << BarycentricWeightsPix({0.75, 0.1, 0.05, 0.1});
	QTest::newRow("big") << 1 << 5 << BarycentricWeightsPix({0.05, 0.15, 0.75, 0.05});
	QTest::newRow("large") << 5 << 5 << BarycentricWeightsPix({0.1, 0.1, 0.05, 0.75});

}

void TestCorrelation2d::barycentricSymmetricRefine2dMatchingData() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<BarycentricSymmetricWeightsPix>("weights");

	QTest::newRow("small") << 1 << 1 << BarycentricSymmetricWeightsPix({0.04, 0.05, 0.02,0.05, 0.05, 0.02, 0.04, 0.03});
	QTest::newRow("avg") << 3 << 3 << BarycentricSymmetricWeightsPix({0.03, 0.02, 0.05, 0.03, 0.04, 0.04, 0.05, 0.02});
	QTest::newRow("wide") << 5 << 1 << BarycentricSymmetricWeightsPix({0.02, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02,0.05});
	QTest::newRow("big") << 1 << 5 << BarycentricSymmetricWeightsPix({0.04, 0.02,0.05, 0.05, 0.02, 0.04, 0.03, 0.03});
	QTest::newRow("large") << 5 << 5 << BarycentricSymmetricWeightsPix({0.05, 0.05, 0.02, 0.04, 0.03, 0.05, 0.02, 0.04});

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



void TestCorrelation2d::testScoreDisp_data() {
	basic2dDisparityData();
}
void TestCorrelation2d::testScoreDisp() {
	test2dDisparity<StereoVision::Correlation::matchingFunctions::ZNCC>();
}

void TestCorrelation2d::testCostDisp_data() {
	basic2dDisparityData();
}
void TestCorrelation2d::testCostDisp() {
	test2dDisparity<StereoVision::Correlation::matchingFunctions::ZSSD>();
}


void TestCorrelation2d::testNCCBarycentricRefine_data() {
	barycentricRefine2dMatchingData();
}
void TestCorrelation2d::testNCCBarycentricRefine() {
	barycentricRefine2dMatchingTest<StereoVision::Correlation::matchingFunctions::NCC>();
}

void TestCorrelation2d::testZNCCBarycentricRefine_data() {
	barycentricRefine2dMatchingData();
}
void TestCorrelation2d::testZNCCBarycentricRefine() {
	barycentricRefine2dMatchingTest<StereoVision::Correlation::matchingFunctions::ZNCC>();
}

void TestCorrelation2d::testSSDBarycentricRefine_data() {
	barycentricRefine2dMatchingData();
}
void TestCorrelation2d::testSSDBarycentricRefine() {
	barycentricRefine2dMatchingTest<StereoVision::Correlation::matchingFunctions::SSD>();
}

void TestCorrelation2d::testZSSDBarycentricRefine_data() {
	barycentricRefine2dMatchingData();
}
void TestCorrelation2d::testZSSDBarycentricRefine() {
	barycentricRefine2dMatchingTest<StereoVision::Correlation::matchingFunctions::ZSSD>();
}

void TestCorrelation2d::testSADBarycentricRefine_data() {
	barycentricRefine2dMatchingData();
}
void TestCorrelation2d::testSADBarycentricRefine() {
	barycentricRefine2dMatchingTest<StereoVision::Correlation::matchingFunctions::SAD>();
}

void TestCorrelation2d::testZSADBarycentricRefine_data() {
	barycentricRefine2dMatchingData();
}
void TestCorrelation2d::testZSADBarycentricRefine() {
	barycentricRefine2dMatchingTest<StereoVision::Correlation::matchingFunctions::ZSAD>();
}


void TestCorrelation2d::testNCCBarycentricSymmetricRefine_data() {
	barycentricSymmetricRefine2dMatchingData();
}
void TestCorrelation2d::testNCCBarycentricSymmetricRefine() {
	barycentricSymmetricRefine2dMatchingTest<StereoVision::Correlation::matchingFunctions::NCC>();
}

QTEST_MAIN(TestCorrelation2d)
#include "testCorrelation2d.moc"
