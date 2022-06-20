#include <QtTest/QtTest>

#include "../test_correlation_utils.h"
#include "correlation/cross_correlations.h"
#include "correlation/image_based_refinement.h"

typedef Multidim::Array<int,2> CompressorMask;
typedef std::array<float,3> SymmetricWeightsSplit;

Q_DECLARE_METATYPE(CompressorMask);
Q_DECLARE_METATYPE(SymmetricWeightsSplit);

class TestCorrelationFilters: public QObject
{
	Q_OBJECT

private Q_SLOTS:
	void initTestCase();

	void testMeanFilter_data();
	void testMeanFilter();

	void testChannelMean_data();
	void testChannelMean();

	void testSigmaFilter_data();
	void testSigmaFilter();

	void testChannelSigma_data();
	void testChannelSigma();

	void testCrossCorrelationFilter_data();
	void testCrossCorrelationFilter();

	void testNCCFilter_data();
	void testNCCFilter();

	void testUnfoldOperator_data();
	void testUnfoldOperator();

	void testUnfoldNCCFilter_data();
	void testUnfoldNCCFilter();

	void testUnfoldCompressor_data();
	void testUnfoldCompressor();

	void testBarycentricNccRefinement_data();
	void testBarycentricNccRefinement();

	void testBarycentricSsdRefinement_data();
	void testBarycentricSsdRefinement();

	void testBarycentricSadRefinement_data();
	void testBarycentricSadRefinement();

	void testBarycentricSymmetricNccRefinement_data();
	void testBarycentricSymmetricNccRefinement();

	void testBarycentricSymmetricSsdRefinement_data();
	void testBarycentricSymmetricSsdRefinement();

	void testBarycentricSymmetricSadRefinement_data();
	void testBarycentricSymmetricSadRefinement();

	void testInputImagesTypes();

private:
	std::default_random_engine re;

};

void TestCorrelationFilters::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}


void TestCorrelationFilters::testMeanFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::addColumn<int>("extend_h");
	QTest::addColumn<int>("extend_v");

	QTest::newRow("Minimal") << 0 << 0 << 0 << 0;
	QTest::newRow("small") << 1 << 1 << 1 << 1;
	QTest::newRow("wide") << 5 << 1 << 7 << 7;
	QTest::newRow("big") << 1 << 5 << 7 << 7;
	QTest::newRow("large") << 5 << 5 << 7 << 7;
}

void TestCorrelationFilters::testMeanFilter() {

	QFETCH(int, h_radius);
	QFETCH(int, v_radius);

	QFETCH(int, extend_h);
	QFETCH(int, extend_v);

	int boxSize = (2*h_radius+1)*(2*v_radius+1);

	int h = 2*v_radius + 1 + extend_v;
	int w = 2*h_radius + 1 + extend_h;

	int i = v_radius + extend_v;
	int j = h_radius + extend_h;

	Multidim::Array<float, 2> rand(h,w);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			rand.at(i,j) = uniformDist(re);
		}
	}

	float sum = 0;

	for(int i = extend_v; i < h; i++) {
		for(int j = extend_h; j < w; j++) {
			sum += rand.at(i,j);
		}
	}

	Multidim::Array<float, 2> avg = StereoVision::Correlation::meanFilter2D(h_radius, v_radius, rand);

	float missalignement = std::abs((sum / boxSize) - avg.at(i, j));
	QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed mean is wrong (error = %1)").arg(missalignement)));

}



void TestCorrelationFilters::testChannelMean_data() {

	QTest::addColumn<int>("nChannels");

	QTest::newRow("few") << 5;
	QTest::newRow("some") << 50;
	QTest::newRow("many") << 500;

}
void TestCorrelationFilters::testChannelMean() {

	QFETCH(int, nChannels);

	Multidim::Array<float, 3> rand(1,1,nChannels);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	float mean = 0;

	for(int i = 0; i < nChannels; i++) {
		rand.at(0,0,i) = uniformDist(re);
		mean += rand.at(0,0,i);
	}

	mean /= nChannels;

	Multidim::Array<float, 2> avg = StereoVision::Correlation::channelsMean(rand);

	float missalignement = std::abs(mean - avg.at(0,0));
	QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed mean is wrong (error = %1)").arg(missalignement)));
}

void TestCorrelationFilters::testSigmaFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::newRow("small") << 1 << 1;
	QTest::newRow("wide") << 5 << 1;
	QTest::newRow("big") << 1 << 5;
	QTest::newRow("large") << 5 << 5;
}

void TestCorrelationFilters::testSigmaFilter() {

	QFETCH(int, h_radius);
	QFETCH(int, v_radius);

	int boxSize = (2*h_radius+1)*(2*v_radius+1);

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 1;

	Multidim::Array<float, 2> rand(h,w);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	float mean = 0;

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			rand.at(i,j) = uniformDist(re);
			mean += rand.at(i,j);
		}
	}

	mean /= boxSize;
	float std = 0;

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			std += (rand.at(i,j) - mean)*(rand.at(i,j) - mean);
		}
	}

	Multidim::Array<float, 2> avg = StereoVision::Correlation::meanFilter2D(h_radius, v_radius, rand);
	Multidim::Array<float, 2> sigma = StereoVision::Correlation::sigmaFilter(h_radius, v_radius, avg, rand);

	float missalignement = std::abs(sqrtf(std) - sigma.at(v_radius, h_radius));
	QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed mean is wrong (error = %1)").arg(missalignement)));

}


void TestCorrelationFilters::testChannelSigma_data() {

	QTest::addColumn<int>("nChannels");

	QTest::newRow("few") << 5;
	QTest::newRow("some") << 50;
	QTest::newRow("many") << 500;
}
void TestCorrelationFilters::testChannelSigma() {

	QFETCH(int, nChannels);

	Multidim::Array<float, 3> rand(1,1,nChannels);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	float mean = 0;

	for(int i = 0; i < nChannels; i++) {
		rand.at(0,0,i) = uniformDist(re);
		mean += rand.at(0,0,i);
	}

	mean /= nChannels;

	float sigma = 0;

	for(int i = 0; i < nChannels; i++) {
		float tmp = rand.at(0,0,i) - mean;
		sigma += tmp*tmp;
	}

	sigma = sqrtf(sigma);

	Multidim::Array<float, 2> sigm = StereoVision::Correlation::channelsZeroMeanNorm(rand);

	float missalignement = std::abs(sigma - sigm.at(0,0));
	QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed mean is wrong (error = %1)").arg(missalignement)));
}

void TestCorrelationFilters::testCrossCorrelationFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("small") << 1 << 1 << 5;
	QTest::newRow("avg") << 3 << 3 << 5;
	QTest::newRow("wide") << 5 << 1 << 5;
	QTest::newRow("big") << 1 << 5 << 5;
	QTest::newRow("large") << 5 << 5 << 5;
}
void TestCorrelationFilters::testCrossCorrelationFilter() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	QFETCH(int, h_radius);
	QFETCH(int, v_radius);
	QFETCH(int, disp_w);

	int h = 2*v_radius + 1;
	int w = 2*h_radius + disp_w + 1;

	Multidim::Array<float, 2> randLeft(h,w);
	Multidim::Array<float, 2> randRight(h,w);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			randLeft.at(i,j) = uniformDist(re);
			randRight.at(i,j) = uniformDist(re);
		}
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZCC;
	Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(randLeft, randRight, h_radius, v_radius, disp_w);

	Multidim::Array<float, 2> window1 = randRight.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(0,2*h_radius+1));

	for (int i = 0; i < disp_w; i++) {

		Multidim::Array<float, 2> window2 = randLeft.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(i,i+2*h_radius+1));

		float unnefectiveVal = InneficientZeromeanCrossCorrelation(window1, window2);
		float effectiveVal = CV.value<Nc>(v_radius, h_radius, i);

		float missalignement = std::abs(unnefectiveVal - effectiveVal);
		QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed NCC is wrong (error = %1 at disp index %2)").arg(missalignement).arg(i)));
	}

}

void TestCorrelationFilters::testNCCFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("small") << 1 << 1 << 5;
	QTest::newRow("avg") << 3 << 3 << 5;
	QTest::newRow("wide") << 5 << 1 << 5;
	QTest::newRow("big") << 1 << 5 << 5;
	QTest::newRow("large") << 5 << 5 << 5;

}
void TestCorrelationFilters::testNCCFilter() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	QFETCH(int, h_radius);
	QFETCH(int, v_radius);
	QFETCH(int, disp_w);

	int h = 2*v_radius + 1;
	int w = 2*h_radius + disp_w + 1;

	Multidim::Array<float, 2> randLeft(h,w);
	Multidim::Array<float, 2> randRight(h,w);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			randLeft.at(i,j) = uniformDist(re);
			randRight.at(i,j) = uniformDist(re);
		}
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZNCC;
	Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(randLeft, randRight, h_radius, v_radius, disp_w);

	Multidim::Array<float, 2> window1 = randRight.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(0,2*h_radius+1));

	for (int i = 0; i < disp_w; i++) {

		Multidim::Array<float, 2> window2 = randLeft.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(i,i+2*h_radius+1));

		float unnefectiveVal = InneficientZeromeanNormalizedCrossCorrelation(window1, window2);
		float effectiveVal = CV.value<Nc>(v_radius, h_radius, i);

		float missalignement = std::abs(unnefectiveVal - effectiveVal);
		QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed NCC is wrong (error = %1 at disp index %2)").arg(missalignement).arg(i)));
	}
}

void TestCorrelationFilters::testUnfoldOperator_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::newRow("small") << 1 << 1;
	QTest::newRow("avg") << 3 << 3;
	QTest::newRow("wide") << 5 << 1;
	QTest::newRow("big") << 1 << 5;
	QTest::newRow("large") << 5 << 5;
}

void TestCorrelationFilters::testUnfoldOperator() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	QFETCH(int, h_radius);
	QFETCH(int, v_radius);

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 1;

	Multidim::Array<float, 2> rand(h,w);
	std::vector<float> values;
	values.reserve(h*w);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val = uniformDist(re);
			rand.at(i,j) = val;
			values.push_back(val);
		}
	}

	Multidim::Array<float, 3> unfolded = StereoVision::Correlation::unfold(h_radius, v_radius, rand, StereoVision::PaddingMargins(0));

	QCOMPARE(unfolded.shape()[0], 1);
	QCOMPARE(unfolded.shape()[1], 1);
	QCOMPARE(unfolded.shape()[2], h*w);

	std::vector<float> extractedVals;
	extractedVals.reserve(h*w);

	for (int i = 0; i < h*w; i++) {
		extractedVals.push_back(unfolded.value<Nc>(0,0,i));
	}

	std::sort(values.begin(), values.end());
	std::sort(extractedVals.begin(), extractedVals.end());

	for (int i = 0; i < h*w; i++) {
		QCOMPARE(values[i], extractedVals[i]);
	}

	Multidim::Array<float, 3> unfoldedPadded = StereoVision::Correlation::unfold(h_radius, v_radius, rand, StereoVision::PaddingMargins());

	QCOMPARE(unfoldedPadded.shape()[0], h);
	QCOMPARE(unfoldedPadded.shape()[1], w);
	QCOMPARE(unfoldedPadded.shape()[2], h*w);

	extractedVals.clear();

	for (int i = 0; i < h*w; i++) {
		extractedVals.push_back(unfoldedPadded.value<Nc>(v_radius,h_radius,i));
	}

	std::sort(extractedVals.begin(), extractedVals.end());

	for (int i = 0; i < h*w; i++) {
		QCOMPARE(values[i], extractedVals[i]);
	}
}


void TestCorrelationFilters::testUnfoldNCCFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("small") << 1 << 1 << 5;
	QTest::newRow("avg") << 3 << 3 << 5;
	QTest::newRow("wide") << 5 << 1 << 5;
	QTest::newRow("big") << 1 << 5 << 5;
	QTest::newRow("large") << 5 << 5 << 5;

}
void TestCorrelationFilters::testUnfoldNCCFilter() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	QFETCH(int, h_radius);
	QFETCH(int, v_radius);
	QFETCH(int, disp_w);

	int h = 2*v_radius + 1;
	int w = 2*h_radius + disp_w + 1;

	Multidim::Array<float, 2> randLeft(h,w);
	Multidim::Array<float, 2> randRight(h,w);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			randLeft.at(i,j) = uniformDist(re);
			randRight.at(i,j) = uniformDist(re);
		}
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZNCC;
	Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(randLeft, randRight, h_radius, v_radius, disp_w);

	Multidim::Array<float, 2> window1 = randRight.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(0,2*h_radius+1));

	for (int i = 0; i < disp_w; i++) {

		Multidim::Array<float, 2> window2 = randLeft.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(i,i+2*h_radius+1));

		float unnefectiveVal = InneficientZeromeanNormalizedCrossCorrelation(window1, window2);
		float effectiveVal = CV.value<Nc>(v_radius, h_radius, i);

		float missalignement = std::abs(unnefectiveVal - effectiveVal);
		QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed NCC is wrong (error = %1 at disp index %2)").arg(missalignement).arg(i)));
	}
}

void TestCorrelationFilters::testUnfoldCompressor_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::newRow("small") << 1 << 1;
	QTest::newRow("avg") << 3 << 3;
	QTest::newRow("wide") << 5 << 1;
	QTest::newRow("big") << 1 << 5;
	QTest::newRow("large") << 5 << 5;

}
void TestCorrelationFilters::testUnfoldCompressor() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	QFETCH(int, h_radius);
	QFETCH(int, v_radius);

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 1;

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	Multidim::Array<float, 2> rand(h,w);
	Multidim::Array<int, 2> mask(h,w);
	std::vector<int> superpixs(h*w);

	int fnum = h;
	int f = 0;

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val = uniformDist(re);
			rand.at(i,j) = val;
			superpixs[i*w + j] = ((f++)%fnum)+1;
		}
	}

	std::shuffle(superpixs.begin(), superpixs.end(), re);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			mask.at<Nc>(i,j) = superpixs[i*w + j];
		}
	}


	std::vector<float> vals(fnum);
	std::vector<float> valsCheck(fnum);

	StereoVision::Correlation::UnFoldCompressor compressor(mask);

	Multidim::Array<float, 3> unfoldedCompressed = StereoVision::Correlation::unfold(compressor, rand, StereoVision::PaddingMargins(0));

	QCOMPARE(fnum, unfoldedCompressed.shape()[2]);

	for (int f = 1; f <= fnum; f++) {
		vals[f-1] = unfoldedCompressed.value<Nc>(0,0,f-1);
		valsCheck[f-1] = 0;
		float count = 0;

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {

				int t = mask.at<Nc>(i,j);

				if (t == f) {
					count += 1.0;
					valsCheck[f-1] += rand.value<Nc>(i,j);
				}
			}
		}

		valsCheck[f-1] /= count;

	}

	std::sort(vals.begin(), vals.end());
	std::sort(valsCheck.begin(), valsCheck.end());

	for (int f = 0; f < fnum; f++) {
		float missalignement = std::abs(vals[f] - valsCheck[f]);
		QVERIFY2(missalignement < 1e-4, qPrintable(QString("feature is wrong (error = %1 at feature index %2)").arg(missalignement).arg(f)));
	}

}

void TestCorrelationFilters::testBarycentricNccRefinement_data() {

	QTest::addColumn<int>("c_radius");
	QTest::addColumn<float>("subpixel_adjustement");

	QTest::newRow("small_l") << 1 << -0.3f;
	QTest::newRow("small_r") << 1 << 0.3f;
	QTest::newRow("avg_l") << 3 << -0.25f;
	QTest::newRow("avg_r") << 3 << 0.25f;
	QTest::newRow("large_l") << 5 << -0.3f;
	QTest::newRow("large_r") << 5 << 0.3f;


}
void TestCorrelationFilters::testBarycentricNccRefinement() {

	QFETCH(int, c_radius);
	QFETCH(float, subpixel_adjustement);

	int h_radius = c_radius;
	int v_radius = c_radius;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 3;

	int f = (2*v_radius + 1)*(2*h_radius + 1);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	Multidim::Array<float, 2> rand(h,w);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val = uniformDist(re);
			rand.at<Nc>(i,j) = val;
		}
	}

	Multidim::Array<float, 3> unfolded = StereoVision::Correlation::unfold(h_radius,
																		   v_radius,
																		   rand,
																		   StereoVision::PaddingMargins(0));

	//Multidim::Array<float, 2> mean = StereoVision::Correlation::channelsMean(unfolded);
	//for (int c = 0; c < f; c++) {
	//	for (int i = 0; i < 3; i++) {
	//		unfolded.at<Nc>(0,i,c) = unfolded.value<Nc>(0,i,c) - mean.value<Nc>(0,i);
	//	}
	//}

	Multidim::Array<float, 3> source(1,3,f);

	for (int c = 0; c < f; c++) {
		float val = 0;
		if (subpixel_adjustement < 0) {
			val = (1+subpixel_adjustement)*unfolded.value<Nc>(0,1,c) - subpixel_adjustement*unfolded.value<Nc>(0,0,c);
		} else {
			val = (1-subpixel_adjustement)*unfolded.value<Nc>(0,1,c) + subpixel_adjustement*unfolded.value<Nc>(0,2,c);
		}
		source.at<Nc>(0,0,c) = val;
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::NCC;
	Multidim::Array<float, 2> disparity = StereoVision::Correlation::refinedBarycentricDisp<matchFunc,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft>(unfolded, source, 0, 0, 3);

	float missalignement = disparity.value<Nc>(0) - (1+subpixel_adjustement);
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(1+subpixel_adjustement)));


	constexpr auto matchFunc2 = StereoVision::Correlation::matchingFunctions::ZNCC;
	Multidim::Array<float, 2> disparity2 = StereoVision::Correlation::refinedBarycentricDisp<matchFunc2,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft>(unfolded, source, 0, 0, 3);

	missalignement = disparity2.value<Nc>(0) - (1+subpixel_adjustement);
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity2.value<Nc>(0)).arg(1+subpixel_adjustement)));

}

void TestCorrelationFilters::testBarycentricSsdRefinement_data() {

	QTest::addColumn<int>("c_radius");
	QTest::addColumn<float>("subpixel_adjustement");

	QTest::newRow("small_l") << 1 << -0.3f;
	QTest::newRow("small_r") << 1 << 0.3f;
	QTest::newRow("avg_l") << 3 << -0.25f;
	QTest::newRow("avg_r") << 3 << 0.25f;
	QTest::newRow("large_l") << 5 << -0.3f;
	QTest::newRow("large_r") << 5 << 0.3f;


}
void TestCorrelationFilters::testBarycentricSsdRefinement() {

	QFETCH(int, c_radius);
	QFETCH(float, subpixel_adjustement);

	int h_radius = c_radius;
	int v_radius = c_radius;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 3;

	int f = (2*v_radius + 1)*(2*h_radius + 1);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	Multidim::Array<float, 2> rand(h,w);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val = uniformDist(re);
			rand.at<Nc>(i,j) = val;
		}
	}

	Multidim::Array<float, 3> unfolded = StereoVision::Correlation::unfold(h_radius,
																		   v_radius,
																		   rand,
																		   StereoVision::PaddingMargins(0));

	Multidim::Array<float, 3> source(1,3,f);

	for (int c = 0; c < f; c++) {
		float val = 0;
		if (subpixel_adjustement < 0) {
			val = (1+subpixel_adjustement)*unfolded.value<Nc>(0,1,c) - subpixel_adjustement*unfolded.value<Nc>(0,0,c);
		} else {
			val = (1-subpixel_adjustement)*unfolded.value<Nc>(0,1,c) + subpixel_adjustement*unfolded.value<Nc>(0,2,c);
		}
		source.at<Nc>(0,0,c) = val;
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::SSD;
	Multidim::Array<float, 2> disparity = StereoVision::Correlation::refinedBarycentricDisp<matchFunc,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft>(unfolded, source, 0, 0, 3);

	float missalignement = disparity.value<Nc>(0) - (1+subpixel_adjustement);
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(1+subpixel_adjustement)));


	constexpr auto matchFunc2 = StereoVision::Correlation::matchingFunctions::ZSSD;
	Multidim::Array<float, 2> disparity2 = StereoVision::Correlation::refinedBarycentricDisp<matchFunc2,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft>(unfolded, source, 0, 0, 3);

	missalignement = disparity2.value<Nc>(0) - (1+subpixel_adjustement);
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity2.value<Nc>(0)).arg(1+subpixel_adjustement)));

}

void TestCorrelationFilters::testBarycentricSadRefinement_data() {

	QTest::addColumn<int>("c_radius");
	QTest::addColumn<float>("subpixel_adjustement");

	QTest::newRow("small_l") << 1 << -0.3f;
	QTest::newRow("small_r") << 1 << 0.3f;
	QTest::newRow("avg_l") << 3 << -0.25f;
	QTest::newRow("avg_r") << 3 << 0.25f;
	QTest::newRow("large_l") << 5 << -0.3f;
	QTest::newRow("large_r") << 5 << 0.3f;


}
void TestCorrelationFilters::testBarycentricSadRefinement() {


	QFETCH(int, c_radius);
	QFETCH(float, subpixel_adjustement);

	int h_radius = c_radius;
	int v_radius = c_radius;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 3;

	int f = (2*v_radius + 1)*(2*h_radius + 1);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	Multidim::Array<float, 2> rand(h,w);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val = uniformDist(re);
			rand.at<Nc>(i,j) = val;
		}
	}

	Multidim::Array<float, 3> unfolded = StereoVision::Correlation::unfold(h_radius,
																		   v_radius,
																		   rand,
																		   StereoVision::PaddingMargins(0));

	Multidim::Array<float, 3> source(1,3,f);

	for (int c = 0; c < f; c++) {
		float val = 0;
		if (subpixel_adjustement < 0) {
			val = (1+subpixel_adjustement)*unfolded.value<Nc>(0,1,c) - subpixel_adjustement*unfolded.value<Nc>(0,0,c);
		} else {
			val = (1-subpixel_adjustement)*unfolded.value<Nc>(0,1,c) + subpixel_adjustement*unfolded.value<Nc>(0,2,c);
		}
		source.at<Nc>(0,0,c) = val;
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::SAD;
	Multidim::Array<float, 2> disparity = StereoVision::Correlation::refinedBarycentricDisp<matchFunc,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft>(unfolded, source, 0, 0, 3);

	float missalignement = disparity.value<Nc>(0) - (1+subpixel_adjustement);
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(1+subpixel_adjustement)));


	constexpr auto matchFunc2 = StereoVision::Correlation::matchingFunctions::ZSAD;
	Multidim::Array<float, 2> disparity2 = StereoVision::Correlation::refinedBarycentricDisp<matchFunc2,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft>(unfolded, source, 0, 0, 3);

	missalignement = disparity2.value<Nc>(0) - (1+subpixel_adjustement);
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity2.value<Nc>(0)).arg(1+subpixel_adjustement)));

}

void TestCorrelationFilters::testBarycentricSymmetricNccRefinement_data() {

	QTest::addColumn<int>("c_radius");
	QTest::addColumn<SymmetricWeightsSplit>("split");

	QTest::newRow("small_l") << 1 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("small_nl") << 1 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("small_r") << 1 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("small_nr") << 1 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});
	QTest::newRow("avg_l") << 3 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("avg_nl") << 3 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("avg_r") << 3 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("avg_nr") << 3 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});
	QTest::newRow("large_l") << 5 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("large_nl") << 5 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("large_r") << 5 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("large_nr") << 5 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});


}
void TestCorrelationFilters::testBarycentricSymmetricNccRefinement() {

	QFETCH(int, c_radius);
	QFETCH(SymmetricWeightsSplit, split);

	float sumWeights = 0;
	float posWeigthed = 0;
	for (int i = 0; i < 3; i++) {
		sumWeights += split[i];
		posWeigthed += i*split[i];
	}

	if (split[1] < 0.5 or
			split[0] > 0.35 or
			split[2] > 0.35 or
			!qFuzzyCompare(sumWeights, 1.f) or
			posWeigthed < 0 or
			posWeigthed > 2) {
		QSKIP("Ill formed weight vector, unable to run test!");
	}

	int h_radius = c_radius;
	int v_radius = c_radius;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 3;

	int f = (2*v_radius + 1)*(2*h_radius + 1);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	Multidim::Array<float, 2> rand(h,w);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val = uniformDist(re);
			rand.at<Nc>(i,j) = val;
		}
	}

	Multidim::Array<float, 3> unfolded = StereoVision::Correlation::unfold(h_radius,
																		   v_radius,
																		   rand,
																		   StereoVision::PaddingMargins(0));

	//Multidim::Array<float, 2> mean = StereoVision::Correlation::channelsMean(unfolded);
	//for (int c = 0; c < f; c++) {
	//	for (int i = 0; i < 3; i++) {
	//		unfolded.at<Nc>(0,i,c) = unfolded.value<Nc>(0,i,c) - mean.value<Nc>(0,i);
	//	}
	//}

	Multidim::Array<float, 3> source(1,3,f);

	for (int c = 0; c < f; c++) {
		float val = 0;
		for (int i = 0; i < 3; i++) {val += split[i]*unfolded.value<Nc>(0,i,c);}
		source.at<Nc>(0,0,c) = val;
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZNCC;
	Multidim::Array<float, 2> disparity = StereoVision::Correlation::refinedBarycentricSymmetricDisp<matchFunc,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft,
																			1>(unfolded, source, 0, 0, 3);

	float missalignement = disparity.value<Nc>(0) - posWeigthed;
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(posWeigthed)));

}

void TestCorrelationFilters::testBarycentricSymmetricSsdRefinement_data() {

	QTest::addColumn<int>("c_radius");
	QTest::addColumn<SymmetricWeightsSplit>("split");

	QTest::newRow("small_l") << 1 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("small_nl") << 1 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("small_r") << 1 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("small_nr") << 1 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});
	QTest::newRow("avg_l") << 3 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("avg_nl") << 3 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("avg_r") << 3 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("avg_nr") << 3 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});
	QTest::newRow("large_l") << 5 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("large_nl") << 5 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("large_r") << 5 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("large_nr") << 5 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});


}
void TestCorrelationFilters::testBarycentricSymmetricSsdRefinement() {

	QFETCH(int, c_radius);
	QFETCH(SymmetricWeightsSplit, split);

	float sumWeights = 0;
	float posWeigthed = 0;
	for (int i = 0; i < 3; i++) {
		sumWeights += split[i];
		posWeigthed += i*split[i];
	}

	if (split[1] < 0.5 or
			split[0] > 0.35 or
			split[2] > 0.35 or
			!qFuzzyCompare(sumWeights, 1.f) or
			posWeigthed < 0 or
			posWeigthed > 2) {
		QSKIP("Ill formed weight vector, unable to run test!");
	}

	int h_radius = c_radius;
	int v_radius = c_radius;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 3;

	int f = (2*v_radius + 1)*(2*h_radius + 1);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	Multidim::Array<float, 2> rand(h,w);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val = uniformDist(re);
			rand.at<Nc>(i,j) = val;
		}
	}

	Multidim::Array<float, 3> unfolded = StereoVision::Correlation::unfold(h_radius,
																		   v_radius,
																		   rand,
																		   StereoVision::PaddingMargins(0));

	Multidim::Array<float, 3> source(1,3,f);

	for (int c = 0; c < f; c++) {
		float val = 0;
		for (int i = 0; i < 3; i++) {val += split[i]*unfolded.value<Nc>(0,i,c);}
		source.at<Nc>(0,0,c) = val;
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZSSD;
	Multidim::Array<float, 2> disparity = StereoVision::Correlation::refinedBarycentricSymmetricDisp<matchFunc,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft,
																			1>(unfolded, source, 0, 0, 3);

	float missalignement = disparity.value<Nc>(0) - posWeigthed;
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(posWeigthed)));

}

void TestCorrelationFilters::testBarycentricSymmetricSadRefinement_data() {

	QTest::addColumn<int>("c_radius");
	QTest::addColumn<SymmetricWeightsSplit>("split");

	QTest::newRow("small_l") << 1 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("small_nl") << 1 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("small_r") << 1 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("small_nr") << 1 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});
	QTest::newRow("avg_l") << 3 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("avg_nl") << 3 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("avg_r") << 3 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("avg_nr") << 3 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});
	QTest::newRow("large_l") << 5 << SymmetricWeightsSplit({0.2, 0.7, 0.1});
	QTest::newRow("large_nl") << 5 << SymmetricWeightsSplit({0.3, 0.8, -0.1});
	QTest::newRow("large_r") << 5 << SymmetricWeightsSplit({0.1, 0.7, 0.2});
	QTest::newRow("large_nr") << 5 << SymmetricWeightsSplit({-0.1, 0.8, 0.3});


}
void TestCorrelationFilters::testBarycentricSymmetricSadRefinement() {

	QFETCH(int, c_radius);
	QFETCH(SymmetricWeightsSplit, split);

	float sumWeights = 0;
	float posWeigthed = 0;
	for (int i = 0; i < 3; i++) {
		sumWeights += split[i];
		posWeigthed += i*split[i];
	}

	if (split[1] < 0.5 or
			split[0] > 0.35 or
			split[2] > 0.35 or
			!qFuzzyCompare(sumWeights, 1.f) or
			posWeigthed < 0 or
			posWeigthed > 2) {
		QSKIP("Ill formed weight vector, unable to run test!");
	}

	int h_radius = c_radius;
	int v_radius = c_radius;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = 2*v_radius + 1;
	int w = 2*h_radius + 3;

	int f = (2*v_radius + 1)*(2*h_radius + 1);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	Multidim::Array<float, 2> rand(h,w);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val = uniformDist(re);
			rand.at<Nc>(i,j) = val;
		}
	}

	Multidim::Array<float, 3> unfolded = StereoVision::Correlation::unfold(h_radius,
																		   v_radius,
																		   rand,
																		   StereoVision::PaddingMargins(0));

	Multidim::Array<float, 3> source(1,3,f);

	for (int c = 0; c < f; c++) {
		float val = 0;
		for (int i = 0; i < 3; i++) {val += split[i]*unfolded.value<Nc>(0,i,c);}
		source.at<Nc>(0,0,c) = val;
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZSAD;
	Multidim::Array<float, 2> disparity = StereoVision::Correlation::refinedBarycentricSymmetricDisp<matchFunc,
																			float,
																			float,
																			3,
																			StereoVision::Correlation::dispDirection::RightToLeft,
																			1>(unfolded, source, 0, 0, 3);

	float missalignement = disparity.value<Nc>(0) - posWeigthed;
	QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(posWeigthed)));
}

void TestCorrelationFilters::testInputImagesTypes() {

	QSKIP("Broken for the moment, as the type optimized cost functions are not guaranteed to lead to the same results again.");

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = 100;
	int w = 100;

	int rx = 3;
	int ry = 3;

	int dw = 12;

	std::uniform_int_distribution<uint8_t> uniformDist(0, 255);

	Multidim::Array<uint8_t, 2> rand1(h,w);
	Multidim::Array<uint8_t, 2> rand2(h,w);

	Multidim::Array<float, 2> float1(h,w);
	Multidim::Array<float, 2> float2(h,w);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			float val1 = uniformDist(re);
			float val2 = uniformDist(re);
			rand1.at<Nc>(i,j) = val1;
			rand2.at<Nc>(i,j) = val2;

			float1.at<Nc>(i,j) = float(val1);
			float2.at<Nc>(i,j) = float(val2);
		}
	}


	constexpr auto matchFunc1 = StereoVision::Correlation::matchingFunctions::NCC;
	constexpr auto matchFunc2 = StereoVision::Correlation::matchingFunctions::ZNCC;

	constexpr auto matchFunc3 = StereoVision::Correlation::matchingFunctions::SSD;
	constexpr auto matchFunc4 = StereoVision::Correlation::matchingFunctions::ZSSD;

	constexpr auto matchFunc5 = StereoVision::Correlation::matchingFunctions::SAD;
	constexpr auto matchFunc6 = StereoVision::Correlation::matchingFunctions::ZSAD;

	auto cv1_1 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc1>(rand1, rand2, rx, ry, dw);
	auto cv1_2 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc1>(float1, float2, rx, ry, dw);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			for (int d = 0; d < dw; d++) {
				float missalignement = cv1_1.at<Nc>(i,j,d) - cv1_2.at<Nc>(i,j,d);
				QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Conversion to float before and during matching gives different values at index (%1, %2, %3)").arg(i).arg(j).arg(d)));
			}
		}
	}

	auto cv2_1 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc2>(rand1, rand2, rx, ry, dw);
	auto cv2_2 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc2>(float1, float2, rx, ry, dw);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			for (int d = 0; d < dw; d++) {
				float missalignement = cv2_1.at<Nc>(i,j,d) - cv2_2.at<Nc>(i,j,d);
				QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Conversion to float before and during matching gives different values at index (%1, %2, %3)").arg(i).arg(j).arg(d)));
			}
		}
	}

	auto cv3_1 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc3>(rand1, rand2, rx, ry, dw);
	auto cv3_2 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc3>(float1, float2, rx, ry, dw);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			for (int d = 0; d < dw; d++) {
				float missalignement = cv3_1.at<Nc>(i,j,d) - cv3_2.at<Nc>(i,j,d);
				QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Conversion to float before and during matching gives different values at index (%1, %2, %3)").arg(i).arg(j).arg(d)));
			}
		}
	}

	auto cv4_1 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc4>(rand1, rand2, rx, ry, dw);
	auto cv4_2 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc4>(float1, float2, rx, ry, dw);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			for (int d = 0; d < dw; d++) {
				float missalignement = cv4_1.at<Nc>(i,j,d) - cv4_2.at<Nc>(i,j,d);
				QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Conversion to float before and during matching gives different values at index (%1, %2, %3)").arg(i).arg(j).arg(d)));
			}
		}
	}

	auto cv5_1 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc5>(rand1, rand2, rx, ry, dw);
	auto cv5_2 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc5>(float1, float2, rx, ry, dw);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			for (int d = 0; d < dw; d++) {
				float missalignement = cv5_1.at<Nc>(i,j,d) - cv5_2.at<Nc>(i,j,d);
				QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Conversion to float before and during matching gives different values at index (%1, %2, %3)").arg(i).arg(j).arg(d)));
			}
		}
	}

	auto cv6_1 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc6>(rand1, rand2, rx, ry, dw);
	auto cv6_2 = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc6>(float1, float2, rx, ry, dw);

	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			for (int d = 0; d < dw; d++) {
				float missalignement = cv6_1.at<Nc>(i,j,d) - cv6_2.at<Nc>(i,j,d);
				QVERIFY2(std::fabs(missalignement) < 1e-4, qPrintable(QString("Conversion to float before and during matching gives different values at index (%1, %2, %3)").arg(i).arg(j).arg(d)));
			}
		}
	}

}

QTEST_MAIN(TestCorrelationFilters)
#include "testCorrelationFilters.moc"
