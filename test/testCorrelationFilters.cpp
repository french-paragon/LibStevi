#include <QtTest/QtTest>

#include "correlation/cross_correlations.h"

typedef Multidim::Array<int,2> CompressorMask;
typedef std::array<float,3> SymmetricWeightsSplit;

Q_DECLARE_METATYPE(CompressorMask);
Q_DECLARE_METATYPE(SymmetricWeightsSplit);

class TestCorrelationFilters: public QObject
{
	Q_OBJECT
private:

	float InneficientCrossCorrelation(Multidim::Array<float, 2> const& windows1,
												Multidim::Array<float, 2> const& windows2) {

		constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

		int h = windows1.shape()[0];
		int w = windows1.shape()[1];

		if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
			return std::nanf("");
		}

		float mean1 = 0;
		float mean2 = 0;

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				mean1 += windows1.value<Nc>(i,j);
				mean2 += windows2.value<Nc>(i,j);

			}
		}

		mean1 /= h*w;
		mean2 /= h*w;

		float cc = 0;

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				float v1 = windows1.value<Nc>(i,j) - mean1;
				float v2 = windows2.value<Nc>(i,j) - mean2;

				cc += v1*v2;

			}
		}

		return cc;
	}

	float InneficientNormalizedCrossCorrelation(Multidim::Array<float, 2> const& windows1,
												Multidim::Array<float, 2> const& windows2) {

		constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

		int h = windows1.shape()[0];
		int w = windows1.shape()[1];

		if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
			return std::nanf("");
		}

		float mean1 = 0;
		float mean2 = 0;

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				mean1 += windows1.value<Nc>(i,j);
				mean2 += windows2.value<Nc>(i,j);

			}
		}

		mean1 /= h*w;
		mean2 /= h*w;

		float cc = 0;
		float s1 = 0;
		float s2 = 0;

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				float v1 = windows1.value<Nc>(i,j) - mean1;
				float v2 = windows2.value<Nc>(i,j) - mean2;

				cc += v1*v2;
				s1 += v1*v1;
				s2 += v2*v2;

			}
		}

		return cc/(sqrtf(s1)*sqrtf(s2));

	}

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

	void benchmarkNCCFilter_data();
	void benchmarkNCCFilter();

	void testUnfoldOperator_data();
	void testUnfoldOperator();

	void benchmarkUnfoldOperator_data();
	void benchmarkUnfoldOperator();

	void testUnfoldNCCFilter_data();
	void testUnfoldNCCFilter();

	void benchmarkUnfoldNCCFilter_data();
	void benchmarkUnfoldNCCFilter();

	void testUnfoldCompressor_data();
	void testUnfoldCompressor();

	void benchmarkCompressedNCCFilter_data();
	void benchmarkCompressedNCCFilter();

	void testBarycentricNccRefinement_data();
	void testBarycentricNccRefinement();

	void testBarycentricSsdRefinement_data();
	void testBarycentricSsdRefinement();

	void testBarycentricSadRefinement_data();
	void testBarycentricSadRefinement();

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

	Multidim::Array<float, 2> sigm = StereoVision::Correlation::channelsSigma(rand);

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

		float unnefectiveVal = InneficientCrossCorrelation(window1, window2);
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

		float unnefectiveVal = InneficientNormalizedCrossCorrelation(window1, window2);
		float effectiveVal = CV.value<Nc>(v_radius, h_radius, i);

		float missalignement = std::abs(unnefectiveVal - effectiveVal);
		QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed NCC is wrong (error = %1 at disp index %2)").arg(missalignement).arg(i)));
	}
}


void TestCorrelationFilters::benchmarkNCCFilter_data() {

	QTest::addColumn<int>("img_height");
	QTest::addColumn<int>("img_width");
	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");


	QTest::newRow("small") << 120 << 160 << 1 << 1 << 5;
	QTest::newRow("small box") << 480 << 640 << 1 << 1 << 20;
	QTest::newRow("medium box") << 480 << 640 << 2 << 2 << 40;
	QTest::newRow("box") << 480 << 640 << 3 << 3 << 50;
	QTest::newRow("large box") << 480 << 640 << 4 << 4 << 50;
}
void TestCorrelationFilters::benchmarkNCCFilter() {

	#ifndef NDEBUG
	QSKIP("No benchmarking in debug mode!");
	return;
	#endif

	QFETCH(int, img_height);
	QFETCH(int, img_width);
	QFETCH(int, h_radius);
	QFETCH(int, v_radius);
	QFETCH(int, disp_w);

	Multidim::Array<float, 2> randLeft(img_height,img_width);
	Multidim::Array<float, 2> randRight(img_height,img_width);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < img_height; i++) {
		for(int j = 0; j < img_width; j++) {
			randLeft.at(i,j) = uniformDist(re);
			randRight.at(i,j) = uniformDist(re);
		}
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZNCC;
	QBENCHMARK_ONCE {
		Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(randLeft, randRight, h_radius, v_radius, disp_w);
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

	Multidim::Array<float, 3> unfolded = StereoVision::Correlation::unfold(h_radius, v_radius, rand, StereoVision::Correlation::PaddingMargins(0));

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

	Multidim::Array<float, 3> unfoldedPadded = StereoVision::Correlation::unfold(h_radius, v_radius, rand, StereoVision::Correlation::PaddingMargins());

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


void TestCorrelationFilters::benchmarkUnfoldOperator_data() {

	QTest::addColumn<int>("img_height");
	QTest::addColumn<int>("img_width");
	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");


	QTest::newRow("small") << 120 << 160 << 1 << 1;
	QTest::newRow("small box") << 480 << 640 << 1 << 1;
	QTest::newRow("medium box") << 480 << 640 << 2 << 2;
	QTest::newRow("box") << 480 << 640 << 3 << 3;
	QTest::newRow("large box") << 480 << 640 << 4 << 4;

}
void TestCorrelationFilters::benchmarkUnfoldOperator() {

	#ifndef NDEBUG
	QSKIP("No benchmarking in debug mode!");
	return;
	#endif

	QFETCH(int, img_height);
	QFETCH(int, img_width);
	QFETCH(int, h_radius);
	QFETCH(int, v_radius);

	Multidim::Array<float, 2> rand(img_height,img_width);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < img_height; i++) {
		for(int j = 0; j < img_width; j++) {
			rand.at(i,j) = uniformDist(re);
		}
	}

	QBENCHMARK_ONCE {
		Multidim::Array<float, 3> CV = StereoVision::Correlation::unfold(h_radius, v_radius, rand);
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

		float unnefectiveVal = InneficientNormalizedCrossCorrelation(window1, window2);
		float effectiveVal = CV.value<Nc>(v_radius, h_radius, i);

		float missalignement = std::abs(unnefectiveVal - effectiveVal);
		QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed NCC is wrong (error = %1 at disp index %2)").arg(missalignement).arg(i)));
	}
}

void TestCorrelationFilters::benchmarkUnfoldNCCFilter_data() {

	QTest::addColumn<int>("img_height");
	QTest::addColumn<int>("img_width");
	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");


	QTest::newRow("small") << 120 << 160 << 1 << 1 << 5;
	QTest::newRow("small box") << 480 << 640 << 1 << 1 << 20;
	QTest::newRow("medium box") << 480 << 640 << 2 << 2 << 40;
	QTest::newRow("box") << 480 << 640 << 3 << 3 << 50;
	QTest::newRow("large box") << 480 << 640 << 4 << 4 << 50;
}
void TestCorrelationFilters::benchmarkUnfoldNCCFilter() {

	#ifndef NDEBUG
	QSKIP("No benchmarking in debug mode!");
	return;
	#endif

	QFETCH(int, img_height);
	QFETCH(int, img_width);
	QFETCH(int, h_radius);
	QFETCH(int, v_radius);
	QFETCH(int, disp_w);

	Multidim::Array<float, 2> randLeft(img_height,img_width);
	Multidim::Array<float, 2> randRight(img_height,img_width);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < img_height; i++) {
		for(int j = 0; j < img_width; j++) {
			randLeft.at(i,j) = uniformDist(re);
			randRight.at(i,j) = uniformDist(re);
		}
	}

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZNCC;
	QBENCHMARK_ONCE {
		Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(randLeft, randRight, h_radius, v_radius, disp_w);
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

	Multidim::Array<float, 3> unfoldedCompressed = StereoVision::Correlation::unfold(compressor, rand, StereoVision::Correlation::PaddingMargins(0));

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


void TestCorrelationFilters::benchmarkCompressedNCCFilter_data() {

	QTest::addColumn<int>("img_height");
	QTest::addColumn<int>("img_width");
	QTest::addColumn<CompressorMask>("compressor_mask");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("640x480-GrPix17R3Filter-w20") << 480 << 640 << StereoVision::Correlation::CompressorGenerators::GrPix17R3Filter() << 20;
	QTest::newRow("640x480-GrPix17R4Filter-w20") << 480 << 640 << StereoVision::Correlation::CompressorGenerators::GrPix17R4Filter() << 20;
}
void TestCorrelationFilters::benchmarkCompressedNCCFilter() {

	#ifndef NDEBUG
	QSKIP("No benchmarking in debug mode!");
	return;
	#endif

	QFETCH(int, img_height);
	QFETCH(int, img_width);
	QFETCH(CompressorMask, compressor_mask);
	QFETCH(int, disp_w);

	Multidim::Array<float, 2> randLeft(img_height,img_width);
	Multidim::Array<float, 2> randRight(img_height,img_width);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < img_height; i++) {
		for(int j = 0; j < img_width; j++) {
			randLeft.at(i,j) = uniformDist(re);
			randRight.at(i,j) = uniformDist(re);
		}
	}

	StereoVision::Correlation::UnFoldCompressor compressor(compressor_mask);

	constexpr auto matchFunc = StereoVision::Correlation::matchingFunctions::ZNCC;
	QBENCHMARK_ONCE {
		Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(randLeft, randRight, compressor, disp_w);
	}
}

void TestCorrelationFilters::testBarycentricNccRefinement_data() {

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
void TestCorrelationFilters::testBarycentricNccRefinement() {

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
																		   StereoVision::Correlation::PaddingMargins(0));

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
	QVERIFY2(missalignement < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(posWeigthed)));
}

void TestCorrelationFilters::testBarycentricSsdRefinement_data() {

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
void TestCorrelationFilters::testBarycentricSsdRefinement() {

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
																		   StereoVision::Correlation::PaddingMargins(0));

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
	QVERIFY2(missalignement < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(posWeigthed)));
}

void TestCorrelationFilters::testBarycentricSadRefinement_data() {

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
void TestCorrelationFilters::testBarycentricSadRefinement() {

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
																		   StereoVision::Correlation::PaddingMargins(0));

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
	QVERIFY2(missalignement < 1e-4, qPrintable(QString("Matching not done properly (subpixel position of first feature vector = %1, expected = %2)").arg(disparity.value<Nc>(0)).arg(posWeigthed)));
}

QTEST_MAIN(TestCorrelationFilters)
#include "testCorrelationFilters.moc"
