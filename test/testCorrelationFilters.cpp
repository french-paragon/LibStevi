#include <QtTest/QtTest>

#include "correlation/ncc.h"

class TestCorrelationNcc: public QObject
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

	void testUnfoldNCCFilter_data();
	void testUnfoldNCCFilter();

	void benchmarkUnfoldNCCFilter_data();
	void benchmarkUnfoldNCCFilter();

private:
	std::default_random_engine re;

};

void TestCorrelationNcc::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}


void TestCorrelationNcc::testMeanFilter_data() {

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

void TestCorrelationNcc::testMeanFilter() {

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



void TestCorrelationNcc::testChannelMean_data() {

	QTest::addColumn<int>("nChannels");

	QTest::newRow("few") << 5;
	QTest::newRow("some") << 50;
	QTest::newRow("many") << 500;

}
void TestCorrelationNcc::testChannelMean() {

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

void TestCorrelationNcc::testSigmaFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::newRow("small") << 1 << 1;
	QTest::newRow("wide") << 5 << 1;
	QTest::newRow("big") << 1 << 5;
	QTest::newRow("large") << 5 << 5;
}

void TestCorrelationNcc::testSigmaFilter() {

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


void TestCorrelationNcc::testChannelSigma_data() {

	QTest::addColumn<int>("nChannels");

	QTest::newRow("few") << 5;
	QTest::newRow("some") << 50;
	QTest::newRow("many") << 500;
}
void TestCorrelationNcc::testChannelSigma() {

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

void TestCorrelationNcc::testCrossCorrelationFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("small") << 1 << 1 << 5;
	QTest::newRow("avg") << 3 << 3 << 5;
	QTest::newRow("wide") << 5 << 1 << 5;
	QTest::newRow("big") << 1 << 5 << 5;
	QTest::newRow("large") << 5 << 5 << 5;
}
void TestCorrelationNcc::testCrossCorrelationFilter() {

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

	Multidim::Array<float, 3> CV = StereoVision::Correlation::ccCostVolume(randLeft, randRight, h_radius, v_radius, disp_w);

	Multidim::Array<float, 2> window1 = randRight.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(0,2*h_radius+1));

	for (int i = 0; i < disp_w; i++) {

		Multidim::Array<float, 2> window2 = randLeft.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(i,i+2*h_radius+1));

		float unnefectiveVal = InneficientCrossCorrelation(window1, window2);
		float effectiveVal = CV.value<Nc>(v_radius, h_radius, i);

		float missalignement = std::abs(unnefectiveVal - effectiveVal);
		QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed NCC is wrong (error = %1 at disp index %2)").arg(missalignement).arg(i)));
	}

}

void TestCorrelationNcc::testNCCFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("small") << 1 << 1 << 5;
	QTest::newRow("avg") << 3 << 3 << 5;
	QTest::newRow("wide") << 5 << 1 << 5;
	QTest::newRow("big") << 1 << 5 << 5;
	QTest::newRow("large") << 5 << 5 << 5;

}
void TestCorrelationNcc::testNCCFilter() {

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

	Multidim::Array<float, 3> CV = StereoVision::Correlation::nccCostVolume(randLeft, randRight, h_radius, v_radius, disp_w);

	Multidim::Array<float, 2> window1 = randRight.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(0,2*h_radius+1));

	for (int i = 0; i < disp_w; i++) {

		Multidim::Array<float, 2> window2 = randLeft.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(i,i+2*h_radius+1));

		float unnefectiveVal = InneficientNormalizedCrossCorrelation(window1, window2);
		float effectiveVal = CV.value<Nc>(v_radius, h_radius, i);

		float missalignement = std::abs(unnefectiveVal - effectiveVal);
		QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed NCC is wrong (error = %1 at disp index %2)").arg(missalignement).arg(i)));
	}
}


void TestCorrelationNcc::benchmarkNCCFilter_data() {

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
void TestCorrelationNcc::benchmarkNCCFilter() {

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

	QBENCHMARK_ONCE {
		Multidim::Array<float, 3> CV = StereoVision::Correlation::nccCostVolume(randLeft, randRight, h_radius, v_radius, disp_w);
	}
}

void TestCorrelationNcc::testUnfoldOperator_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");

	QTest::newRow("small") << 1 << 1;
	QTest::newRow("avg") << 3 << 3;
	QTest::newRow("wide") << 5 << 1;
	QTest::newRow("big") << 1 << 5;
	QTest::newRow("large") << 5 << 5;
}

void TestCorrelationNcc::testUnfoldOperator() {

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

void TestCorrelationNcc::testUnfoldNCCFilter_data() {

	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("small") << 1 << 1 << 5;
	QTest::newRow("avg") << 3 << 3 << 5;
	QTest::newRow("wide") << 5 << 1 << 5;
	QTest::newRow("big") << 1 << 5 << 5;
	QTest::newRow("large") << 5 << 5 << 5;

}
void TestCorrelationNcc::testUnfoldNCCFilter() {

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

	Multidim::Array<float, 3> CV = StereoVision::Correlation::nccUnfoldBasedCostVolume(randLeft, randRight, h_radius, v_radius, disp_w);

	Multidim::Array<float, 2> window1 = randRight.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(0,2*h_radius+1));

	for (int i = 0; i < disp_w; i++) {

		Multidim::Array<float, 2> window2 = randLeft.subView(Multidim::DimSlice(0,2*v_radius+1), Multidim::DimSlice(i,i+2*h_radius+1));

		float unnefectiveVal = InneficientNormalizedCrossCorrelation(window1, window2);
		float effectiveVal = CV.value<Nc>(v_radius, h_radius, i);

		float missalignement = std::abs(unnefectiveVal - effectiveVal);
		QVERIFY2(missalignement < 1e-3, qPrintable(QString("Reconstructed NCC is wrong (error = %1 at disp index %2)").arg(missalignement).arg(i)));
	}
}

void TestCorrelationNcc::benchmarkUnfoldNCCFilter_data() {

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
void TestCorrelationNcc::benchmarkUnfoldNCCFilter() {

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

	QBENCHMARK_ONCE {
		Multidim::Array<float, 3> CV = StereoVision::Correlation::nccUnfoldBasedCostVolume(randLeft, randRight, h_radius, v_radius, disp_w);
	}

}

QTEST_MAIN(TestCorrelationNcc)
#include "testCorrelationFilters.moc"
