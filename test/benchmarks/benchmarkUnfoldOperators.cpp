#include <QtTest/QtTest>

#include <correlation/unfold.h>

typedef Multidim::Array<int,2> CompressorMask;

Q_DECLARE_METATYPE(CompressorMask);

class BenchmarkUnfoldOperators: public QObject
{
	Q_OBJECT

private Q_SLOTS:
	void initTestCase();

	void benchmarkUnfoldOperator_data();
	void benchmarkUnfoldOperator();

	void benchmarkUnfoldOperatorWithCompresor_data();
	void benchmarkUnfoldOperatorWithCompresor();

private:
	std::default_random_engine re;

};

void BenchmarkUnfoldOperators::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}


void BenchmarkUnfoldOperators::benchmarkUnfoldOperator_data() {

	QTest::addColumn<int>("img_height");
	QTest::addColumn<int>("img_width");
	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");


	QTest::newRow("120x160px img - 3x3 windows") << 120 << 160 << 1 << 1;
	QTest::newRow("480x640px img - 3x3 windows") << 480 << 640 << 1 << 1;
	QTest::newRow("480x640px img - 5x5 windows") << 480 << 640 << 2 << 2;
	QTest::newRow("480x640px img - 7x7 windows") << 480 << 640 << 3 << 3;
	QTest::newRow("480x640px img - 9x9 windows") << 480 << 640 << 4 << 4;

}
void BenchmarkUnfoldOperators::benchmarkUnfoldOperator() {

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

	QBENCHMARK {
		Multidim::Array<float, 3> FV = StereoVision::Correlation::unfold(h_radius, v_radius, rand);
	}

}

void BenchmarkUnfoldOperators::benchmarkUnfoldOperatorWithCompresor_data() {

	QTest::addColumn<int>("img_height");
	QTest::addColumn<int>("img_width");
	QTest::addColumn<CompressorMask>("compressor_mask");


	QTest::newRow("120x160px img - GrPix17R3Filter windows") << 120 << 160 << StereoVision::Correlation::CompressorGenerators::GrPix17R3Filter();
	QTest::newRow("120x160px img - GrPix17R4Filter windows") << 120 << 160 << StereoVision::Correlation::CompressorGenerators::GrPix17R4Filter();
	QTest::newRow("480x640px img - GrPix17R3Filter windows") << 480 << 640 << StereoVision::Correlation::CompressorGenerators::GrPix17R3Filter();
	QTest::newRow("480x640px img - GrPix17R4Filter windows") << 480 << 640 << StereoVision::Correlation::CompressorGenerators::GrPix17R4Filter();

}
void BenchmarkUnfoldOperators::benchmarkUnfoldOperatorWithCompresor() {

	QFETCH(int, img_height);
	QFETCH(int, img_width);
	QFETCH(CompressorMask, compressor_mask);

	Multidim::Array<float, 2> rand(img_height,img_width);

	std::uniform_real_distribution<float> uniformDist(-1, 1);

	for(int i = 0; i < img_height; i++) {
		for(int j = 0; j < img_width; j++) {
			rand.at(i,j) = uniformDist(re);
		}
	}

	StereoVision::Correlation::UnFoldCompressor compressor(compressor_mask);

	QBENCHMARK {
		Multidim::Array<float, 3> FV = StereoVision::Correlation::unfold(compressor, rand);
	}
}

QTEST_MAIN(BenchmarkUnfoldOperators)
#include "benchmarkUnfoldOperators.moc"
