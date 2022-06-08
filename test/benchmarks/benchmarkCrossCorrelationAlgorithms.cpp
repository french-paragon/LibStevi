#include <QtTest/QtTest>

#include "../test_correlation_utils.h"
#include "correlation/cross_correlations.h"
#include "correlation/hierarchical.h"
#include "correlation/sgm.h"

enum resolutions {
	VerySmall,
	StandardDefinition,
	HighDefinition
};

typedef Multidim::Array<int,2> CompressorMask;

Q_DECLARE_METATYPE(CompressorMask);

Q_DECLARE_METATYPE(StereoVision::Correlation::matchingFunctions);
Q_DECLARE_METATYPE(resolutions);

class BenchmarkCrossCorrelationAlgorithms: public QObject
{
	Q_OBJECT

private Q_SLOTS:
	void initTestCase();

	void benchmarkLocalAlgorithm_data();
	void benchmarkLocalAlgorithm();

	void benchmarkLocalAlgorithmWithCompressor_data();
	void benchmarkLocalAlgorithmWithCompressor();

	void benchmarkHierarchicalAlgorithm_data();
	void benchmarkHierarchicalAlgorithm();

	void benchmarkSemiGlobalAlgorithm_data();
	void benchmarkSemiGlobalAlgorithm();

private:
	std::default_random_engine re;

	Multidim::Array<float, 2> _source_vsmall;
	Multidim::Array<float, 2> _target_vsmall;
	Multidim::Array<int, 2> _gt_disp_vsmall;

	Multidim::Array<float, 2> _source_sd;
	Multidim::Array<float, 2> _target_sd;
	Multidim::Array<int, 2> _gt_disp_sd;

	Multidim::Array<float, 2> _source_hd;
	Multidim::Array<float, 2> _target_hd;
	Multidim::Array<int, 2> _gt_disp_hd;

	struct TestSetRefs {
		Multidim::Array<float, 2>& source;
		Multidim::Array<float, 2>& target;
		Multidim::Array<int, 2>& gt_disp;
	};

	TestSetRefs getTestSetReferences(resolutions res) {
		switch (res) {
		case VerySmall:
			return {_source_vsmall, _target_vsmall, _gt_disp_vsmall};
		case StandardDefinition:
			return {_source_sd, _target_sd, _gt_disp_sd};
		case HighDefinition:
			return {_source_hd, _target_hd, _gt_disp_hd};
		}
		return {_source_vsmall, _target_vsmall, _gt_disp_vsmall};
	}

	template<StereoVision::Correlation::matchingFunctions matchFunc>
	void BenchmarkLocalMatchingAlgorithmsImpl() {


		QFETCH(resolutions, img_resolution);
		QFETCH(int, h_radius);
		QFETCH(int, v_radius);
		QFETCH(int, disp_w);

		TestSetRefs testSet = getTestSetReferences(img_resolution);

		Multidim::Array<float, 2>& source = testSet.source;
		Multidim::Array<float, 2>& target = testSet.target;

		uint8_t h_r = h_radius;
		uint8_t v_r = v_radius;

		Multidim::Array<StereoVision::Correlation::disp_t, 2> disp;

		QBENCHMARK_ONCE {
			Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(target, source, h_r, v_r, disp_w);
			disp = StereoVision::Correlation::selectedIndexToDisp<StereoVision::Correlation::disp_t, StereoVision::Correlation::dispDirection::RightToLeft>
					(StereoVision::Correlation::extractSelectedIndex<StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::extractionStrategy>(CV), 0);
		}

		QVERIFY2(disp.shape()[0] == source.shape()[0] and disp.shape()[1] == source.shape()[1],
				qPrintable(QString("result disparity has wrong shape (%1, %2), expected (%3, %4)")
						   .arg(disp.shape()[0])
							.arg(disp.shape()[1])
							.arg(source.shape()[0])
							.arg(source.shape()[1])
							)
				);
	}

	void BenchmarkLocalMatchingAlgorithmsImplIntrm(StereoVision::Correlation::matchingFunctions matchFunc) {
		switch (matchFunc) {
		case StereoVision::Correlation::matchingFunctions::CC :
			BenchmarkLocalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::CC>();
			break;
		case StereoVision::Correlation::matchingFunctions::NCC :
			BenchmarkLocalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::NCC>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZNCC :
			BenchmarkLocalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZNCC>();
			break;
		case StereoVision::Correlation::matchingFunctions::SSD :
			BenchmarkLocalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::SSD>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSSD :
			BenchmarkLocalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZSSD>();
			break;
		case StereoVision::Correlation::matchingFunctions::SAD :
			BenchmarkLocalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::SAD>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSAD :
			BenchmarkLocalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZSAD>();
			break;
		default:
			QSKIP("Unsupported matching function for the test");
		}
	}

	template<StereoVision::Correlation::matchingFunctions matchFunc>
	void BenchmarkLocalMatchingAlgorithmsWithCompressorImpl() {


		QFETCH(resolutions, img_resolution);
		QFETCH(CompressorMask, compressor_mask);
		QFETCH(int, disp_w);

		TestSetRefs testSet = getTestSetReferences(img_resolution);

		Multidim::Array<float, 2>& source = testSet.source;
		Multidim::Array<float, 2>& target = testSet.target;

		StereoVision::Correlation::UnFoldCompressor compressor(compressor_mask);

		Multidim::Array<StereoVision::Correlation::disp_t, 2> disp;

		QBENCHMARK_ONCE {
			Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(target, source, compressor, disp_w);
			disp = StereoVision::Correlation::selectedIndexToDisp<StereoVision::Correlation::disp_t, StereoVision::Correlation::dispDirection::RightToLeft>
					(StereoVision::Correlation::extractSelectedIndex<StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::extractionStrategy>(CV), 0);
		}

		QVERIFY2(disp.shape()[0] == source.shape()[0] and disp.shape()[1] == source.shape()[1],
				qPrintable(QString("result disparity has wrong shape (%1, %2), expected (%3, %4)")
						   .arg(disp.shape()[0])
							.arg(disp.shape()[1])
							.arg(source.shape()[0])
							.arg(source.shape()[1])
							)
				);
	}

	void BenchmarkLocalMatchingAlgorithmsWithCompressorImplIntrm(StereoVision::Correlation::matchingFunctions matchFunc) {
		switch (matchFunc) {
		case StereoVision::Correlation::matchingFunctions::CC :
			BenchmarkLocalMatchingAlgorithmsWithCompressorImpl<StereoVision::Correlation::matchingFunctions::CC>();
			break;
		case StereoVision::Correlation::matchingFunctions::NCC :
			BenchmarkLocalMatchingAlgorithmsWithCompressorImpl<StereoVision::Correlation::matchingFunctions::NCC>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZNCC :
			BenchmarkLocalMatchingAlgorithmsWithCompressorImpl<StereoVision::Correlation::matchingFunctions::ZNCC>();
			break;
		case StereoVision::Correlation::matchingFunctions::SSD :
			BenchmarkLocalMatchingAlgorithmsWithCompressorImpl<StereoVision::Correlation::matchingFunctions::SSD>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSSD :
			BenchmarkLocalMatchingAlgorithmsWithCompressorImpl<StereoVision::Correlation::matchingFunctions::ZSSD>();
			break;
		case StereoVision::Correlation::matchingFunctions::SAD :
			BenchmarkLocalMatchingAlgorithmsWithCompressorImpl<StereoVision::Correlation::matchingFunctions::SAD>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSAD :
			BenchmarkLocalMatchingAlgorithmsWithCompressorImpl<StereoVision::Correlation::matchingFunctions::ZSAD>();
			break;
		default:
			QSKIP("Unsupported matching function for the test");
		}
	}

	template<StereoVision::Correlation::matchingFunctions matchFunc, int depth>
	void BenchmarkHierarchicalAlgorithmsImpl() {

		QFETCH(resolutions, img_resolution);
		QFETCH(int, h_radius);
		QFETCH(int, v_radius);
		QFETCH(int, disp_w);

		TestSetRefs testSet = getTestSetReferences(img_resolution);

		Multidim::Array<float, 2>& source = testSet.source;
		Multidim::Array<float, 2>& target = testSet.target;

		uint8_t h_r = h_radius;
		uint8_t v_r = v_radius;

		StereoVision::Correlation::OffsetedCostVolume<float> result;

		QBENCHMARK_ONCE {
		result = StereoVision::Correlation::hiearchicalTruncatedCostVolume<matchFunc, depth>
				(target,
				 source,
				 h_r,
				 v_r,
				 disp_w);
		}

		QVERIFY2(result.disp_estimate.shape()[0] == source.shape()[0] and result.disp_estimate.shape()[1] == source.shape()[1],
				qPrintable(QString("result disparity has wrong shape (%1, %2), expected (%3, %4)")
						   .arg(result.disp_estimate.shape()[0])
							.arg(result.disp_estimate.shape()[1])
							.arg(source.shape()[0])
							.arg(source.shape()[1])
							)
				);
	}

	template<int depth>
	void BenchmarkHierarchicalAlgorithmsImplIntrm(StereoVision::Correlation::matchingFunctions matchFunc) {
		switch (matchFunc) {
		case StereoVision::Correlation::matchingFunctions::CC :
			BenchmarkHierarchicalAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::CC, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::NCC :
			BenchmarkHierarchicalAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::NCC, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZNCC :
			BenchmarkHierarchicalAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZNCC, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::SSD :
			BenchmarkHierarchicalAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::SSD, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSSD :
			BenchmarkHierarchicalAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZSSD, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::SAD :
			BenchmarkHierarchicalAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::SAD, depth>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSAD :
			BenchmarkHierarchicalAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZSAD, depth>();
			break;
		default:
			QSKIP("Unsupported matching function for the test");
		}
	}


	template<StereoVision::Correlation::matchingFunctions matchFunc, int nDirections>
	void BenchmarkSemiGlobalMatchingAlgorithmsImpl() {


		QFETCH(resolutions, img_resolution);
		QFETCH(int, h_radius);
		QFETCH(int, v_radius);
		QFETCH(int, disp_w);

		TestSetRefs testSet = getTestSetReferences(img_resolution);

		Multidim::Array<float, 2>& source = testSet.source;
		Multidim::Array<float, 2>& target = testSet.target;

		uint8_t h_r = h_radius;
		uint8_t v_r = v_radius;

		//small P1 and P2
		float P1 = 0.001;
		float P2 = 0.01;
		float Pout = 100;

		Multidim::Array<StereoVision::Correlation::disp_t, 2> disp;

		QBENCHMARK_ONCE {
			Multidim::Array<float, 3> CV = StereoVision::Correlation::unfoldBasedCostVolume<matchFunc>(target, source, h_r, v_r, disp_w);
			Multidim::Array<float, 3> SGM_CV = StereoVision::Correlation::sgmCostVolume<nDirections, StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::extractionStrategy>
					(CV, P1, P2, StereoVision::Margins(), Pout);
			disp = StereoVision::Correlation::selectedIndexToDisp<StereoVision::Correlation::disp_t, StereoVision::Correlation::dispDirection::RightToLeft>
					(StereoVision::Correlation::extractSelectedIndex<StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::extractionStrategy>(SGM_CV), 0);
		}

		QVERIFY2(disp.shape()[0] == source.shape()[0] and disp.shape()[1] == source.shape()[1],
				qPrintable(QString("result disparity has wrong shape (%1, %2), expected (%3, %4)")
						   .arg(disp.shape()[0])
							.arg(disp.shape()[1])
							.arg(source.shape()[0])
							.arg(source.shape()[1])
							)
				);
	}

	template<int nDirections>
	void BenchmarkSemiGlobalMatchingAlgorithmsImplIntrm(StereoVision::Correlation::matchingFunctions matchFunc) {
		switch (matchFunc) {
		case StereoVision::Correlation::matchingFunctions::CC :
			BenchmarkSemiGlobalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::CC, nDirections>();
			break;
		case StereoVision::Correlation::matchingFunctions::NCC :
			BenchmarkSemiGlobalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::NCC, nDirections>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZNCC :
			BenchmarkSemiGlobalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZNCC, nDirections>();
			break;
		case StereoVision::Correlation::matchingFunctions::SSD :
			BenchmarkSemiGlobalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::SSD, nDirections>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSSD :
			BenchmarkSemiGlobalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZSSD, nDirections>();
			break;
		case StereoVision::Correlation::matchingFunctions::SAD :
			BenchmarkSemiGlobalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::SAD, nDirections>();
			break;
		case StereoVision::Correlation::matchingFunctions::ZSAD :
			BenchmarkSemiGlobalMatchingAlgorithmsImpl<StereoVision::Correlation::matchingFunctions::ZSAD, nDirections>();
			break;
		default:
			QSKIP("Unsupported matching function for the test");
		}
	}

};

void BenchmarkCrossCorrelationAlgorithms::initTestCase() {

	std::random_device rd;
	re.seed(rd());

	PatchBaseMatchingTestPair smallSet = generateParallaxSquareImage(48, 64, 16, 16, 24, 2, 8, re);

	_source_vsmall = smallSet.source;
	_target_vsmall = smallSet.target;
	_gt_disp_vsmall = smallSet.gt_disp;

	PatchBaseMatchingTestPair sdSet = generateParallaxSquareImage(480, 640, 160, 160, 24, 8, 32, re);

	_source_sd = sdSet.source;
	_target_sd = sdSet.target;
	_gt_disp_sd = sdSet.gt_disp;

	PatchBaseMatchingTestPair hdSet = generateParallaxSquareImage(1080, 1920, 320, 320, 380, 8, 64, re);

	_source_hd = hdSet.source;
	_target_hd = hdSet.target;
	_gt_disp_hd = hdSet.gt_disp;

}

void BenchmarkCrossCorrelationAlgorithms::benchmarkHierarchicalAlgorithm_data() {
	QTest::addColumn<int>("depth");
	QTest::addColumn<StereoVision::Correlation::matchingFunctions>("matchFunc");
	QTest::addColumn<resolutions>("img_resolution");
	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("Very small img - level1 - 5x5 windows - level1 - disp 16 - zncc") << 1 << StereoVision::Correlation::matchingFunctions::ZNCC << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - level2 - 5x5 windows - level2 - disp 16 - zncc") << 2 << StereoVision::Correlation::matchingFunctions::ZNCC << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - level3 - 5x5 windows - level3 - disp 16 - zncc") << 3 << StereoVision::Correlation::matchingFunctions::ZNCC << VerySmall << 2 << 2 << 16;

	#ifndef NDEBUG
	//no large benchmarching in debug mode, they are too slow.
	return;
	#endif

	QTest::newRow("Standard definition img - 5x5 windows - level1 - disp 20 - zncc") << 1 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 20;
	QTest::newRow("Standard definition img - 5x5 windows - level2 - disp 20 - zncc") << 2 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 20;
	QTest::newRow("Standard definition img - 5x5 windows - level3 - disp 20 - zncc") << 3 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 20;

	QTest::newRow("Standard definition img - 5x5 windows - level1 - disp 160 - zncc") << 1 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - level2 - disp 160 - zncc") << 2 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - level3 - disp 160 - zncc") << 3 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 160;

	QTest::newRow("Standard definition img - 5x5 windows - level1 - disp 300 - zncc") << 1 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 300;
	QTest::newRow("Standard definition img - 5x5 windows - level2 - disp 300 - zncc") << 2 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 300;
	QTest::newRow("Standard definition img - 5x5 windows - level3 - disp 300 - zncc") << 3 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 300;

	QTest::newRow("Standard definition img - 7x7 windows - level1 - disp 20 - zncc") << 1 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 3 << 3 << 20;
	QTest::newRow("Standard definition img - 7x7 windows - level2 - disp 20 - zncc") << 2 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 3 << 3 << 20;
	QTest::newRow("Standard definition img - 7x7 windows - level3 - disp 20 - zncc") << 3 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 3 << 3 << 20;

	QTest::newRow("Standard definition img - 7x7 windows - level1 - disp 160 - zncc") << 1 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 3 << 3 << 160;
	QTest::newRow("Standard definition img - 7x7 windows - level2 - disp 160 - zncc") << 2 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 3 << 3 << 160;
	QTest::newRow("Standard definition img - 7x7 windows - level3 - disp 160 - zncc") << 3 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 3 << 3 << 160;

	QTest::newRow("High definition img - 7x7 windows - level1 - disp 320 - zncc") << 1 << StereoVision::Correlation::matchingFunctions::ZNCC << HighDefinition << 3 << 3 << 320;
	QTest::newRow("High definition img - 7x7 windows - level2 - disp 320 - zncc") << 2 << StereoVision::Correlation::matchingFunctions::ZNCC << HighDefinition << 3 << 3 << 320;
	QTest::newRow("High definition img - 7x7 windows - level3 - disp 320 - zncc") << 3 << StereoVision::Correlation::matchingFunctions::ZNCC << HighDefinition << 3 << 3 << 320;
}
void BenchmarkCrossCorrelationAlgorithms::benchmarkHierarchicalAlgorithm() {

	QFETCH(int, depth);
	QFETCH(StereoVision::Correlation::matchingFunctions, matchFunc);


	switch (depth) {
	case 1:
		BenchmarkHierarchicalAlgorithmsImplIntrm<1>(matchFunc);
		break;
	case 2:
		BenchmarkHierarchicalAlgorithmsImplIntrm<2>(matchFunc);
		break;
	case 3:
		BenchmarkHierarchicalAlgorithmsImplIntrm<3>(matchFunc);
		break;
	default:
		QSKIP("This test suit support depth ranging from 1 to 3 for the hierarchical matching unit test");
	}

}

void BenchmarkCrossCorrelationAlgorithms::benchmarkLocalAlgorithm_data() {

	QTest::addColumn<StereoVision::Correlation::matchingFunctions>("matchFunc");
	QTest::addColumn<resolutions>("img_resolution");
	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("Very small img - 5x5 windows - disp 16 - ncc") << StereoVision::Correlation::matchingFunctions::NCC << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - 5x5 windows - disp 16 - zncc") << StereoVision::Correlation::matchingFunctions::ZNCC << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - 5x5 windows - disp 16 - ssd") << StereoVision::Correlation::matchingFunctions::SSD << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - 5x5 windows - disp 16 - zssd") << StereoVision::Correlation::matchingFunctions::ZSSD << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - 5x5 windows - disp 16 - sad") << StereoVision::Correlation::matchingFunctions::SAD << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - 5x5 windows - disp 16 - zsad") << StereoVision::Correlation::matchingFunctions::ZSAD << VerySmall << 2 << 2 << 16;

	#ifndef NDEBUG
	//no large benchmarching in debug mode, they are too slow.
	return;
	#endif

	//standard definition
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - ncc") << StereoVision::Correlation::matchingFunctions::NCC << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - zncc") << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - ssd") << StereoVision::Correlation::matchingFunctions::SSD << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - zssd") << StereoVision::Correlation::matchingFunctions::ZSSD << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - sad") << StereoVision::Correlation::matchingFunctions::SAD << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - zsad") << StereoVision::Correlation::matchingFunctions::ZSAD << StandardDefinition << 2 << 2 << 160;

	//high definition
	QTest::newRow("High definition img - 7x7 windows - disp 320 - ncc") << StereoVision::Correlation::matchingFunctions::NCC << HighDefinition << 3 << 3 << 320;
	QTest::newRow("High definition img - 7x7 windows - disp 320 - zncc") << StereoVision::Correlation::matchingFunctions::ZNCC << HighDefinition << 3 << 3 << 320;
	QTest::newRow("High definition img - 7x7 windows - disp 320 - ssd") << StereoVision::Correlation::matchingFunctions::SSD << HighDefinition << 3 << 3 << 320;
	QTest::newRow("High definition img - 7x7 windows - disp 320 - zssd") << StereoVision::Correlation::matchingFunctions::ZSSD << HighDefinition << 3 << 3 << 320;
	QTest::newRow("High definition img - 7x7 windows - disp 320 - sad") << StereoVision::Correlation::matchingFunctions::SAD << HighDefinition << 3 << 3 << 320;
	QTest::newRow("High definition img - 7x7 windows - disp 320 - zsad") << StereoVision::Correlation::matchingFunctions::ZSAD << HighDefinition << 3 << 3 << 320;

}
void BenchmarkCrossCorrelationAlgorithms::benchmarkLocalAlgorithm() {

	QFETCH(StereoVision::Correlation::matchingFunctions, matchFunc);

	BenchmarkLocalMatchingAlgorithmsImplIntrm(matchFunc);
}

void BenchmarkCrossCorrelationAlgorithms::benchmarkLocalAlgorithmWithCompressor_data() {

	QTest::addColumn<StereoVision::Correlation::matchingFunctions>("matchFunc");
	QTest::addColumn<resolutions>("img_resolution");
	QTest::addColumn<CompressorMask>("compressor_mask");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("Very small img - GrPix17R3 windows - disp 16 - zncc")
			<< StereoVision::Correlation::matchingFunctions::ZNCC
			<< VerySmall
			<< StereoVision::Correlation::CompressorGenerators::GrPix17R3Filter()
			<< 16;
	QTest::newRow("Very small img - GrPix17R4 windows - disp 16 - zncc")
			<< StereoVision::Correlation::matchingFunctions::ZNCC
			<< VerySmall
			<< StereoVision::Correlation::CompressorGenerators::GrPix17R4Filter()
			<< 16;

	#ifndef NDEBUG
	//no large benchmarching in debug mode, they are too slow.
	return;
	#endif

	//standard definition
	QTest::newRow("Standard definition img - GrPix17R3 windows - disp 160 - zncc")
			<< StereoVision::Correlation::matchingFunctions::ZNCC
			<< StandardDefinition
			<< StereoVision::Correlation::CompressorGenerators::GrPix17R3Filter()
			<< 160;
	QTest::newRow("Standard definition img - GrPix17R4 windows - disp 160 - zncc")
			<< StereoVision::Correlation::matchingFunctions::ZNCC
			<< StandardDefinition
			<< StereoVision::Correlation::CompressorGenerators::GrPix17R4Filter()
			<< 160;

	//high definition
	QTest::newRow("High definition img - GrPix17R3 windows - disp 320 - zncc")
			<< StereoVision::Correlation::matchingFunctions::ZNCC
			<< HighDefinition
			<< StereoVision::Correlation::CompressorGenerators::GrPix17R3Filter()
			<< 320;
	QTest::newRow("High definition img - GrPix17R4 windows - disp 320 - zncc")
			<< StereoVision::Correlation::matchingFunctions::ZNCC
			<< HighDefinition
			<< StereoVision::Correlation::CompressorGenerators::GrPix17R4Filter()
			<< 320;

}
void BenchmarkCrossCorrelationAlgorithms::benchmarkLocalAlgorithmWithCompressor() {

	QFETCH(StereoVision::Correlation::matchingFunctions, matchFunc);

	BenchmarkLocalMatchingAlgorithmsWithCompressorImplIntrm(matchFunc);
}

void BenchmarkCrossCorrelationAlgorithms::benchmarkSemiGlobalAlgorithm_data() {

	QTest::addColumn<int>("nDirections");
	QTest::addColumn<StereoVision::Correlation::matchingFunctions>("matchFunc");
	QTest::addColumn<resolutions>("img_resolution");
	QTest::addColumn<int>("h_radius");
	QTest::addColumn<int>("v_radius");
	QTest::addColumn<int>("disp_w");

	QTest::newRow("Very small img - 5x5 windows - disp 16 - 4 directions - zncc") << 4 << StereoVision::Correlation::matchingFunctions::ZNCC << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - 5x5 windows - disp 16 - 8 directions - zncc") << 8 << StereoVision::Correlation::matchingFunctions::ZNCC << VerySmall << 2 << 2 << 16;
	QTest::newRow("Very small img - 5x5 windows - disp 16 - 16 directions - zncc") << 16 << StereoVision::Correlation::matchingFunctions::ZNCC << VerySmall << 2 << 2 << 16;

	#ifndef NDEBUG
	//no large benchmarching in debug mode, they are too slow.
	return;
	#endif

	//standard definition
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - 4 directions - zncc") << 4 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - 8 directions - zncc") << 8 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 160;
	QTest::newRow("Standard definition img - 5x5 windows - disp 160 - 16 directions - zncc") << 16 << StereoVision::Correlation::matchingFunctions::ZNCC << StandardDefinition << 2 << 2 << 160;

	//high definition (disable for the moment -> too slow
	//QTest::newRow("High definition img - 7x7 windows - disp 320 - 4 directions - zncc") << 4 << StereoVision::Correlation::matchingFunctions::ZNCC << HighDefinition << 3 << 3 << 320;
	//QTest::newRow("High definition img - 7x7 windows - disp 320 - 8 directions - zncc") << 8 << StereoVision::Correlation::matchingFunctions::ZNCC << HighDefinition << 3 << 3 << 320;
	//QTest::newRow("High definition img - 7x7 windows - disp 320 - 16 directions - zncc") << 16 << StereoVision::Correlation::matchingFunctions::ZNCC << HighDefinition << 3 << 3 << 320;

}
void BenchmarkCrossCorrelationAlgorithms::benchmarkSemiGlobalAlgorithm() {

	QFETCH(int, nDirections);
	QFETCH(StereoVision::Correlation::matchingFunctions, matchFunc);


	switch (nDirections) {
	case 4:
		BenchmarkSemiGlobalMatchingAlgorithmsImplIntrm<4>(matchFunc);
		break;
	case 8:
		BenchmarkSemiGlobalMatchingAlgorithmsImplIntrm<8>(matchFunc);
		break;
	case 16:
		BenchmarkSemiGlobalMatchingAlgorithmsImplIntrm<16>(matchFunc);
		break;
	default:
		QSKIP("The SGM routine support only 4, 8 or 16 directions!");
	}

}

QTEST_MAIN(BenchmarkCrossCorrelationAlgorithms)
#include "benchmarkCrossCorrelationAlgorithms.moc"
