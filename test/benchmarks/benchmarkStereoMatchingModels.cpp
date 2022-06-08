#include <QTest>

#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QMap>

#include <MultidimArrays/MultidimArrays.h>

#include <cstdint>

#include "correlation/cross_correlations.h"
#include "correlation/hierarchical.h"
#include "correlation/patchmatch.h"
#include "io/image_io.h"
#include "utils/randomcache.h"

using namespace StereoVision::Correlation;

typedef Multidim::Array<disp_t, 2> (*dispFuncFloat)(Multidim::Array<float, 3> const&, Multidim::Array<float, 3> const&, StereoVision::Random::NumbersCache<int>*);
typedef Multidim::Array<disp_t, 2> (*dispFuncByte)(Multidim::Array<uint8_t, 3> const&, Multidim::Array<uint8_t, 3> const&, StereoVision::Random::NumbersCache<int>*);

struct dispFuncF {
	dispFuncFloat dispFunc;
};

struct dispFuncB {
	dispFuncByte dispFunc;
};

Q_DECLARE_METATYPE(dispFuncF);
Q_DECLARE_METATYPE(dispFuncB);

class BenchmarkStereoMatchingModels: public QObject{

	Q_OBJECT
private Q_SLOTS:
	void initTestCase();

	void benchmarkFloatDispFunc_data();
	void benchmarkFloatDispFunc();

	void benchmarkUint8DispFunc_data();
	void benchmarkUint8DispFunc();

private:

	struct testImageInfo {

		testImageInfo() {
			baseName = "";
			left_rgb_path = "";
			right_rgb_path = "";
			left_nir_path = "";
			right_nir_path = "";
			left_disp_path = "";
			right_disp_path = "";
		}

		inline bool isSet() const {
			return !baseName.isEmpty() and
					!left_rgb_path.isEmpty() and
					!right_rgb_path.isEmpty() and
					!left_nir_path.isEmpty() and
					!right_nir_path.isEmpty() and
					!left_disp_path.isEmpty() and
					!right_disp_path.isEmpty();
		}

		QString baseName;
		QString left_rgb_path;
		QString right_rgb_path;
		QString left_nir_path;
		QString right_nir_path;
		QString left_disp_path;
		QString right_disp_path;
	};

	QVector<testImageInfo> _test_imgs;

	std::default_random_engine re;
	StereoVision::Random::NumbersCache<int> _random_cache;
};

void BenchmarkStereoMatchingModels::initTestCase() {

	QString source_dir = "@CMAKE_SOURCE_DIR@";
	QString exec_dir = "@CMAKE_CURRENT_BINARY_DIR@";

	_test_imgs.clear();

	QDir src(source_dir);
	bool found = src.cd("test/test_data/stereo_images");

	if (!found) {
		QSKIP("Test could no find test data, make sure you download it in your source tree (see ReadMe.md for more infos).");
	}

	QDirIterator it(src);

	QMap<QString, testImageInfo> testImgInfos;

	while (it.hasNext()) {
		QString path = it.next();
		QFileInfo info(path);

		if (path.toLower().endsWith(".bmp") or
				path.toLower().endsWith(".pfm")) {
			QString basename = info.baseName();
			QStringList split = basename.split("_");

			if (split.size() != 3) {
				continue;
			}

			QString symbol = split[0].toLower();
			QString type = split[1].toLower();
			QString side = split[2].toLower();

			if (type != "rgb" and type != "nir" and type != "disp") {
				continue;
			}

			if (side != "left" and side != "right") {
				continue;
			}

			if (!testImgInfos.contains(symbol)) {
				testImgInfos[symbol] = testImageInfo();
				testImgInfos[symbol].baseName = symbol;
			}

			if (type == "rgb") {
				if (side == "left") {
					testImgInfos[symbol].left_rgb_path = path;
				}
				if (side == "right") {
					testImgInfos[symbol].right_rgb_path = path;
				}
			}

			if (type == "nir") {
				if (side == "left") {
					testImgInfos[symbol].left_nir_path = path;
				}
				if (side == "right") {
					testImgInfos[symbol].right_nir_path = path;
				}
			}

			if (type == "disp") {
				if (side == "left") {
					testImgInfos[symbol].left_disp_path = path;
				}
				if (side == "right") {
					testImgInfos[symbol].right_disp_path = path;
				}
			}

		}
	}

	for (testImageInfo const& info : testImgInfos) {
		if (info.isSet()) {
			_test_imgs.push_back(info);
		}
	}

	if (_test_imgs.isEmpty()) {
		QSKIP("found no images to process");
	}

	std::random_device rd;
	re.seed(rd());

	_random_cache = StereoVision::Random::NumbersCache<int>(1<<16, re);
}

template<matchingFunctions matchFunc, typename T_IMG, int searchRadius, int searchDist, int nIter, int nRandomSearch>
Multidim::Array<disp_t, 2> testPatchMatch(Multidim::Array<T_IMG, 3> const& img_left,
										  Multidim::Array<T_IMG, 3> const& img_right,
										  StereoVision::Random::NumbersCache<int>* rngCache) {

	Multidim::Array<T_IMG, 3> fVolLeft = unfold<T_IMG, T_IMG>(searchRadius, searchRadius, img_left);
	Multidim::Array<T_IMG, 3> fVolRight = unfold<T_IMG, T_IMG>(searchRadius, searchRadius, img_right);

	std::optional<StereoVision::Random::NumbersCache<int>> optCache = std::nullopt;
	if (rngCache != nullptr) {
		optCache = *rngCache;
	}

	Multidim::Array<disp_t, 3> out = patchMatch<matchFunc, 1, T_IMG>(fVolRight,
																	 fVolLeft,
																	 searchOffset<1>(0,searchDist),
																	 nIter,
																	 nRandomSearch,
																	 std::nullopt,
																	 optCache);


	disp_t* data = out.takePointer(); //take the pointer instead of getting a view, to ensure move semantic can be used to move the resulting array once converted.

	Multidim::Array<disp_t, 2> ret(data, {out.shape()[0], out.shape()[1]}, {out.strides()[0], out.strides()[1]}, true);

	return ret;
}

template<matchingFunctions matchFunc, typename T_IMG, int searchRadius, int searchDist>
Multidim::Array<disp_t, 2> testDenseMatch(Multidim::Array<T_IMG, 3> const& img_left,
										  Multidim::Array<T_IMG, 3> const& img_right,
										  StereoVision::Random::NumbersCache<int>* rngCache) {

	(void) rngCache;

	Multidim::Array<T_IMG, 3> fVolLeft = unfold<T_IMG, T_IMG>(searchRadius, searchRadius, img_left);
	Multidim::Array<T_IMG, 3> fVolRight = unfold<T_IMG, T_IMG>(searchRadius, searchRadius, img_right);

	Multidim::Array<float, 3> CV = unfoldBasedCostVolume<matchFunc>(fVolLeft, fVolRight, searchRadius, searchRadius, searchDist);
	Multidim::Array<disp_t, 2> disp = selectedIndexToDisp<StereoVision::Correlation::disp_t, StereoVision::Correlation::dispDirection::RightToLeft>
			(StereoVision::Correlation::extractSelectedIndex<StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::extractionStrategy>(CV), 0);

	return disp;
}

template<matchingFunctions matchFunc, typename T_IMG, int searchRadius, int searchDist, int depth>
Multidim::Array<disp_t, 2> testHierarchicalMatch(Multidim::Array<T_IMG, 3> const& img_left,
												 Multidim::Array<T_IMG, 3> const& img_right,
												 StereoVision::Random::NumbersCache<int>* rngCache) {

	using TCV = typename std::conditional<std::is_integral_v<T_IMG>, int32_t, float>::type;
	constexpr dispDirection dDir = dispDirection::RightToLeft;

	auto shapel = img_left.shape();
	auto shaper = img_right.shape();

	if (shapel != shaper) {
		return Multidim::Array<disp_t, 2>();
	}

	(void) rngCache;

	StereoVision::Correlation::OffsetedCostVolume<TCV> result =
			StereoVision::Correlation::hiearchicalTruncatedCostVolume<matchFunc, depth, T_IMG, T_IMG, 3, dDir,TCV>
			(img_left,
			 img_right,
			 searchRadius,
			 searchRadius,
			 searchDist);

	return result.disp_estimate;

	return result.disp_estimate;
}

void BenchmarkStereoMatchingModels::benchmarkFloatDispFunc_data() {

	QTest::addColumn<dispFuncF>("disp_function");
	QTest::addColumn<bool>("useRngCache");


	QTest::newRow("testPatchMatch<cost = matchingFunctions::NCC, type = float, searchRadius = 3, searchRange = 120, niter = 5, nRandomSearch = 4> (no rng cache)")
			<< dispFuncF({testPatchMatch<matchingFunctions::NCC,float,3,120,5,4>}) << false;


	QTest::newRow("testPatchMatch<cost = matchingFunctions::NCC, type = float, searchRadius = 3, searchRange = 120, niter = 5, nRandomSearch = 4> (rng cache)")
			<< dispFuncF({testPatchMatch<matchingFunctions::NCC,float,3,120,5,4>}) << true;


	QTest::newRow("testDenseMatch<cost = matchingFunctions::NCC, type = float, searchRadius = 3, searchRange = 120>")
			<< dispFuncF({testDenseMatch<matchingFunctions::NCC,float,3,120>}) << false;


	QTest::newRow("testHierarchicalMatch<cost = matchingFunctions::NCC, type = float, searchRadius = 3, searchRange = 120, depth = 2>")
			<< dispFuncF({testHierarchicalMatch<matchingFunctions::NCC,float,3,120, 2>}) << false;


	QTest::newRow("testHierarchicalMatch<cost = matchingFunctions::NCC, type = float, searchRadius = 3, searchRange = 120, depth = 3>")
			<< dispFuncF({testHierarchicalMatch<matchingFunctions::NCC,float,3,120, 3>}) << false;

}
void BenchmarkStereoMatchingModels::benchmarkFloatDispFunc() {

	QFETCH(dispFuncF, disp_function);
	QFETCH(bool, useRngCache);

	if (_test_imgs.isEmpty()) {
		QSKIP("found no images to process");
	}

	Multidim::Array<disp_t, 2> disp;

	Multidim::Array<float, 3> img_left = StereoVision::IO::readImage<float>(_test_imgs[0].left_nir_path.toStdString());
	Multidim::Array<float, 3> img_right = StereoVision::IO::readImage<float>(_test_imgs[0].right_nir_path.toStdString());

	if (img_left.empty() or img_right.empty()) {
		QSKIP("Could not read images");
	}

	StereoVision::Random::NumbersCache<int>* rngCacheptr = (useRngCache) ? &_random_cache : nullptr;

	QBENCHMARK {
		disp = disp_function.dispFunc(img_left, img_right, rngCacheptr);
	}

	if (disp.empty()) {
		QSKIP("error when computing disparity");
	}
}

void BenchmarkStereoMatchingModels::benchmarkUint8DispFunc_data() {

	QTest::addColumn<dispFuncB>("disp_function");
	QTest::addColumn<bool>("useRngCache");


	QTest::newRow("testPatchMatch<cost = matchingFunctions::NCC, type = uint8_t, searchRadius = 3, searchRange = 120, niter = 5, nRandomSearch = 4> (no rng cache)")
			<< dispFuncB({testPatchMatch<matchingFunctions::NCC,uint8_t,3,120,5,4>}) << false;


	QTest::newRow("testPatchMatch<cost = matchingFunctions::NCC, type = uint8_t, searchRadius = 3, searchRange = 120, niter = 5, nRandomSearch = 4> (rng cache)")
			<< dispFuncB({testPatchMatch<matchingFunctions::NCC,uint8_t,3,120,5,4>}) << true;


	QTest::newRow("testDenseMatch<cost = matchingFunctions::NCC, type = uint8_t, searchRadius = 3, searchRange = 120>")
			<< dispFuncB({testDenseMatch<matchingFunctions::NCC,uint8_t,3,120>}) << false;


	QTest::newRow("testHierarchicalMatch<cost = matchingFunctions::NCC, type = uint8_t, searchRadius = 3, searchRange = 120, depth = 2>")
			<< dispFuncB({testHierarchicalMatch<matchingFunctions::NCC,uint8_t,3,120, 2>}) << false;


	QTest::newRow("testHierarchicalMatch<cost = matchingFunctions::NCC, type = uint8_t, searchRadius = 3, searchRange = 120, depth = 3>")
			<< dispFuncB({testHierarchicalMatch<matchingFunctions::NCC,uint8_t,3,120, 3>}) << false;

}
void BenchmarkStereoMatchingModels::benchmarkUint8DispFunc() {

	QFETCH(dispFuncB, disp_function);
	QFETCH(bool, useRngCache);

	if (_test_imgs.isEmpty()) {
		QSKIP("found no images to process");
	}

	Multidim::Array<disp_t, 2> disp;

	Multidim::Array<uint8_t, 3> img_left = StereoVision::IO::readImage<uint8_t>(_test_imgs[0].left_nir_path.toStdString());
	Multidim::Array<uint8_t, 3> img_right = StereoVision::IO::readImage<uint8_t>(_test_imgs[0].right_nir_path.toStdString());

	if (img_left.empty() or img_right.empty()) {
		QSKIP("Could not read images");
	}

	StereoVision::Random::NumbersCache<int>* rngCacheptr = (useRngCache) ? &_random_cache : nullptr;

	QBENCHMARK {
		disp = disp_function.dispFunc(img_left, img_right, rngCacheptr);
	}
}

QTEST_MAIN(BenchmarkStereoMatchingModels)
#include "benchmarkStereoMatchingModels.moc"
