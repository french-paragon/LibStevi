#include <QTest>

#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QMap>

#include <MultidimArrays/MultidimArrays.h>

#include "correlation/cross_correlations.h"
#include "correlation/hierarchical.h"
#include "correlation/patchmatch.h"
#include "io/image_io.h"

using namespace StereoVision::Correlation;

typedef Multidim::Array<disp_t, 2> (*dispFuncFloat)(Multidim::Array<float, 3> const&, Multidim::Array<float, 3> const&);
typedef Multidim::Array<disp_t, 2> (*dispFuncByte)(Multidim::Array<uint8_t, 3> const&, Multidim::Array<uint8_t, 3> const&);

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
}

template<matchingFunctions matchFunc, typename T_IMG, int searchRadius, int searchDist, int nIter, int nRandomSearch>
Multidim::Array<disp_t, 2> testPatchMatch(Multidim::Array<T_IMG, 3> const& img_left, Multidim::Array<T_IMG, 3> const& img_right) {

	Multidim::Array<T_IMG, 3> fVolLeft = unfold<T_IMG, T_IMG>(searchRadius, searchRadius, img_left);
	Multidim::Array<T_IMG, 3> fVolRight = unfold<T_IMG, T_IMG>(searchRadius, searchRadius, img_right);

	Multidim::Array<disp_t, 3> out = patchMatch<matchFunc, 1, T_IMG>(fVolRight,
																	 fVolLeft,
																	 searchOffset<1>(0,searchDist),
																	 nIter,
																	 nRandomSearch);


	disp_t* data = out.takePointer(); //take the pointer instead of getting a view, to ensure move semantic can be used to move the resulting array once converted.

	Multidim::Array<disp_t, 2> ret(data, {out.shape()[0], out.shape()[1]}, {out.strides()[0], out.strides()[1]}, true);

	return ret;
}

void BenchmarkStereoMatchingModels::benchmarkFloatDispFunc_data() {

	QTest::addColumn<dispFuncF>("disp_function");


	QTest::newRow("testPatchMatch<cost = matchingFunctions::NCC, type = float, searchRadius = 3, searchRange = 120, niter = 5, nRandomSearch = 4>")
			<< dispFuncF({testPatchMatch<matchingFunctions::NCC,float,3,120,5,4>});
}
void BenchmarkStereoMatchingModels::benchmarkFloatDispFunc() {

	QFETCH(dispFuncF, disp_function);

	if (_test_imgs.isEmpty()) {
		QSKIP("found no images to process");
	}

	Multidim::Array<disp_t, 2> disp;

	Multidim::Array<float, 3> img_left = StereoVision::IO::readImage<float>(_test_imgs[0].left_nir_path.toStdString());
	Multidim::Array<float, 3> img_right = StereoVision::IO::readImage<float>(_test_imgs[0].right_nir_path.toStdString());

	if (img_left.empty() or img_right.empty()) {
		QSKIP("Could not read images");
	}

	QBENCHMARK {
		disp = disp_function.dispFunc(img_left, img_right);
	}
}

QTEST_MAIN(BenchmarkStereoMatchingModels)
#include "benchmarkStereoMatchingModels.moc"
