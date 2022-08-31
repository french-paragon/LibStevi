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

#include "io/image_io.h"

#include <MultidimArrays/MultidimArrays.h>

class BenchmarkImageSavingAlgorithms: public QObject
{

	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void benchmarkImageSaving_data();
	void benchmarkImageSaving();

private:
	std::default_random_engine re;

	template<typename T>
	void benchmarkImageSavingImpl(int w, int h, int channels, QString ext) {

		T maxVal = 0xFF;
		if (std::is_same_v<T, uint16_t>) {
			maxVal = static_cast<T>(0xFFFF);
		}
		if (std::is_same_v<T, uint32_t>) {
			maxVal = static_cast<T>(0xFFFFFFFF);
		}

		std::uniform_int_distribution<std::conditional_t<std::is_integral_v<T>, T, uint32_t>> idist(0,maxVal);
		std::uniform_real_distribution<std::conditional_t<std::is_integral_v<T>, float, T>> fdist(0,maxVal);

		Multidim::Array<T,3> img(h,w,channels);

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				for (int c = 0; c < channels; c++) {
					T val = (std::is_integral_v<T>) ? idist(re) : fdist(re);
					img.atUnchecked(i,j,c) = val;
				}
			}
		}

		QString formatInfo = QString("%1%2").arg(((std::is_integral_v<T>) ? "i" : "f")).arg(sizeof (T)*8);
		QString filename = QString("%1_%2_%3_%4.%5").arg(w).arg(h).arg(channels).arg(formatInfo).arg(ext);
		QString filepath = QDir::current().filePath(filename);
		std::string fname = filepath.toStdString();

		QFile f(filepath);
		if (f.exists()) {
			f.remove(); //remove the file, in case it already exist.
		}

		volatile bool ok;

		QBENCHMARK_ONCE {
			ok = ok and StereoVision::IO::writeImage<T,T>(fname,img);
		}

	}

};

void BenchmarkImageSavingAlgorithms::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void BenchmarkImageSavingAlgorithms::benchmarkImageSaving_data() {

	QTest::addColumn<int>("w");
	QTest::addColumn<int>("h");
	QTest::addColumn<int>("channels");
	QTest::addColumn<int>("bitdepth");
	QTest::addColumn<bool>("floating_point");
	QTest::addColumn<QString>("extension");

	QTest::newRow("SD 8bit stevimg grayscale") << 640 << 480 << 1 << 8 << false << "stevimg";
	QTest::newRow("SD 8bit stevimg rgb") << 640 << 480 << 3 << 8 << false << "stevimg";
	QTest::newRow("RealSenseMaxRes 8bit stevimg grayscale") << 1200 << 800 << 1 << 8 << false << "stevimg";
	QTest::newRow("RealSenseMaxRes 8bit stevimg rgb") << 1200 << 800 << 3 << 8 << false << "stevimg";

	QTest::newRow("SD 16bit stevimg grayscale") << 640 << 480 << 1 << 16 << false << "stevimg";
	QTest::newRow("SD 16bit stevimg rgb") << 640 << 480 << 3 << 16 << false << "stevimg";
	QTest::newRow("RealSenseMaxRes 16bit stevimg grayscale") << 1200 << 800 << 1 << 16 << false << "stevimg";
	QTest::newRow("RealSenseMaxRes 16bit stevimg rgb") << 1200 << 800 << 3 << 16 << false << "stevimg";

	QTest::newRow("SD 8bit cimg grayscale") << 640 << 480 << 1 << 8 << false << "cimg";
	QTest::newRow("SD 8bit cimg rgb") << 640 << 480 << 3 << 8 << false << "cimg";
	QTest::newRow("RealSenseMaxRes 8bit cimg grayscale") << 1200 << 800 << 1 << 8 << false << "cimg";
	QTest::newRow("RealSenseMaxRes 8bit cimg rgb") << 1200 << 800 << 3 << 8 << false << "cimg";

	QTest::newRow("SD 16bit cimg grayscale") << 640 << 480 << 1 << 16 << false << "cimg";
	QTest::newRow("SD 16bit cimg rgb") << 640 << 480 << 3 << 16 << false << "cimg";
	QTest::newRow("RealSenseMaxRes 16bit cimg grayscale") << 1200 << 800 << 1 << 16 << false << "cimg";
	QTest::newRow("RealSenseMaxRes 16bit cimg rgb") << 1200 << 800 << 3 << 16 << false << "cimg";

	QTest::newRow("SD 8bit bmp grayscale") << 640 << 480 << 1 << 8 << false << "bmp";
	QTest::newRow("SD 8bit bmp rgb") << 640 << 480 << 3 << 8 << false << "bmp";
	QTest::newRow("RealSenseMaxRes 8bit bmp grayscale") << 1200 << 800 << 1 << 8 << false << "bmp";
	QTest::newRow("RealSenseMaxRes 8bit bmp rgb") << 1200 << 800 << 3 << 8 << false << "bmp";

#ifdef JPEGAVAILABLE

	QTest::newRow("SD 8bit jpg grayscale") << 640 << 480 << 1 << 8 << false << "jpg";
	QTest::newRow("SD 8bit jpg rgb") << 640 << 480 << 3 << 8 << false << "jpg";
	QTest::newRow("RealSenseMaxRes 8bit jpg grayscale") << 1200 << 800 << 1 << 8 << false << "jpg";
	QTest::newRow("RealSenseMaxRes 8bit jpg rgb") << 1200 << 800 << 3 << 8 << false << "jpg";

#endif

#ifdef PNGAVAILABLE

	QTest::newRow("SD 8bit png grayscale") << 640 << 480 << 1 << 8 << false << "png";
	QTest::newRow("SD 8bit png rgb") << 640 << 480 << 3 << 8 << false << "png";
	QTest::newRow("RealSenseMaxRes 8bit png grayscale") << 1200 << 800 << 1 << 8 << false << "png";
	QTest::newRow("RealSenseMaxRes 8bit png rgb") << 1200 << 800 << 3 << 8 << false << "png";

	QTest::newRow("SD 16bit png grayscale") << 640 << 480 << 1 << 16 << false << "png";
	QTest::newRow("SD 16bit png rgb") << 640 << 480 << 3 << 16 << false << "png";
	QTest::newRow("RealSenseMaxRes 16bit png grayscale") << 1200 << 800 << 1 << 16 << false << "png";
	QTest::newRow("RealSenseMaxRes 16bit png rgb") << 1200 << 800 << 3 << 16 << false << "png";

#endif

#ifdef TIFFAVAILABLE

	QTest::newRow("SD 8bit tiff grayscale") << 640 << 480 << 1 << 8 << false << "tiff";
	QTest::newRow("SD 8bit tiff rgb") << 640 << 480 << 3 << 8 << false << "tiff";
	QTest::newRow("RealSenseMaxRes 8bit tiff grayscale") << 1200 << 800 << 1 << 8 << false << "tiff";
	QTest::newRow("RealSenseMaxRes 8bit tiff rgb") << 1200 << 800 << 3 << 8 << false << "tiff";

	QTest::newRow("SD 16bit tiff grayscale") << 640 << 480 << 1 << 16 << false << "tiff";
	QTest::newRow("SD 16bit tiff rgb") << 640 << 480 << 3 << 16 << false << "tiff";
	QTest::newRow("RealSenseMaxRes 16bit tiff grayscale") << 1200 << 800 << 1 << 16 << false << "tiff";
	QTest::newRow("RealSenseMaxRes 16bit tiff rgb") << 1200 << 800 << 3 << 16 << false << "tiff";

#endif

}
void BenchmarkImageSavingAlgorithms::benchmarkImageSaving() {

	QFETCH(int, w);
	QFETCH(int, h);
	QFETCH(int, channels);
	QFETCH(int, bitdepth);
	QFETCH(bool, floating_point);
	QFETCH(QString, extension);

	if (floating_point == false) {
		if (bitdepth != 8 and bitdepth != 16 and bitdepth != 32) {
			QSKIP("Invalid bit depth provided for integer precision image");
		}

		if (bitdepth == 8) {
			benchmarkImageSavingImpl<uint8_t>(w, h, channels, extension);
		}

		if (bitdepth == 16) {
			benchmarkImageSavingImpl<uint16_t>(w, h, channels, extension);
		}

		if (bitdepth == 32) {
			benchmarkImageSavingImpl<uint32_t>(w, h, channels, extension);
		}

	} else {

		if (bitdepth != 32) {
			QSKIP("Invalid bit depth provided for floating point precision image");
		}

		if (bitdepth == 32) {
			benchmarkImageSavingImpl<float>(w, h, channels, extension);
		}
	}

}

QTEST_MAIN(BenchmarkImageSavingAlgorithms)
#include "benchmarkImageSaving.moc"
