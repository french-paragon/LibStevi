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

#include <random>

class TestImageIO: public QObject
{

	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testSetvimageSaving_data();
	void testSetvimageSaving();

    void testMultispectralTiffSaving();

private:
	std::default_random_engine re;

	template<typename T>
	void testSetvimageSavingImpl(int w, int h, int channels) {

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
		QString filename = QString("%1_%2_%3_%4.%5").arg(w).arg(h).arg(channels).arg(formatInfo).arg("stevimg");
		QString filepath = QDir::current().filePath(filename);
		std::string fname = filepath.toStdString();

		QFile f(filepath);
		if (f.exists()) {
			f.remove(); //remove the file, in case it already exist.
		}

		bool ok = StereoVision::IO::writeImage<T,T>(fname,img);

		QVERIFY2(ok, "Failed to write image to disk");

		Multidim::Array<T, 3> reloaded = StereoVision::IO::readImage<T>(fname);

		QCOMPARE(img.shape(), reloaded.shape());

		for (int i = 0; i < img.shape()[0]; i++) {
			for (int j = 0; j < img.shape()[1]; j++) {
				for (int c = 0; c < img.shape()[2]; c++) {
					QCOMPARE(img.atUnchecked(i,j,c), reloaded.atUnchecked(i,j,c));
				}
			}
		}

	}

};

void TestImageIO::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestImageIO::testSetvimageSaving_data() {

	QTest::addColumn<int>("w");
	QTest::addColumn<int>("h");
	QTest::addColumn<int>("channels");
	QTest::addColumn<int>("bitdepth");
	QTest::addColumn<bool>("floating_point");

	QTest::newRow("SD 8bit stevimg grayscale") << 640 << 480 << 1 << 8 << false;
	QTest::newRow("SD 8bit stevimg rgb") << 640 << 480 << 3 << 8 << false;
	QTest::newRow("RealSenseMaxRes 8bit stevimg grayscale") << 1200 << 800 << 1 << 8 << false;
	QTest::newRow("RealSenseMaxRes 8bit stevimg rgb") << 1200 << 800 << 3 << 8 << false;

	QTest::newRow("SD 16bit stevimg grayscale") << 640 << 480 << 1 << 16 << false;
	QTest::newRow("SD 16bit stevimg rgb") << 640 << 480 << 3 << 16 << false;
	QTest::newRow("RealSenseMaxRes 16bit stevimg grayscale") << 1200 << 800 << 1 << 16 << false;
	QTest::newRow("RealSenseMaxRes 16bit stevimg rgb") << 1200 << 800 << 3 << 16 << false;

}
void TestImageIO::testSetvimageSaving() {

	QFETCH(int, w);
	QFETCH(int, h);
	QFETCH(int, channels);
	QFETCH(int, bitdepth);
	QFETCH(bool, floating_point);

	if (floating_point == false) {
		if (bitdepth != 8 and bitdepth != 16 and bitdepth != 32) {
			QSKIP("Invalid bit depth provided for integer precision image");
		}

		if (bitdepth == 8) {
			testSetvimageSavingImpl<uint8_t>(w, h, channels);
		}

		if (bitdepth == 16) {
			testSetvimageSavingImpl<uint16_t>(w, h, channels);
		}

		if (bitdepth == 32) {
			testSetvimageSavingImpl<uint32_t>(w, h, channels);
		}

	} else {

		if (bitdepth != 32) {
			QSKIP("Invalid bit depth provided for floating point precision image");
		}

		if (bitdepth == 32) {
			testSetvimageSavingImpl<float>(w, h, channels);
		}
	}

}



void TestImageIO::testMultispectralTiffSaving() {

#ifndef STEVI_IO_USE_TIFF
    QSKIP("Not using native tiff library");
#endif

    QString filename = "hyperspectral_tiff_test.tiff";

    QString filepath = QDir::current().filePath(filename);
    std::string fname = filepath.toStdString();

    QFile f(filepath);
    if (f.exists()) {
        f.remove(); //remove the file, in case it already exist.
    }

    Multidim::Array<float, 3> pseudotiff(6,33,42);

    bool ok = StereoVision::IO::writeImage<float>(fname, pseudotiff);

    QVERIFY2(ok, "could not save tiff with more than 3 channels");
}

QTEST_MAIN(TestImageIO)
#include "testImageIO.moc"
