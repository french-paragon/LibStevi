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

#include "imageProcessing/finiteDifferences.h"

#include <MultidimArrays/MultidimArrays.h>

#include <random>

class TestFiniteDifferences: public QObject
{

	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testFiniteDifferences();

private:
	std::default_random_engine re;

};

void TestFiniteDifferences::initTestCase() {

	std::random_device rd;
	re.seed(rd());

}

void TestFiniteDifferences::testFiniteDifferences() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	constexpr int width = 5;
	constexpr int height = 5;
	constexpr int channels = 3;

	Multidim::Array<uint8_t,2> array(height, width);
	Multidim::Array<uint8_t,3> img(height, width, channels);

	std::uniform_int_distribution<uint8_t> idist(0,255);

	for (int it = 0; it < 50; it++) {

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				array.at<Nc>(i,j) = idist(re);

				for (int c = 0; c < channels; c++) {
					img.at<Nc>(i,j,c) = idist(re);
				}

			}
		}

		Multidim::Array<int16_t,2> out2DaxX = StereoVision::finiteDifference<uint8_t, StereoVision::Geometry::ImageAxis::X, int16_t>(array);
		Multidim::Array<int16_t,2> out2DaxY = StereoVision::finiteDifference<uint8_t, StereoVision::Geometry::ImageAxis::Y, int16_t>(array);

		Multidim::Array<int16_t,3> out3DaxX = StereoVision::finiteDifference<uint8_t, StereoVision::Geometry::ImageAxis::X, int16_t>(img);
		Multidim::Array<int16_t,3> out3DaxY = StereoVision::finiteDifference<uint8_t, StereoVision::Geometry::ImageAxis::Y, int16_t>(img);


		QCOMPARE(array.shape(), out2DaxX.shape());
		QCOMPARE(array.shape(), out2DaxY.shape());

		QCOMPARE(img.shape(), out3DaxX.shape());
		QCOMPARE(img.shape(), out3DaxY.shape());

		int16_t array_diff_x = static_cast<int16_t>(array.value<Nc>(1,3)) + 2*static_cast<int16_t>(array.value<Nc>(2,3)) + static_cast<int16_t>(array.value<Nc>(3,3))
				- static_cast<int16_t>(array.value<Nc>(1,1)) - 2*static_cast<int16_t>(array.value<Nc>(2,1)) - static_cast<int16_t>(array.value<Nc>(3,1));
		int16_t array_diff_y = static_cast<int16_t>(array.value<Nc>(3,1)) + 2*static_cast<int16_t>(array.value<Nc>(3,2)) + static_cast<int16_t>(array.value<Nc>(3,3))
				- static_cast<int16_t>(array.value<Nc>(1,1)) - 2*static_cast<int16_t>(array.value<Nc>(1,2)) - static_cast<int16_t>(array.value<Nc>(1,3));

		QCOMPARE(out2DaxX.value<Nc>(2,2), array_diff_x);
		QCOMPARE(out2DaxY.value<Nc>(2,2), array_diff_y);


		for (int c = 0; c < channels; c++) {

			int16_t img_diff_x = static_cast<int16_t>(img.value<Nc>(1,3,c)) + 2*static_cast<int16_t>(img.value<Nc>(2,3,c)) + static_cast<int16_t>(img.value<Nc>(3,3,c))
					- static_cast<int16_t>(img.value<Nc>(1,1,c)) - 2*static_cast<int16_t>(img.value<Nc>(2,1,c)) - static_cast<int16_t>(img.value<Nc>(3,1,c));
			int16_t img_diff_y = static_cast<int16_t>(img.value<Nc>(3,1,c)) + 2*static_cast<int16_t>(img.value<Nc>(3,2,c)) + static_cast<int16_t>(img.value<Nc>(3,3,c))
					- static_cast<int16_t>(img.value<Nc>(1,1,c)) - 2*static_cast<int16_t>(img.value<Nc>(1,2,c)) - static_cast<int16_t>(img.value<Nc>(1,3,c));

			QCOMPARE(out3DaxX.value<Nc>(2,2,c), img_diff_x);
			QCOMPARE(out3DaxY.value<Nc>(2,2,c), img_diff_y);
		}
	}

}


QTEST_MAIN(TestFiniteDifferences)
#include "testFiniteDifferences.moc"
