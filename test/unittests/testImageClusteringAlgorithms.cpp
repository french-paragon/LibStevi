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

#include "imageProcessing/meanShiftClustering.h"

#include <iostream>
#include <random>

using namespace StereoVision::ImageProcessing;

Multidim::Array<float, 3> getPiecewiseConstantImage(int w,
													int h,
													std::default_random_engine & re,
													float dist) {

	std::uniform_real_distribution<float> valuesDist(0., 1.);

	float delta = std::sqrt(dist*dist/3.);

	std::array<float, 3> color1 = {valuesDist(re), valuesDist(re), valuesDist(re)};
	std::array<float, 3> color2 = {color1[0]+delta, color1[1]+delta, color1[2]+delta};

	Multidim::Array<float, 3> img(w, h, 3);

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			for (int c = 0; c < 3; c++) {
				img.atUnchecked(i,j,c) = (i < j) ? color1[c] : color2[c];
			}
		}
	}

	return img;
}

class TestImageClusteringAlgorithms: public QObject
{

	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testMeanShiftClustering_data();
	void testMeanShiftClustering();

private:
	std::default_random_engine re;

};


void TestImageClusteringAlgorithms::initTestCase() {
	//srand((unsigned int) time(nullptr));
	std::random_device rd;
	re.seed(rd());
}

void TestImageClusteringAlgorithms::testMeanShiftClustering_data() {

	QTest::addColumn<int>("w");
	QTest::addColumn<int>("h");
	QTest::addColumn<float>("radius");

	QTest::newRow("minuscule") << 10 << 10 << 0.5f;
	QTest::newRow("small") << 50 << 50 << 0.5f;
	QTest::newRow("average") << 100 << 100 << 0.5f;
}

void TestImageClusteringAlgorithms::testMeanShiftClustering() {

	QFETCH(int, w);
	QFETCH(int, h);
	QFETCH(float, radius);

	auto img = getPiecewiseConstantImage(w,h,re, 1.5*radius);

	std::function<float(std::vector<float> const& v1,std::vector<float> const& v2)> kernel = RadiusKernel<float>(radius);
	auto approx = meanShiftClustering(img, kernel, -1, std::optional<float>(1e-6));

	auto imgShape = img.shape();
	auto approxShape = approx.shape();

	for (size_t i = 0; i < imgShape.size(); i++) {
		QCOMPARE(imgShape[i], approxShape[i]);
	}

	std::array<float, 3> avg_color = {0,0,0};

	for (int i = 0; i < w; i++) {

		for (int j = 0; j < w; j++) {

			for (int c = 0; c < 3; c++) {
				avg_color[c] += img.valueUnchecked(i,j,c);
			}

		}

	}

	for (int c = 0; c < 3; c++) {
		avg_color[c] /= w*h;
	}

	for (int i = 0; i < w; i++) {

		for (int j = 0; j < w; j++) {

			for (int c = 0; c < 3; c++) {
				float imgVal = img.valueUnchecked(i,j,c);
				float approxVal = approx.valueUnchecked(i,j,c);

				float errorVal = imgVal - approxVal;
				float errorAvg = avg_color[c] - approxVal;

				QVERIFY2(std::fabs(errorVal) < 1e-4 or std::fabs(errorAvg) < 1e-4,
						 qPrintable(QString("The approximated color is not either the original image value or the average value (error val = %1, error avg = %2).")
									.arg(errorVal).arg(errorAvg)));
			}

		}

	}
}

QTEST_MAIN(TestImageClusteringAlgorithms)
#include "testImageClusteringAlgorithms.moc"
