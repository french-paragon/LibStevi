/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2023  Paragon<french.paragon@gmail.com>

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

#include <QtTest/QtTest>

#include <MultidimArrays/MultidimArrays.h>
#include "imageProcessing/foregroundSegmentation.h"

#include <random>

class TestSegmentation: public QObject
{
	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testGlobalSegmentation_data();
	void testGlobalSegmentation();

private:
	std::default_random_engine re;

};

void TestSegmentation::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestSegmentation::testGlobalSegmentation_data() {

	QTest::addColumn<int>("height");
	QTest::addColumn<int>("width");
	QTest::addColumn<int>("centerI");
	QTest::addColumn<int>("centerJ");
	QTest::addColumn<int>("Radius");

	QTest::newRow("Small images") << 120 << 260 << 80 << 180 << 10;

}
void TestSegmentation::testGlobalSegmentation() {

	constexpr auto Foreground = StereoVision::ImageProcessing::FgBgSegmentation::Foreground;
	constexpr auto Background = StereoVision::ImageProcessing::FgBgSegmentation::Background;

	QFETCH(int, height);
	QFETCH(int, width);
	QFETCH(int, centerI);
	QFETCH(int, centerJ);
	QFETCH(int, Radius);

	Multidim::Array<uint8_t,3> img(height, width, 3);
	Multidim::Array<bool,2> gt_mask(height, width);

	std::array<uint8_t, 3> colorFg = {255, 128, 0};
	std::array<uint8_t, 3> colorBg = {120, 50, 27};

	int cost_bg = 69;
	int cost_fg = 42;

	float prob_fg_fg = 95;
	float prob_fg_bg = 05;

	std::bernoulli_distribution dist_fg_fg(prob_fg_fg);
	std::bernoulli_distribution dist_fg_bg(prob_fg_bg);

	Multidim::Array<int,3> segmentation_cost(height, width, 2);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			bool isFg = (i - centerI)*(i - centerI) + (j - centerJ)*(j - centerJ) < Radius*Radius;

			gt_mask.atUnchecked(i,j) = (isFg) ? Foreground : Background;

			for (int c = 0; c < 3; c++) {
				img.atUnchecked(i,j,c) = (isFg) ? colorFg[c] : colorBg[c];
			}

			bool fgIsFg = dist_fg_fg(re);
			bool bgIsFg = dist_fg_bg(re);

			segmentation_cost.atUnchecked(i,j,Foreground) = (fgIsFg) ? cost_fg : cost_bg;
			segmentation_cost.atUnchecked(i,j,Background) = (bgIsFg) ? cost_fg : cost_bg;

		}
	}

	StereoVision::ImageProcessing::GuidedMaskCostPolicy maskCostPolicy(cost_bg, img);

	Multidim::Array<StereoVision::ImageProcessing::FgBgSegmentation::MaskInfo, 2> mask =
			StereoVision::ImageProcessing::getGlobalRefinedMask(segmentation_cost, maskCostPolicy);

	int nPixels = 0;
	int nCorrectPixels = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			if (gt_mask.valueUnchecked(i,j) == mask.valueUnchecked(i,j)) {
				nCorrectPixels++;
			}

			nPixels++;

		}
	}

	float propCorrect = float(nCorrectPixels)/float(nPixels);

	QVERIFY2(propCorrect >= 0.98, "Proportion of correct pixels is not large enough");

}

QTEST_MAIN(TestSegmentation)
#include "testSegmentation.moc"
