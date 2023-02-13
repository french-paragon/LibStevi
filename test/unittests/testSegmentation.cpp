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
#include "imageProcessing/morphologicalOperators.h"

#include <random>

struct SegmentationProblem {

	Multidim::Array<uint8_t,3> img;
	Multidim::Array<bool,2> gt_mask;
	Multidim::Array<int,3> segmentation_cost;

};

SegmentationProblem buildCirleProblem(int height,
									  int width,
									  int centerI,
									  int centerJ,
									  int Radius,
									  std::default_random_engine & re,
									  std::array<uint8_t, 3> colorFg = {255, 128, 0},
									  std::array<uint8_t, 3> colorBg = {120, 50, 27},
									  int cost_bg = 69,
									  int cost_fg = 42) {

	constexpr auto Foreground = StereoVision::ImageProcessing::FgBgSegmentation::Foreground;
	constexpr auto Background = StereoVision::ImageProcessing::FgBgSegmentation::Background;

	Multidim::Array<uint8_t,3> img(height, width, 3);
	Multidim::Array<bool,2> gt_mask(height, width);

	float prob_fg_fg = 0.95;
	float prob_fg_bg = 0.05;

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

			bool roughIsFg = dist_fg_fg(re);

			if (!isFg) {
				roughIsFg = dist_fg_bg(re);
			}

			segmentation_cost.atUnchecked(i,j,Foreground) = (roughIsFg) ? cost_fg : cost_bg;
			segmentation_cost.atUnchecked(i,j,Background) = (roughIsFg) ? cost_bg : cost_fg;

		}
	}

	return {img, gt_mask, segmentation_cost};
}

class TestSegmentation: public QObject
{
	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testGlobalSegmentation_data();
	void testGlobalSegmentation();

	void testPartialGlobalSegmentation_data();
	void testPartialGlobalSegmentation();

	void testHierarchicalGlobalSegmentation_data();
	void testHierarchicalGlobalSegmentation();

	void testOtsuThresholding();

private:
	std::default_random_engine re;

	template<int depth>
	void testHierarchicalGlobalSegmentation_impl(int height,
												 int width,
												 int centerI,
												 int centerJ,
												 int Radius) {

		std::array<uint8_t, 3> colorFg = {255, 128, 0};
		std::array<uint8_t, 3> colorBg = {120, 50, 27};

		int cost_bg = 69;
		int cost_fg = 42;

		std::array<SegmentationProblem, depth> problems;
		std::array<StereoVision::ImageProcessing::GuidedMaskCostPolicy<int, uint8_t>*, depth> policies;

		for (int i = 0; i < depth; i++) {
			int actualDepth = depth-i-1;

			int scale = 1;
			for (int d = 0; d < i; d++) {
				scale *= 2;
			}

			problems[actualDepth] = buildCirleProblem(scale*height,
													  scale*width,
													  scale*centerI,
													  scale*centerJ,
													  scale*Radius,
													  re,
													  colorFg,
													  colorBg,
													  cost_bg,
													  cost_fg);

			policies[actualDepth] = new StereoVision::ImageProcessing::GuidedMaskCostPolicy(cost_bg, problems[actualDepth].img);

		}

		std::array<Multidim::Array<int, 3> const*, depth> costs;
		std::array<StereoVision::ImageProcessing::MaskCostPolicy<int> const*, depth> cost_policies;

		for (int d = 0; d < depth; d++) {
			costs[d] = &problems[d].segmentation_cost;
			cost_policies[d] = policies[d];
		}


		Multidim::Array<StereoVision::ImageProcessing::FgBgSegmentation::MaskInfo, 2> mask =
				StereoVision::ImageProcessing::hierarchicalGlobalRefinedMask(costs, cost_policies);

		for (int d = 0; d < depth; d++) {
			delete policies[d];
		}

		int nPixels = 0;
		int nCorrectPixels = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				if (problems[0].gt_mask.valueUnchecked(i,j) == mask.valueUnchecked(i,j)) {
					nCorrectPixels++;
				}

				nPixels++;

			}
		}

		float propCorrect = float(nCorrectPixels)/float(nPixels);

		QVERIFY2(propCorrect >= 0.98, "Proportion of correct pixels is not large enough");

	}

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

	QTest::newRow("Small images") << 120 << 260 << 60 << 180 << 50;

}
void TestSegmentation::testGlobalSegmentation() {

	QFETCH(int, height);
	QFETCH(int, width);
	QFETCH(int, centerI);
	QFETCH(int, centerJ);
	QFETCH(int, Radius);

	std::array<uint8_t, 3> colorFg = {255, 128, 0};
	std::array<uint8_t, 3> colorBg = {120, 50, 27};

	int cost_bg = 69;
	int cost_fg = 42;

	SegmentationProblem problem = buildCirleProblem(height, width, centerI, centerJ, Radius, re, colorFg, colorBg, cost_bg, cost_fg);

	StereoVision::ImageProcessing::GuidedMaskCostPolicy maskCostPolicy(cost_bg, problem.img);

	Multidim::Array<StereoVision::ImageProcessing::FgBgSegmentation::MaskInfo, 2> mask =
			StereoVision::ImageProcessing::getGlobalRefinedMask(problem.segmentation_cost, maskCostPolicy);

	int nPixels = 0;
	int nCorrectPixels = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			if (problem.gt_mask.valueUnchecked(i,j) == mask.valueUnchecked(i,j)) {
				nCorrectPixels++;
			}

			nPixels++;

		}
	}

	float propCorrect = float(nCorrectPixels)/float(nPixels);

	QVERIFY2(propCorrect >= 0.98, "Proportion of correct pixels is not large enough");

}



void TestSegmentation::testPartialGlobalSegmentation_data() {
	QTest::addColumn<int>("height");
	QTest::addColumn<int>("width");
	QTest::addColumn<int>("centerI");
	QTest::addColumn<int>("centerJ");
	QTest::addColumn<int>("Radius");
	QTest::addColumn<int>("searchRadius");

	QTest::newRow("Small images") << 120 << 260 << 60 << 180 << 50 << 10;
}

void TestSegmentation::testPartialGlobalSegmentation() {

	using MaskInfo = StereoVision::ImageProcessing::FgBgSegmentation::MaskInfo;

	QFETCH(int, height);
	QFETCH(int, width);
	QFETCH(int, centerI);
	QFETCH(int, centerJ);
	QFETCH(int, Radius);
	QFETCH(int, searchRadius);

	std::array<uint8_t, 3> colorFg = {255, 128, 0};
	std::array<uint8_t, 3> colorBg = {120, 50, 27};

	int cost_bg = 69;
	int cost_fg = 42;

	SegmentationProblem problem = buildCirleProblem(height, width, centerI, centerJ, Radius, re, colorFg, colorBg, cost_bg, cost_fg);

	Multidim::Array<MaskInfo, 2> extended_mask = StereoVision::ImageProcessing::dilation<bool, MaskInfo>(searchRadius, searchRadius, problem.gt_mask);
	Multidim::Array<MaskInfo, 2> eroded_mask = StereoVision::ImageProcessing::erosion<bool, MaskInfo>(searchRadius, searchRadius, problem.gt_mask);

	Multidim::Array<bool, 2> optimizablePixels(problem.gt_mask.shape());

	for (int i = 0; i < problem.gt_mask.shape()[0]; i++) {
		for (int j = 0; j < problem.gt_mask.shape()[1]; j++) {

			optimizablePixels.atUnchecked(i,j) = (extended_mask.valueUnchecked(i,j) == MaskInfo::Foreground and
												  eroded_mask.valueUnchecked(i,j) == MaskInfo::Background);

		}
	}

	StereoVision::ImageProcessing::GuidedMaskCostPolicy maskCostPolicy(4*cost_bg, problem.img);

	Multidim::Array<MaskInfo, 2> mask =
			StereoVision::ImageProcessing::getPartialGlobalRefinedMask(problem.segmentation_cost, maskCostPolicy, optimizablePixels, eroded_mask);

	int nPixels = 0;
	int nCorrectPixels = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			if (!optimizablePixels.valueUnchecked(i,j)) {
				continue;
			}

			if (problem.gt_mask.valueUnchecked(i,j) == mask.valueUnchecked(i,j)) {
				nCorrectPixels++;
			}

			nPixels++;

		}
	}

	float propCorrect = float(nCorrectPixels)/float(nPixels);

	QVERIFY2(propCorrect >= 0.95, "Proportion of correct pixels is not large enough");

}

void TestSegmentation::testHierarchicalGlobalSegmentation_data() {

	QTest::addColumn<int>("height");
	QTest::addColumn<int>("width");
	QTest::addColumn<int>("centerI");
	QTest::addColumn<int>("centerJ");
	QTest::addColumn<int>("Radius");
	QTest::addColumn<int>("depth");

	QTest::newRow("Up to SD") << 120 << 160 << 80 << 180 << 10 << 3;

}
void TestSegmentation::testHierarchicalGlobalSegmentation() {

	QFETCH(int, height);
	QFETCH(int, width);
	QFETCH(int, centerI);
	QFETCH(int, centerJ);
	QFETCH(int, Radius);
	QFETCH(int, depth);

	if (depth != 2 and depth != 3 and depth != 4) {
		QSKIP("Skipping test (unsupported depth");
	}

	switch (depth) {
	case 2:
		testHierarchicalGlobalSegmentation_impl<2>(height,
												   width,
												   centerI,
												   centerJ,
												   Radius);
		break;
	case 3:
		testHierarchicalGlobalSegmentation_impl<3>(height,
												   width,
												   centerI,
												   centerJ,
												   Radius);
		break;
	case 4:
		testHierarchicalGlobalSegmentation_impl<4>(height,
												   width,
												   centerI,
												   centerJ,
												   Radius);
		break;
	}

}



void TestSegmentation::testOtsuThresholding() {

	Multidim::Array<int,2> img(3,6);

	Multidim::IndexConverter<2> converter(img.shape());

	std::vector<std::tuple<int,int>> bins;

	int nlevels = 11;
	int midPoint = nlevels/2;

	if (nlevels % 2 == 0) {
		QSKIP("Invalid number of levels, number must be odd");
	}

	if (midPoint % 2 == 0) {
		QSKIP("Invalid midpoint of levels, number must be odd");
	}

	bins.reserve(nlevels);

	for (int i = 0; i < nlevels; i++) {
		if (i == midPoint) {
			bins.emplace_back(i,0);
		}

		int modulo = (i > midPoint) ? i - (midPoint+1) : i;
		int subMidPoint = midPoint / 2;

		if (modulo < subMidPoint) {
			bins.emplace_back(i,modulo+1);
		} else {
			bins.emplace_back(i,midPoint-modulo);
		}
	}

	int idx = 0;
	for (std::tuple<int,int> const& bin : bins) {
		for (int i = 0; i < std::get<1>(bin); i++) {

			std::array<int,2> pos = converter.getIndexFromPseudoFlatId(idx);
			img.atUnchecked(pos) = std::get<0>(bin);
			idx++;
		}
	}

	StereoVision::ImageProcessing::Histogram<int> histogram(img);

	std::optional<int> threshold = StereoVision::ImageProcessing::computeOtsuThreshold(histogram);

	QVERIFY2(threshold.has_value(), "The function did not return a value!");
	QVERIFY2(threshold.value() == midPoint, "The function did not return the expected value!");
}

QTEST_MAIN(TestSegmentation)
#include "testSegmentation.moc"
