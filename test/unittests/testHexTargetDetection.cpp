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

#include "imageProcessing/hexagonalRGBTargetsDetection.h"

#include "io/image_io.h"

#include <MultidimArrays/MultidimArrays.h>

#include <random>
#include <set>
#include <vector>
#include <utility>

template<StereoVision::Color::RedGreenBlue MC = StereoVision::Color::Blue,
		 StereoVision::Color::RedGreenBlue PC = StereoVision::Color::Red,
		 StereoVision::Color::RedGreenBlue NC = StereoVision::Color::Green>
Multidim::Array<uint8_t, 3> generateTestTarget(int imgRadius = 250,
											   bool color1_positiv = true,
											   bool color2_positiv = true,
											   bool color3_positiv = false,
											   bool color4_positiv = true,
											   bool color5_positiv = false) {

	static_assert (StereoVision::ImageProcessing::HexRgbTarget::validateHexTargetColors(MC, PC, NC), "Invalid color scheme provided");

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int imgSide = 2*imgRadius + 1;

	Multidim::Array<uint8_t, 3> img (imgSide, imgSide, 3);

	float big_radius = 0.7*imgRadius;
	float small_radius = 0.1*imgRadius;
	Eigen::Vector2f origin = Eigen::Vector2f(big_radius + imgRadius, imgRadius);
	Eigen::Vector2f pt1 = Eigen::Vector2f(big_radius*std::cos(M_PI/3) + imgRadius, big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt2 = Eigen::Vector2f(-big_radius*std::cos(M_PI/3) + imgRadius, big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt3 = Eigen::Vector2f(-big_radius + imgRadius, imgRadius);
	Eigen::Vector2f pt4 = Eigen::Vector2f(-big_radius*std::cos(M_PI/3) + imgRadius, -big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt5 = Eigen::Vector2f(big_radius*std::cos(M_PI/3) + imgRadius, -big_radius*std::sin(M_PI/3) + imgRadius);

	int refColor_channel = (MC == StereoVision::Color::Blue) ? 2 : ((MC == StereoVision::Color::Green) ? 1 : 0);
	int posColor_channel = (PC == StereoVision::Color::Blue) ? 2 : ((PC == StereoVision::Color::Green) ? 1 : 0);
	int negColor_channel = (NC == StereoVision::Color::Blue) ? 2 : ((NC == StereoVision::Color::Green) ? 1 : 0);

	for (int i = 0; i < imgSide; i++) {

		for (int j = 0; j < imgSide; j++) {

			img.at<Nc>(i,j,0) = 255;
			img.at<Nc>(i,j,1) = 255;
			img.at<Nc>(i,j,2) = 255;

			Eigen::Vector2f pos = Eigen::Vector2f(j,i);

			if ((pos - origin).norm() < small_radius) {
				img.at<Nc>(i,j,refColor_channel) = 255;
				img.at<Nc>(i,j,posColor_channel) = 0;
				img.at<Nc>(i,j,negColor_channel) = 0;
			}

			if ((pos - pt1).norm() < small_radius) {
				img.at<Nc>(i,j,refColor_channel) = 0;
				img.at<Nc>(i,j,(color1_positiv) ? posColor_channel : negColor_channel) = 255;
				img.at<Nc>(i,j,(color1_positiv) ? negColor_channel : posColor_channel) = 0;
			}

			if ((pos - pt2).norm() < small_radius) {
				img.at<Nc>(i,j,refColor_channel) = 0;
				img.at<Nc>(i,j,(color2_positiv) ? posColor_channel : negColor_channel) = 255;
				img.at<Nc>(i,j,(color2_positiv) ? negColor_channel : posColor_channel) = 0;
			}

			if ((pos - pt3).norm() < small_radius) {
				img.at<Nc>(i,j,refColor_channel) = 0;
				img.at<Nc>(i,j,(color3_positiv) ? posColor_channel : negColor_channel) = 255;
				img.at<Nc>(i,j,(color3_positiv) ? negColor_channel : posColor_channel) = 0;
			}

			if ((pos - pt4).norm() < small_radius) {
				img.at<Nc>(i,j,refColor_channel) = 0;
				img.at<Nc>(i,j,(color4_positiv) ? posColor_channel : negColor_channel) = 255;
				img.at<Nc>(i,j,(color4_positiv) ? negColor_channel : posColor_channel) = 0;
			}

			if ((pos - pt5).norm() < small_radius) {
				img.at<Nc>(i,j,refColor_channel) = 0;
				img.at<Nc>(i,j,(color5_positiv) ? posColor_channel : negColor_channel) = 255;
				img.at<Nc>(i,j,(color5_positiv) ? negColor_channel : posColor_channel) = 0;
			}
		}
	}

	return img;

}

template<StereoVision::Color::RedGreenBlue MC = StereoVision::Color::Blue,
		 StereoVision::Color::RedGreenBlue PC = StereoVision::Color::Red,
		 StereoVision::Color::RedGreenBlue NC = StereoVision::Color::Green>
Multidim::Array<uint8_t, 3> generateSubpixelTestTarget(int imgRadius = 250,
													   float vShift = 0.3,
													   float hShift = 0.7,
													   float relBlurDist = 0.1,
													   bool color1_positiv = true,
													   bool color2_positiv = true,
													   bool color3_positiv = false,
													   bool color4_positiv = true,
													   bool color5_positiv = false) {

	static_assert (StereoVision::ImageProcessing::HexRgbTarget::validateHexTargetColors(MC, PC, NC), "Invalid color scheme provided");

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int imgSide = 2*imgRadius + 1;

	Multidim::Array<uint8_t, 3> img (imgSide, imgSide, 3);

	float big_radius = 0.7*imgRadius;
	float small_radius = 0.1*imgRadius;
	float blurDist = relBlurDist*small_radius;
	Eigen::Vector2f origin = Eigen::Vector2f(big_radius + imgRadius, imgRadius);
	Eigen::Vector2f pt1 = Eigen::Vector2f(big_radius*std::cos(M_PI/3) + imgRadius, big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt2 = Eigen::Vector2f(-big_radius*std::cos(M_PI/3) + imgRadius, big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt3 = Eigen::Vector2f(-big_radius + imgRadius, imgRadius);
	Eigen::Vector2f pt4 = Eigen::Vector2f(-big_radius*std::cos(M_PI/3) + imgRadius, -big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt5 = Eigen::Vector2f(big_radius*std::cos(M_PI/3) + imgRadius, -big_radius*std::sin(M_PI/3) + imgRadius);

	int refColor_channel = (MC == StereoVision::Color::Blue) ? 2 : ((MC == StereoVision::Color::Green) ? 1 : 0);
	int posColor_channel = (PC == StereoVision::Color::Blue) ? 2 : ((PC == StereoVision::Color::Green) ? 1 : 0);
	int negColor_channel = (NC == StereoVision::Color::Blue) ? 2 : ((NC == StereoVision::Color::Green) ? 1 : 0);

	for (int i = 0; i < imgSide; i++) {

		for (int j = 0; j < imgSide; j++) {

			img.at<Nc>(i,j,0) = 255;
			img.at<Nc>(i,j,1) = 255;
			img.at<Nc>(i,j,2) = 255;

			Eigen::Vector2f pos = Eigen::Vector2f(j,i);
			pos[1] -= vShift;
			pos[0] -= hShift;

			uint8_t colorvalue = 0;

			int coloredChannel = refColor_channel;

			float nOrigin = (pos - origin).norm();
			if ( nOrigin < small_radius+blurDist) {
				coloredChannel = refColor_channel;

				if (nOrigin > small_radius) {
					colorvalue = (1-(nOrigin - small_radius)/blurDist) * 255;
				} else {
					colorvalue = 255;
				}
			}

			float nPt1 = (pos - pt1).norm();
			if ( nPt1 < small_radius+blurDist) {
				coloredChannel = (color1_positiv) ? posColor_channel : negColor_channel;

				if (nPt1 > small_radius) {
					colorvalue = (1-(nPt1 - small_radius)/blurDist) * 255;
				} else {
					colorvalue = 255;
				}
			}

			float nPt2 = (pos - pt2).norm();
			if ( nPt2 < small_radius+blurDist) {
				coloredChannel = (color2_positiv) ? posColor_channel : negColor_channel;

				if (nPt2 > small_radius) {
					colorvalue = (1-(nPt2 - small_radius)/blurDist) * 255;
				} else {
					colorvalue = 255;
				}
			}

			float nPt3 = (pos - pt3).norm();
			if ( nPt3 < small_radius+blurDist) {
				coloredChannel = (color3_positiv) ? posColor_channel : negColor_channel;

				if (nPt3 > small_radius) {
					colorvalue = (1-(nPt3 - small_radius)/blurDist) * 255;
				} else {
					colorvalue = 255;
				}
			}

			float nPt4 = (pos - pt4).norm();
			if ( nPt4 < small_radius+blurDist) {
				coloredChannel = (color4_positiv) ? posColor_channel : negColor_channel;

				if (nPt4 > small_radius) {
					colorvalue = (1-(nPt4 - small_radius)/blurDist) * 255;
				} else {
					colorvalue = 255;
				}
			}

			float nPt5 = (pos - pt5).norm();
			if ( nPt5 < small_radius+blurDist) {
				coloredChannel = (color5_positiv) ? posColor_channel : negColor_channel;

				if (nPt5 > small_radius) {
					colorvalue = (1-(nPt5 - small_radius)/blurDist) * 255;
				} else {
					colorvalue = 255;
				}
			}

			for (int c = 0; c < 3; c++) {

				img.at<Nc>(i,j,c) = 255;

				if (c == coloredChannel) {
					continue;
				}

				img.at<Nc>(i,j,c) = 255 - colorvalue;
			}
		}
	}

	return img;

}


class TestHexTargetDetection: public QObject
{

	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testTargetFinding();
	void testSubpixelAdjustement();

private:
	std::default_random_engine re;

};

void TestHexTargetDetection::initTestCase() {

	std::random_device rd;
	re.seed(rd());

}

void TestHexTargetDetection::testTargetFinding() {

	constexpr StereoVision::Color::RedGreenBlue MC = StereoVision::Color::Blue;
	constexpr StereoVision::Color::RedGreenBlue PC = StereoVision::Color::Green;
	constexpr StereoVision::Color::RedGreenBlue NC = StereoVision::Color::Red;

	int imgRadius = 250;

	bool color1pos = true;
	bool color2pos = true;
	bool color3pos = false;
	bool color4pos = true;
	bool color5pos = false;

	Multidim::Array<uint8_t, 3> target = generateTestTarget<MC, PC, NC>(imgRadius,
																		color1pos,
																		color2pos,
																		color3pos,
																		color4pos,
																		color5pos);


	std::vector<StereoVision::ImageProcessing::HexRgbTarget::HexTargetPosition> candidates =
			StereoVision::ImageProcessing::HexRgbTarget::detectHexTargets<uint8_t, MC, PC, NC>(target, 50, 10, 10, 2000, 0.6, 1.0);

	QCOMPARE(candidates.size(), 1);

	QCOMPARE(candidates[0].dotsPositives[0], color1pos);
	QCOMPARE(candidates[0].dotsPositives[1], color2pos);
	QCOMPARE(candidates[0].dotsPositives[2], color3pos);
	QCOMPARE(candidates[0].dotsPositives[3], color4pos);
	QCOMPARE(candidates[0].dotsPositives[4], color5pos);

}

void TestHexTargetDetection::testSubpixelAdjustement() {

	constexpr StereoVision::Color::RedGreenBlue MC = StereoVision::Color::Blue;
	constexpr StereoVision::Color::RedGreenBlue PC = StereoVision::Color::Green;
	constexpr StereoVision::Color::RedGreenBlue NC = StereoVision::Color::Red;

	int imgRadius = 250;

	float vShift = 0.3;
	float hShift = 0.7;
	float relBlurDist = 0.1;

	bool color1pos = true;
	bool color2pos = true;
	bool color3pos = false;
	bool color4pos = true;
	bool color5pos = false;


	float big_radius = 0.7*imgRadius;
	Eigen::Vector2f origin = Eigen::Vector2f(big_radius + imgRadius, imgRadius);
	Eigen::Vector2f pt1 = Eigen::Vector2f(big_radius*std::cos(M_PI/3) + imgRadius, big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt2 = Eigen::Vector2f(-big_radius*std::cos(M_PI/3) + imgRadius, big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt3 = Eigen::Vector2f(-big_radius + imgRadius, imgRadius);
	Eigen::Vector2f pt4 = Eigen::Vector2f(-big_radius*std::cos(M_PI/3) + imgRadius, -big_radius*std::sin(M_PI/3) + imgRadius);
	Eigen::Vector2f pt5 = Eigen::Vector2f(big_radius*std::cos(M_PI/3) + imgRadius, -big_radius*std::sin(M_PI/3) + imgRadius);

	Multidim::Array<uint8_t, 3> target = generateSubpixelTestTarget<MC, PC, NC>(imgRadius,
																				vShift,
																				hShift,
																				relBlurDist,
																				color1pos,
																				color2pos,
																				color3pos,
																				color4pos,
																				color5pos);

	std::vector<StereoVision::ImageProcessing::HexRgbTarget::HexTargetPosition> candidates =
			StereoVision::ImageProcessing::HexRgbTarget::detectHexTargets<uint8_t, MC, PC, NC>(target, 50, 10, 10, 2500, 0.6, 1.0);

	QCOMPARE(candidates.size(), 1);

	QCOMPARE(candidates[0].dotsPositives[0], color1pos);
	QCOMPARE(candidates[0].dotsPositives[1], color2pos);
	QCOMPARE(candidates[0].dotsPositives[2], color3pos);
	QCOMPARE(candidates[0].dotsPositives[3], color4pos);
	QCOMPARE(candidates[0].dotsPositives[4], color5pos);

	const float tol = 0.2;

	float vAlignPt1 = candidates[0].dotsPositions[0][0] - (pt1[1] + vShift);
	float hAlignPt1 = candidates[0].dotsPositions[0][1] - (pt1[0] + hShift);

	float vAlignPt2 = candidates[0].dotsPositions[1][0] - (pt2[1] + vShift);
	float hAlignPt2 = candidates[0].dotsPositions[1][1] - (pt2[0] + hShift);

	float vAlignPt3 = candidates[0].dotsPositions[2][0] - (pt3[1] + vShift);
	float hAlignPt3 = candidates[0].dotsPositions[2][1] - (pt3[0] + hShift);

	float vAlignPt4 = candidates[0].dotsPositions[3][0] - (pt4[1] + vShift);
	float hAlignPt4 = candidates[0].dotsPositions[3][1] - (pt4[0] + hShift);

	float vAlignPt5 = candidates[0].dotsPositions[4][0] - (pt5[1] + vShift);
	float hAlignPt5 = candidates[0].dotsPositions[4][1] - (pt5[0] + hShift);

	QVERIFY2(std::abs(vAlignPt1) < tol, "Misaligned subpixel correction");
	QVERIFY2(std::abs(hAlignPt1) < tol, "Misaligned subpixel correction");

	QVERIFY2(std::abs(vAlignPt2) < tol, "Misaligned subpixel correction");
	QVERIFY2(std::abs(hAlignPt2) < tol, "Misaligned subpixel correction");

	QVERIFY2(std::abs(vAlignPt3) < tol, "Misaligned subpixel correction");
	QVERIFY2(std::abs(hAlignPt3) < tol, "Misaligned subpixel correction");

	QVERIFY2(std::abs(vAlignPt4) < tol, "Misaligned subpixel correction");
	QVERIFY2(std::abs(hAlignPt4) < tol, "Misaligned subpixel correction");

	QVERIFY2(std::abs(vAlignPt5) < tol, "Misaligned subpixel correction");
	QVERIFY2(std::abs(hAlignPt5) < tol, "Misaligned subpixel correction");

}


QTEST_MAIN(TestHexTargetDetection)
#include "testHexTargetDetection.moc"
