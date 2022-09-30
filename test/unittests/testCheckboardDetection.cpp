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

#include "imageProcessing/checkBoardDetection.h"

#include <MultidimArrays/MultidimArrays.h>

#include <random>
#include <set>
#include <vector>
#include <utility>

std::vector<StereoVision::discretCheckCornerInfos> generateTestCandidatesSet(int nVertical = 3,
																			 int nHorizontal = 4,
																			 int nWrong = 0,
																			 int dist = 10) {

	std::set<std::pair<int, int>> addedPoints;
	std::vector<StereoVision::discretCheckCornerInfos> ret;

	for (int i = 0; i < nVertical; i++) {
		for (int j = 0; j < nHorizontal; j++) {

			float angle = M_PI_4;

			if ((i+j)%2 == 1) {
				angle += M_PI_2;
			}

			ret.emplace_back(j*dist, i*dist, -1, 1, angle);

			addedPoints.insert({j*dist, i*dist});
		}
	}

	float wrong_angle = M_PI_2;
	std::random_device rd;
	std::default_random_engine re;
	re.seed(rd());

	std::uniform_int_distribution rPosHorz(-dist, dist*(nHorizontal+1));
	std::uniform_int_distribution rPosVert(-dist, dist*(nVertical+1));

	for (int i = 0; i < nWrong; i++) {

		int x = rPosHorz(re);
		int y = rPosVert(re);

		while (addedPoints.count({x,y}) > 0) {
			x++;
			y++;
		}

		ret.emplace_back(x, y, -1, 1, wrong_angle);

		addedPoints.insert({x, y});
	}

	return ret;

}

Multidim::Array<float, 2> generateBoard(int nVertical,
										int nHorizontal,
										int squareSide = 30,
										bool invert = false) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 2> ret((nVertical+1)*squareSide + nVertical,
								  (nHorizontal+1)*squareSide + nHorizontal);

	for (int i = 0; i <= nVertical; i++) {

		int startVert = i*(squareSide+1);

		for (int j = 0; j <= nHorizontal; j++) {

			int starHorz = j*(squareSide+1);

			float color = (i%2 == j%2) ? 1. : 0.;
			if (invert) {
				color = (i%2 == j%2) ? 0. : 1.;
			}

			for (int pixi = 0; pixi < squareSide; pixi++) {
				for (int pixj = 0; pixj < squareSide; pixj++) {

					ret.at<Nc>(startVert + pixi, starHorz+pixj) = color;

				}
			}

			for (int pix = 0; pix < squareSide; pix++) {
				if (startVert + squareSide < ret.shape()[0]) {
					ret.at<Nc>(startVert + squareSide, starHorz+pix) = 0.5;
				}

				if (starHorz + squareSide < ret.shape()[1]) {
					ret.at<Nc>(startVert + pix, starHorz+squareSide) = 0.5;
				}
			}

			if (startVert + squareSide < ret.shape()[0] and starHorz + squareSide < ret.shape()[1]) {
				ret.at<Nc>(startVert + squareSide, starHorz+squareSide) = 0.5;
			}

		}
	}

	return ret;

}

Multidim::Array<float, 2> generateSubPixelModel(
		float subX,
		float subY,
		float theta,
		int radius = 3) {

	constexpr int aliasing_radius = 4;
	constexpr float aliasing_delta = 0.5;


	int side = 2*radius+1;

	Multidim::Array<float, 2> ret(side, side);

	for (int pos_y = -radius; pos_y <= radius; pos_y++) {
		for (int pos_x = -radius; pos_x <= radius; pos_x++) {

			float mean = 0;
			float sumWeight = 0;

			for (int delta_pos_y = -aliasing_radius; delta_pos_y <= aliasing_radius; delta_pos_y++) {
				for (int delta_pos_x = -aliasing_radius; delta_pos_x <= aliasing_radius; delta_pos_x++) {

					float aliasingDistX = delta_pos_x*aliasing_delta;
					float aliasingDistY = delta_pos_y*aliasing_delta;
					float weigth = std::exp(-(aliasingDistX*aliasingDistX + aliasingDistY*aliasingDistY));

					float px = pos_x + aliasingDistX - subX;
					float py = pos_y + aliasingDistY - subY;

					float trsfX = std::cos(-theta)*px - std::sin(-theta)*py;
					float trsfY = std::cos(-theta)*py + std::sin(-theta)*px;

					if (trsfX*trsfY > 0) {
						mean += weigth;
					}

					sumWeight += weigth;

				}
			}

			mean /= sumWeight;

			ret.atUnchecked(pos_y+radius,pos_x+radius) = mean;

		}
	}

	return ret;

}

class TestCheckboardDetection: public QObject
{

	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testCandidateFinding();
	void testCheckBoardExtraction();
	void testCandidateRefinement();

private:
	std::default_random_engine re;

};

void TestCheckboardDetection::initTestCase() {

	std::random_device rd;
	re.seed(rd());

}

void TestCheckboardDetection::testCandidateFinding() {

	int squareside = 30;
	int nVert = 3;
	int nHorz = 3;

	Multidim::Array<float, 2> board = generateBoard(nVert,nHorz,squareside);

	auto candidates = StereoVision::checkBoardCornersCandidates(board, 1, 2, 0.0001);

	QCOMPARE(candidates.size(), nVert*nHorz);

}


void TestCheckboardDetection::testCheckBoardExtraction() {

	int squareside = 30;
	int nVert = 3;
	int nHorz = 4;
	int nWrong = 7;

	std::vector<StereoVision::discretCheckCornerInfos> pure = generateTestCandidatesSet(nVert, nHorz, 0, squareside);

	auto extracted = StereoVision::isolateCheckBoard(pure);

	QCOMPARE(extracted.nPointsFound(), nVert*nHorz);
	QCOMPARE(extracted.rows(), nVert);
	QCOMPARE(extracted.cols(), nHorz);

	std::vector<StereoVision::discretCheckCornerInfos> wrong = generateTestCandidatesSet(nVert, nHorz, nWrong, squareside);

	extracted = StereoVision::isolateCheckBoard(wrong);

	QCOMPARE(extracted.nPointsFound(), nVert*nHorz);
	QCOMPARE(extracted.rows(), nVert);
	QCOMPARE(extracted.cols(), nHorz);
}



void  TestCheckboardDetection::testCandidateRefinement() {

	constexpr int radius = 3;

	std::uniform_real_distribution<float> subPixelRange(-0.3, 0.3);
	std::uniform_real_distribution<float> thetaRange(-M_PI, M_PI);

	float subX = subPixelRange(re);
	float subY = subPixelRange(re);

	float theta = thetaRange(re);

	Multidim::Array<float, 2> subpixelModel = generateSubPixelModel(subX, subY, theta, radius);

	std::cout << subpixelModel << std::endl;

	Eigen::Vector2f subpixel = StereoVision::fitCheckboardCornerCenter(subpixelModel, Eigen::Vector2i(radius, radius), theta + M_PI_4 + 0.01, radius, 15);

	QVERIFY2(std::fabs(subpixel.x() - subX) < 1e-1, "Unacurate subpixel matching !");
	QVERIFY2(std::fabs(subpixel.y() - subY) < 1e-1, "Unacurate subpixel matching !");

}

QTEST_MAIN(TestCheckboardDetection)
#include "testCheckboardDetection.moc"
