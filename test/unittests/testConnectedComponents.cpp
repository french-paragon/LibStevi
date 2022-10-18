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

#include "imageProcessing/connectedComponents.h"

#include <MultidimArrays/MultidimArrays.h>

#include <random>
#include <set>
#include <vector>
#include <utility>

Multidim::Array<bool, 2> generate2DSquareComponents(int compGridSide, int compDist = 10) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int cellSize = 2*compDist;

	int imgSide = compGridSide*cellSize + compDist;

	Multidim::Array<bool, 2> img (imgSide, imgSide);



	for (int i = 0; i < imgSide; i++) {

		for (int j = 0; j < imgSide; j++) {

			bool activeI = (i/compDist)%2 == 1;
			bool activeJ = (j/compDist)%2 == 1;
			bool active = activeI and activeJ;

			img.at<Nc>(i,j) = active;
		}
	}

	return img;

}


class TestConnectedComponents: public QObject
{

	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testComponentsFinding();

private:
	std::default_random_engine re;

};

void TestConnectedComponents::initTestCase() {

	std::random_device rd;
	re.seed(rd());

}

void TestConnectedComponents::testComponentsFinding() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int gridSide = 2;
	int compSide = 2;

	Multidim::Array<bool, 2> comps = generate2DSquareComponents(gridSide, compSide);

	auto [clusters, clustersInfos] = StereoVision::ImageProcessing::connectedComponents<2, StereoVision::Contiguity::allDimsCanChange>(comps);

	QCOMPARE(clustersInfos.size(), gridSide*gridSide);

	for (uint i = 0; i < clustersInfos.size(); i++) {
		QCOMPARE(clustersInfos[i].boundingBoxCornerMax[0] - clustersInfos[i].boundingBoxCornerMin[0] + 1, compSide);
		QCOMPARE(clustersInfos[i].boundingBoxCornerMax[1] - clustersInfos[i].boundingBoxCornerMin[1] + 1, compSide);
	}

	auto imShape = comps.shape();
	auto csShape = clusters.shape();

	QCOMPARE(csShape.size(), imShape.size());

	QCOMPARE(csShape[0], imShape[0]);
	QCOMPARE(csShape[1], imShape[1]);

	for (int i = 0; i < imShape[0]; i++) {
		for (int j = 0; j < imShape[1]; j++) {

			QVERIFY2((clusters.at<Nc>(i,j) == 0 and comps.at<Nc>(i,j) == false) or (clusters.at<Nc>(i,j) > 0 and comps.at<Nc>(i,j) == true),
					 "Component should not be detected outside of foreground pixels");

			for (int di = -1; di <= 1; di++) {
				for (int dj = -1; dj <= 1; dj++) {

					int shifted_val = clusters.valueOrAlt({i+di,j+dj},0);

					QVERIFY2((shifted_val == clusters.at<Nc>(i,j)) or (shifted_val == 0) or (clusters.at<Nc>(i,j) == 0),
							 "Component should not be detected outside of foreground pixels");
				}
			}

		}
	}

}


QTEST_MAIN(TestConnectedComponents)
#include "testConnectedComponents.moc"
