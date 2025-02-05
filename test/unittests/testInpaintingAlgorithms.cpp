#include <QtTest/QtTest>

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2025  Paragon<french.paragon@gmail.com>

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

#include "imageProcessing/inpainting.h"

#include <MultidimArrays/MultidimArrays.h>

#include <iostream>
#include <random>
#include <optional>

using namespace StereoVision;
using namespace StereoVision::ImageProcessing;

class TestInpaintingAlgorithms: public QObject
{

    Q_OBJECT
private Q_SLOTS:

    void initTestCase();

    void testNearestPixelInpainting();

private:
    std::default_random_engine re;

};

void TestInpaintingAlgorithms::initTestCase() {
    std::random_device rd;
    re.seed(rd());
}

void TestInpaintingAlgorithms::testNearestPixelInpainting() {

    using imgT = uint8_t;

    std::uniform_int_distribution<imgT> rngColor(0,255);

    constexpr int side = 42;
    constexpr int depth = 6;
    constexpr int nPts = 27;

    std::uniform_int_distribution<int> rngPtPos(0,side-1);

    std::array<uint8_t,depth> bg_color;
    std::array<uint8_t,depth> pt_color;

    for (int c = 0; c < depth; c++) {
        bg_color[c] = rngColor(re);
        pt_color[c] = rngColor(re);
    }

    Multidim::Array<imgT,3> img(side,side,depth);
    Multidim::Array<bool,2> mask(side,side);
    std::vector<std::array<int,2>> points(nPts);

    for (int i = 0; i < side; i++) {
        for (int j = 0; j < side; j++) {

            mask.atUnchecked(i,j) = false;

            for (int c = 0; c < depth; c++) {
                img.atUnchecked(i,j,c) = bg_color[c];
            }
        }
    }

    for (int p = 0; p < nPts; p++) {

        int i = rngPtPos(re);
        int j = rngPtPos(re);

        points[p][0] = i;
        points[p][1] = j;

        mask.atUnchecked(i,j) = true;

        for (int di = -1; di <= 1; di++) {

            if (i+di < 0 or i + di >= side) {
                continue;
            }

            for (int dj = -1; dj <= 1; dj++) {

                if (j+dj < 0 or j + dj >= side) {
                    continue;
                }

                for (int c = 0; c < depth; c++) {
                    img.atUnchecked(i+di,j+dj,c) = pt_color[c];
                }

            }
        }

        for (int c = 0; c < depth; c++) { // ensure the target points have the wrong color
            img.atUnchecked(i,j,c) = bg_color[c];
        }

    }

    Multidim::Array<imgT,3> inpainted = nearestInPaintingBatched<imgT, 3, 1>(img, mask, {2});

    QCOMPARE(inpainted.shape()[0], img.shape()[0]);
    QCOMPARE(inpainted.shape()[1], img.shape()[1]);
    QCOMPARE(inpainted.shape()[2], img.shape()[2]);

    for (int p = 0; p < nPts; p++) {

        int i = points[p][0];
        int j = points[p][1];

        for (int c = 0; c < depth; c++) { // ensure the target points now have the expected color
            QCOMPARE(inpainted.atUnchecked(i,j,c), pt_color[c]);
        }

    }

}

QTEST_MAIN(TestInpaintingAlgorithms)
#include "testInpaintingAlgorithms.moc"
