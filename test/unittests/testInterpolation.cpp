#include <QtTest/QtTest>

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

#include "interpolation/interpolation.h"

#include <MultidimArrays/MultidimArrays.h>

#include <iostream>
#include <random>
#include <optional>

using namespace StereoVision;

class TestInterpolationAlgorithms: public QObject
{

    Q_OBJECT
private Q_SLOTS:

    void initTestCase();

    void testValueInterpolation();

private:
    std::default_random_engine re;

};


void TestInterpolationAlgorithms::initTestCase() {
    //srand((unsigned int) time(nullptr));
    std::random_device rd;
    re.seed(rd());
}

void TestInterpolationAlgorithms::testValueInterpolation() {

    std::uniform_real_distribution<float> dist(-1,1);

    //2D interpolation with basic kernel

    for (int i = 0; i < 10; i++) {
        Multidim::Array<float,2> test(2,2);
        float interpolated = 0;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                float val = dist(re);
                interpolated += 0.25*val;
                test.atUnchecked(i,j) = val;
            }
        }

        float val = Interpolation::interpolateValue<2, float, Interpolation::pyramidFunction<float, 2>, 0>(test, {0.5,0.5});

        QCOMPARE(val, interpolated);
    }

}


QTEST_MAIN(TestInterpolationAlgorithms)
#include "testInterpolation.moc"
