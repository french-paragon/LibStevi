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

#include <QtTest/QtTest>

#include <statistics/covarianceKernels.h>

using namespace StereoVision::Statistics;

class TestCovarianceKernels: public QObject
{

    Q_OBJECT

private Q_SLOTS:

    void initTestCase();

    void testMaternKernel();

};

void TestCovarianceKernels::initTestCase() {

}

void TestCovarianceKernels::testMaternKernel() {

    float corr_f = CovarianceKernels::Matern<float>::corrFunction(0.5, 42, 69);
    double corr_d = CovarianceKernels::Matern<double>::corrFunction(0.5, 42, 69);

    QVERIFY(std::isfinite(corr_f));
    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_f - std::exp(-69./42.)) < 1e-5);
    QVERIFY(std::abs(corr_d - std::exp(-69./42.)) < 1e-5);

    corr_f = CovarianceKernels::Matern<float>::corrFunction(1.5, 42, 69);
    corr_d = CovarianceKernels::Matern<double>::corrFunction(1.5, 42, 69);

    QVERIFY(std::isfinite(corr_f));
    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_f - (1+std::sqrt(3)*69/42)*std::exp(-std::sqrt(3)*69./42.)) < 1e-5);
    QVERIFY(std::abs(corr_d - (1+std::sqrt(3)*69/42)*std::exp(-std::sqrt(3)*69./42.)) < 1e-5);

    corr_d = CovarianceKernels::Matern<double>::corrFunction(100, 42, 69);

    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_d - 0.25825341) < 1e-3);

    corr_d = CovarianceKernels::Matern<double>::corrFunction(150, 42, 69);

    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_d - std::exp(-(69.*69.)/(2*42.*42.))) < 1e-3);



}

QTEST_MAIN(TestCovarianceKernels)
#include "testCovarianceKernels.moc"
