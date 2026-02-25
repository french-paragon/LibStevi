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
    void testMaternKernelDerivatives();

};

void TestCovarianceKernels::initTestCase() {

}

void TestCovarianceKernels::testMaternKernel() {

    float corr_f = CovarianceKernels::Matern<float>::corrFunction(0.5, 42, 69);
    double corr_d = CovarianceKernels::Matern<double>::corrFunction(0.5, 42, 69);

    float expected = 0.1934266046003925; //computed using maxima
    QVERIFY(std::abs(expected - std::exp(-69./42.)) < 1e-5); //closed form solution

    qInfo() << "Half integer values at 1/2: float = " << corr_f << " double = " << corr_d << " expected = " << expected;

    QVERIFY(std::isfinite(corr_f));
    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_f - expected) < 1e-5);
    QVERIFY(std::abs(corr_d - expected) < 1e-5);

    corr_f = CovarianceKernels::Matern<float>::corrFunction(1.5, 42, 69);
    corr_d = CovarianceKernels::Matern<double>::corrFunction(1.5, 42, 69);

    expected = 0.2234415821938806; //computed using maxima
    QVERIFY(std::abs(expected - (1+std::sqrt(3)*69/42)*std::exp(-std::sqrt(3)*69./42.)) < 1e-5); //closed form solution

    qInfo() << "Half integer values at 3/2: float = " << corr_f << " double = " << corr_d << " expected = " << expected;

    QVERIFY(std::isfinite(corr_f));
    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_f - expected) < 1e-5);
    QVERIFY(std::abs(corr_d - expected) < 1e-5);

    corr_f = CovarianceKernels::Matern<float>::corrFunction(1.1, 27, 33);
    corr_d = CovarianceKernels::Matern<double>::corrFunction(1.1, 27, 33);

    expected = 0.3546855679187097; //computed using maxima

    qInfo() << "Non half integer values at 1.1: float = " << corr_f << " double = " << corr_d << " expected = " << expected;

    QVERIFY(std::isfinite(corr_f));
    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_f - expected) < 1e-5);
    QVERIFY(std::abs(corr_d - expected) < 1e-5);

    corr_d = CovarianceKernels::Matern<double>::corrFunction(100, 42, 69);

    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_d - 0.25825341) < 1e-3);

    corr_d = CovarianceKernels::Matern<double>::corrFunction(150, 42, 69);

    QVERIFY(std::isfinite(corr_d));

    QVERIFY(std::abs(corr_d - std::exp(-(69.*69.)/(2*42.*42.))) < 1e-3);



}


void TestCovarianceKernels::testMaternKernelDerivatives() {

    //d derivative

    float diff_f = CovarianceKernels::Matern<float>::diffCorrFunctionD(1.5, 42, 69);
    double diff_d = CovarianceKernels::Matern<double>::diffCorrFunctionD(1.5, 42, 69);

    float expected_d_derivative = -0.006818386052628004; //computed using maxima

    qInfo() << "Half integer d derivatives: float = " << diff_f << " double = " << diff_d << " expected = " << expected_d_derivative;

    QVERIFY(std::isfinite(diff_f));
    QVERIFY(std::isfinite(diff_d));

    QVERIFY(std::abs(diff_f - expected_d_derivative) < 1e-5);
    QVERIFY(std::abs(diff_d - expected_d_derivative) < 1e-7);

    diff_f = CovarianceKernels::Matern<float>::diffCorrFunctionD(1.1, 27, 33);
    diff_d = CovarianceKernels::Matern<double>::diffCorrFunctionD(1.1, 27, 33);

    expected_d_derivative = -0.01491936535458704; //computed using maxima

    qInfo() << "Non half integer d derivatives: float = " << diff_f << " double = " << diff_d << " expected = " << expected_d_derivative;

    QVERIFY(std::isfinite(diff_f));
    QVERIFY(std::isfinite(diff_d));

    QVERIFY(std::abs(diff_f - expected_d_derivative) < 1e-2);
    QVERIFY(std::abs(diff_d - expected_d_derivative) < 1e-7);

    diff_f = CovarianceKernels::Matern<float>::diffCorrFunctionD(151, 6, 13);
    diff_d = CovarianceKernels::Matern<double>::diffCorrFunctionD(151, 6, 13);

    expected_d_derivative = -0.03432305968508653; //computed using maxima

    qInfo() << "Large nu d derivatives: float = " << diff_f << " double = " << diff_d << " expected = " << expected_d_derivative;

    QVERIFY(std::isfinite(diff_f));
    QVERIFY(std::isfinite(diff_d));

    QVERIFY(std::abs(diff_f - expected_d_derivative) < 1e-3);
    QVERIFY(std::abs(diff_d - expected_d_derivative) < 1e-3);

    //rho derivative

    diff_f = CovarianceKernels::Matern<float>::diffCorrFunctionRho(1.5, 42, 69);
    diff_d = CovarianceKernels::Matern<double>::diffCorrFunctionRho(1.5, 42, 69);

    float expected_rho_derivative = 0.01120163422931744; //computed using maxima

    qInfo() << "Half integer rho derivatives: float = " << diff_f << " double = " << diff_d << " expected = " << expected_rho_derivative;

    QVERIFY(std::isfinite(diff_f));
    QVERIFY(std::isfinite(diff_d));

    QVERIFY(std::abs(diff_f - expected_rho_derivative) < 1e-5);
    QVERIFY(std::abs(diff_d - expected_rho_derivative) < 1e-7);

    diff_f = CovarianceKernels::Matern<float>::diffCorrFunctionRho(1.1, 27, 33);
    diff_d = CovarianceKernels::Matern<double>::diffCorrFunctionRho(1.1, 27, 33);

    expected_rho_derivative = 0.01823477987782858; //computed using maxima

    qInfo() << "Non half integer rho derivatives: float = " << diff_f << " double = " << diff_d << " expected = " << expected_rho_derivative;

    QVERIFY(std::isfinite(diff_f));
    QVERIFY(std::isfinite(diff_d));

    QVERIFY(std::abs(diff_f - expected_rho_derivative) < 1e-3);
    QVERIFY(std::abs(diff_d - expected_rho_derivative) < 1e-7);

    diff_f = CovarianceKernels::Matern<float>::diffCorrFunctionRho(151, 6, 13);
    diff_d = CovarianceKernels::Matern<double>::diffCorrFunctionRho(151, 6, 13);

    expected_rho_derivative = 0.07436662931768368; //computed using maxima, not 100% sure about that one, it looks like it is flirting with numerical precision limits in maxima

    qInfo() << "Non half integer rho derivatives: float = " << diff_f << " double = " << diff_d << " expected = " << expected_rho_derivative;

    QVERIFY(std::isfinite(diff_f));
    QVERIFY(std::isfinite(diff_d));

    QVERIFY(std::abs(diff_f - expected_rho_derivative) < 1e-3);
    QVERIFY(std::abs(diff_d - expected_rho_derivative) < 1e-3);
}

QTEST_MAIN(TestCovarianceKernels)
#include "testCovarianceKernels.moc"
