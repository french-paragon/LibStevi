#ifndef COVARIANCEKERNELS_H
#define COVARIANCEKERNELS_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2025-2026  Paragon<french.paragon@gmail.com>

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

#include <cmath>

namespace StereoVision {
namespace Statistics {

/*!
 * Implementation of covariance kernels for spatial statistics
 */
namespace CovarianceKernels {

/*!
 * \brief The Matern class represent the Matern covariance kernel
 *
 * While this class can be used in a generic fashion, is it optimized for small half integer nu, 1/2, 3/2 and 5/2,
 * and uses exact epxression for both the kernel and derivatives in these cases. Limits are used for large nu and should be numerically stable.
 *
 * Implementation should work for other nu, but risks being numerically unstable, especially with floating point precision.
 */
template<typename T>
class Matern {

public:
    Matern(T nu, T rho) :
        _nu(nu),
        _rho(rho)
    {

    }

    static T corrFunction(T nu, T rho, T d) {

        T scaledD = std::sqrt(2*nu)*d/rho;

        //treat the case of large nu, for large nu, the function converge to an exponential kernel
        if (nu > 150) { //empirical limit observed for double type
            return std::exp(-(d*d)/(2*rho*rho));
        }

        //easier expression for standard half integer formulations
        if (std::abs(nu - 0.5) < 1e-7) {
            return std::exp(-d/rho);
        }
        if (std::abs(nu - 1.5) < 1e-7) {
            return (1 + std::sqrt(3)*d/rho)*std::exp(-std::sqrt(3)*d/rho);
        }
        if (std::abs(nu - 2.5) < 1e-7) {
            return (1 + std::sqrt(5)*d/rho + (5*d*d)/(3*rho*rho))*std::exp(-std::sqrt(5)*d/rho);
        }

        //compute in log space for numerical stability
        T log = (1-nu)*std::log(2) - std::lgamma(nu);
        log += nu*std::log(scaledD);

        T bessel = std::cyl_bessel_k(nu, scaledD);

        T logBessel = std::log(bessel);

        if (std::isfinite(logBessel)) {
            return std::exp(log + logBessel);
        }

        if (!std::isfinite(bessel)) {
            //in case bessel is not finite return the exponential approximation
            return std::exp(-(d*d)/(2*rho*rho));
        }

        return std::exp(log)*bessel;
    }

    static T covFunction(T sigma0, T nu, T rho, T d) {
        return sigma0*sigma0*corrFunction(nu, rho, d);
    }

    static T diffCorrFunctionD(T nu, T rho, T d) {

        T diff = std::numeric_limits<T>::infinity();

        //treat the case of large nu, for large nu, the function converge to an exponential kernel
        if (nu > 150) { //empirical limit observed for double type
            diff = -2*std::exp(-(d*d)/(2*rho*rho)) * d/(2*rho*rho);
        }

        //easier expression for standard half integer formulations
        if (std::abs(nu - 0.5) < 1e-7) {
            diff = -std::exp(-d/rho)/rho;
        }
        if (std::abs(nu - 1.5) < 1e-7) {
            diff = (std::sqrt(3)/rho)*std::exp(-std::sqrt(3)*d/rho) -
                   (1 + std::sqrt(3)*d/rho)*std::exp(-std::sqrt(3)*d/rho) * std::sqrt(3)/rho;
        }
        if (std::abs(nu - 2.5) < 1e-7) {
            diff = (std::sqrt(5)/rho + 2*(5*d)/(3*rho*rho))*std::exp(-std::sqrt(5)*d/rho) -
                   (1 + std::sqrt(5)*d/rho + (5*d*d)/(3*rho*rho))*std::exp(-std::sqrt(5)*d/rho) * std::sqrt(5)/rho;
        }

        if (std::isfinite(diff)) {
            return diff;
        }

        T scaledD = std::sqrt(2*nu)*d/rho;

        //compute in log space for numerical stability
        T log = (1-nu)*std::log(2) - std::lgamma(nu);
        T dlog = (1-nu)*std::log(2) - std::lgamma(nu) + std::log(nu*std::sqrt(2*nu)/rho);
        log += nu*std::log(scaledD);
        dlog += (nu-1)*std::log(scaledD);

        T bessel = std::cyl_bessel_k(nu, scaledD);
        T dbessel = -(std::cyl_bessel_k(nu-1, scaledD) + std::cyl_bessel_k(nu+1, scaledD))/2;

        T logBessel = std::log(bessel);
        T logdBessel = std::log(dbessel);

        if (std::isfinite(logBessel)) {
            diff = std::exp(dlog + logBessel) + std::exp(log + logdBessel);
        }

        if (std::isfinite(diff)) {
            return diff;
        }

        //else use numerical approximation
        T delta = (sizeof(T) <= sizeof(float)) ? 1e-3 : 1e-5;
        T next = corrFunction(nu,rho,d+delta);
        T prev = corrFunction(nu,rho,d-delta);

        return (next - prev) / (2*delta);

    }

    static T diffCorrFunctionRho(T nu, T rho, T d) {

        T diff = std::numeric_limits<T>::infinity();

        //treat the case of large nu, for large nu, the function converge to an exponential kernel
        if (nu > 150) { //empirical limit observed for double type
            diff = 2*std::exp(-(d*d)/(2*rho*rho)) * (d*d)/(2*rho*rho*rho);
        }

        //easier expression for standard half integer formulations
        else if (std::abs(nu - 0.5) < 1e-7) {
            diff = std::exp(-d/rho)*d/(rho*rho);
        }
        else if (std::abs(nu - 1.5) < 1e-7) {
            diff = (-std::sqrt(3)*d/(rho*rho))*std::exp(-std::sqrt(3)*d/rho) +
                   (1 + std::sqrt(3)*d/rho)*std::exp(-std::sqrt(3)*d/rho) * std::sqrt(3)*d/(rho*rho);
        }
        else if (std::abs(nu - 2.5) < 1e-7) {
            diff = (-std::sqrt(5)*d/(rho*rho) - 2*(5*d*d)/(3*rho*rho*rho))*std::exp(-std::sqrt(5)*d/rho) +
                   (1 + std::sqrt(5)*d/rho + (5*d*d)/(3*rho*rho))*std::exp(-std::sqrt(5)*d/rho) * std::sqrt(5)*d/(rho*rho);
        }

        if (std::isfinite(diff)) {
            return diff;
        }

        //else use numerical approximation
        T delta = (sizeof(T) <= sizeof(float)) ? 1e-3 : 1e-5;
        T next = corrFunction(nu,rho+delta,d);
        T prev = corrFunction(nu,rho-delta,d);

        return (next - prev) / (2*delta);

    }

    T operator()(T d) {
        return corrFunction(_nu, _rho, d);
    }

    T operator()(T sigma0, T d) {
        return sigma0*sigma0*corrFunction(_nu, _rho, d);
    }

    T diff(T d) {
        return diffCorrFunctionD(_nu,_rho, d);
    }

    T diff(T sigma0, T d) {
        return sigma0*sigma0*diffCorrFunctionD(_nu,_rho, d);
    }

protected:

    T _nu;
    T _rho;
};

}

} //namespace Statistics
} //namespace StereoVision


#endif // COVARIANCEKERNELS_H
