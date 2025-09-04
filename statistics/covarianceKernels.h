#ifndef COVARIANCEKERNELS_H
#define COVARIANCEKERNELS_H

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

#include <cmath>

namespace StereoVision {
namespace Statistics {

/*!
 * Implementation of covariance kernels for spatial statistics
 */
namespace CovarianceKernels {

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

    T operator()(T d) {
        return corrFunction(_nu, _rho, d);
    }

    T operator()(T sigma0, T d) {
        return sigma0*sigma0*corrFunction(_nu, _rho, d);
    }

protected:

    T _nu;
    T _rho;
};

}

} //namespace Statistics
} //namespace StereoVision


#endif // COVARIANCEKERNELS_H
