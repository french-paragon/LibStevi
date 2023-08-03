#ifndef STANDARDCONVOLUTIONFILTERS_H
#define STANDARDCONVOLUTIONFILTERS_H

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

#include "./convolutions.h"

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <cmath>

namespace StereoVision {
namespace ImageProcessing {
namespace Convolution {

template<typename T, typename... Ds>
Filter<T, Ds...> uniformGaussianFilter(T sigma, int radius, bool normalize, Ds... axisDefinitions) {

    using CoeffType = Multidim::Array<T, Filter<T, Ds...>::nFilterAxes>;

    std::array<int, Filter<T, Ds...>::nFilterAxes> shape;

    for (int i = 0; i < Filter<T, Ds...>::nFilterAxes; i++) {
        shape[i] = 2*radius+1;
    }

    CoeffType coefficients(shape);

    Multidim::IndexConverter<Filter<T, Ds...>::nFilterAxes> idxCvrt(shape);

    double var = sigma*sigma;

    std::vector<typename CoeffType::ShapeBlock> idxs(idxCvrt.numberOfPossibleIndices());

    for (int i = 0; i < idxCvrt.numberOfPossibleIndices(); i++) {
        auto idx = idxCvrt.getIndexFromPseudoFlatId(i);
        idxs[i] = idx;
    }

    for (auto const& idx : idxs) {

        double val = 1;

        for (int d : idx) {
            double delta = d - radius;
            val *= std::exp(-delta*delta/var);
        }

        coefficients.atUnchecked(idx) = static_cast<T>(val);
    }

    if (normalize) {
        T sum = 0;

        for (auto const& idx : idxs) {
            sum += coefficients.valueUnchecked(idx);
        }

        for (auto const& idx : idxs) {
            coefficients.atUnchecked(idx) /= sum;
        }
    }

    return Filter<T, Ds...>(coefficients, axisDefinitions...);
}

} //namespace Convolution
} //namespace ImageProcessing
} //namespace StereoVision

#endif // STANDARDCONVOLUTIONFILTERS_H
