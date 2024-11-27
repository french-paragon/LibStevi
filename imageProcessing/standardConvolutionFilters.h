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

namespace Internals {

template<int idx, bool keepStrides, typename T>
T removePaddingFromNonSeparated(int axisId, T const& axisDefinition) {
    return axisDefinition;
}

template<int axisId, bool keepStrides>
MovingWindowAxis removePaddingFromNonSeparated(int nonCollapsedAxisId, MovingWindowAxis const& axisDefinition) {

    int stride = 0;

    if (keepStrides) {
        stride = axisDefinition.stride();
    }

    if (axisId == nonCollapsedAxisId) {
        return MovingWindowAxis(stride, axisDefinition.padding()); //if the axis is the separated axis, do not remove padding
    }

    return MovingWindowAxis(stride, PaddingInfos()); //remove padding in other cases.
}

template<typename T, bool keepStrides, typename... Ds, size_t... idxs>
Filter<T, Ds...> instanciateSeparatedFilter(Multidim::Array<T, Filter<T, Ds...>::nFilterAxes> const& coefficients,
                                            int movingAxId,
                                            std::index_sequence<idxs...>,
                                            Ds... axisDefinitions) {

    int AxId = -1;
    int count = movingAxId;

    auto axisTypes = Filter<T, Ds...>::axisTypes;

    for (int i = 0; i < axisTypes.size(); i++) {
        if (axisTypes[i] == AxisType::Moving) {
            if (count == 0) {
                AxId = i;
                break;
            }
            count--;
        }
    }

    assert(AxId >= 0 and count == 0); //check for error

    return Filter<T, Ds...>(coefficients, removePaddingFromNonSeparated<idxs, keepStrides>(AxId, axisDefinitions)...);
}

}

template<typename T, typename... Ds>
Filter<T, Ds...> constantFilter(T val, int radius, Ds... axisDefinitions) {

    using CoeffType = Multidim::Array<T, Filter<T, Ds...>::nFilterAxes>;

    std::array<int, Filter<T, Ds...>::nFilterAxes> shape;

    for (int i = 0; i < Filter<T, Ds...>::nFilterAxes; i++) {
        shape[i] = 2*radius+1;
    }

    CoeffType coefficients(shape);

    Multidim::IndexConverter<Filter<T, Ds...>::nFilterAxes> idxCvrt(shape);

    for (int i = 0; i < idxCvrt.numberOfPossibleIndices(); i++) {

        auto idx = idxCvrt.getIndexFromPseudoFlatId(i);

        coefficients.atUnchecked(idx) = static_cast<T>(val);
    }

    return Filter<T, Ds...>(coefficients, axisDefinitions...);

}

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


template<typename T, typename... Ds>
std::array<Filter<T, Ds...>, Filter<T, Ds...>::nAxesOfType(AxisType::Moving)> separatedConstantFilter(T val, int radius, Ds... axisDefinitions) {
    using CoeffType = Multidim::Array<T, Filter<T, Ds...>::nFilterAxes>;

    constexpr int nFilters = Filter<T, Ds...>::nAxesOfType(AxisType::Moving);

    std::array<Filter<T, Ds...>, nFilters> ret;

    for (int i = 0; i < nFilters; i++) {

        std::array<int, Filter<T, Ds...>::nFilterAxes> shape;

        for (int j = 0; j < Filter<T, Ds...>::nFilterAxes; j++) {
            shape[j] = 1;
        }

        shape[i] = 2*radius+1;

        CoeffType coefficients(shape);

        std::array<int, Filter<T, Ds...>::nFilterAxes> idx;

        for (int j = 0; j < Filter<T, Ds...>::nFilterAxes; j++) {
            idx[j] = 0;
        }

        double sum = 0;

        for (int d = 0; d < 2*radius+1; d++) {
            idx[i] = d;

            coefficients.atUnchecked(idx) = (i == 0) ? val : 1;
        }

        if (i == 0) {
            ret[i] = Internals::instanciateSeparatedFilter<T, true>(coefficients, i, std::index_sequence_for<Ds...>(), axisDefinitions...);
        } else {
            ret[i] = Internals::instanciateSeparatedFilter<T, false>(coefficients, i, std::index_sequence_for<Ds...>(), axisDefinitions...);
        }

    }

    return ret;
}

/*!
 * \brief separatedGaussianFilters return a collection of filters which, when applied sequentially, gives a Gaussian
 * \param axisDefinitions
 * \return
 */
template<typename T, typename... Ds>
std::array<Filter<T, Ds...>, Filter<T, Ds...>::nAxesOfType(AxisType::Moving)> separatedGaussianFilters(T sigma, int radius, bool normalize, Ds... axisDefinitions) {

    using CoeffType = Multidim::Array<T, Filter<T, Ds...>::nFilterAxes>;

    constexpr int nFilters = Filter<T, Ds...>::nAxesOfType(AxisType::Moving);

    std::array<Filter<T, Ds...>, nFilters> ret;

    double var = sigma*sigma;

    for (int i = 0; i < nFilters; i++) {

        std::array<int, Filter<T, Ds...>::nFilterAxes> shape;

        for (int j = 0; j < Filter<T, Ds...>::nFilterAxes; j++) {
            shape[j] = 1;
        }

        shape[i] = 2*radius+1;

        CoeffType coefficients(shape);

        std::array<int, Filter<T, Ds...>::nFilterAxes> idx;

        for (int j = 0; j < Filter<T, Ds...>::nFilterAxes; j++) {
            idx[j] = 0;
        }

        double sum = 0;

        for (int d = 0; d < 2*radius+1; d++) {
            idx[i] = d;

            int delta = d-radius;
            double coeff = std::exp(-delta*delta/var);

            coefficients.atUnchecked(idx) = coeff;

            sum += coeff;
        }

        if (normalize) {

            for (int d = 0; d < 2*radius+1; d++) {
                idx[i] = d;

                coefficients.atUnchecked(idx) /= sum;
            }

        }

        if (i == 0) {
            ret[i] = Internals::instanciateSeparatedFilter<T, true>(coefficients, i, std::index_sequence_for<Ds...>(), axisDefinitions...);
        } else {
            ret[i] = Internals::instanciateSeparatedFilter<T, false>(coefficients, i, std::index_sequence_for<Ds...>(), axisDefinitions...);
        }


    }

    return ret;

}

template<typename T, typename... Ds>
std::array<Filter<T, Ds...>, Filter<T, Ds...>::nAxesOfType(AxisType::Moving)> finiteDifferencesKernels(Ds... axisDefinitions) {


    using CoeffType = Multidim::Array<T, Filter<T, Ds...>::nFilterAxes>;

    constexpr int nFilters = Filter<T, Ds...>::nAxesOfType(AxisType::Moving);

    std::array<Filter<T, Ds...>, nFilters> ret;

    for (int i = 0; i < nFilters; i++) {

        std::array<int, Filter<T, Ds...>::nFilterAxes> shape;

        for (int j = 0; j < Filter<T, Ds...>::nFilterAxes; j++) {
            shape[j] = 1;
        }

        shape[i] = 3;

        CoeffType coefficients(shape);

        std::array<int, Filter<T, Ds...>::nFilterAxes> idx;

        for (int j = 0; j < Filter<T, Ds...>::nFilterAxes; j++) {
            idx[j] = 0;
        }

        idx[i] = 0;
        coefficients.atUnchecked(idx) = -1;

        idx[i] = 1;
        coefficients.atUnchecked(idx) = 0;

        idx[i] = 2;
        coefficients.atUnchecked(idx) = 1;

        if (i == 0) {
            ret[i] = Internals::instanciateSeparatedFilter<T, true>(coefficients, i, std::index_sequence_for<Ds...>(), axisDefinitions...);
        } else {
            ret[i] = Internals::instanciateSeparatedFilter<T, false>(coefficients, i, std::index_sequence_for<Ds...>(), axisDefinitions...);
        }
    }

    return ret;
}

template<typename T, typename... Ds>
std::array<Filter<T, Ds...>, Filter<T, Ds...>::nAxesOfType(AxisType::Moving)> extendLinearKernels(Ds... axisDefinitions) {


    using CoeffType = Multidim::Array<T, Filter<T, Ds...>::nFilterAxes>;

    constexpr int nFilters = Filter<T, Ds...>::nAxesOfType(AxisType::Moving);

    std::array<Filter<T, Ds...>, nFilters> ret;

    for (int i = 0; i < nFilters; i++) {

        std::array<int, Filter<T, Ds...>::nFilterAxes> shape;

        for (int j = 0; j < Filter<T, Ds...>::nFilterAxes; j++) {
            shape[j] = 1;
        }

        shape[i] = 3;

        CoeffType coefficients(shape);

        std::array<int, Filter<T, Ds...>::nFilterAxes> idx;

        for (int j = 0; j < Filter<T, Ds...>::nFilterAxes; j++) {
            idx[j] = 0;
        }

        idx[i] = 0;
        coefficients.atUnchecked(idx) = 1;

        idx[i] = 1;
        coefficients.atUnchecked(idx) = 2;

        idx[i] = 2;
        coefficients.atUnchecked(idx) = 1;

        if (i == 0) {
            ret[i] = Internals::instanciateSeparatedFilter<T, true>(coefficients, i, std::index_sequence_for<Ds...>(), axisDefinitions...);
        } else {
            ret[i] = Internals::instanciateSeparatedFilter<T, false>(coefficients, i, std::index_sequence_for<Ds...>(), axisDefinitions...);
        }
    }

    return ret;
}

} //namespace Convolution
} //namespace ImageProcessing
} //namespace StereoVision

#endif // STANDARDCONVOLUTIONFILTERS_H
