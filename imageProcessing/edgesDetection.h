#ifndef LIBSTEVI_EDGEDETECTIONS_H
#define LIBSTEVI_EDGEDETECTIONS_H

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

#include <array>
#include <vector>

#include <MultidimArrays/MultidimArrays.h>

#include "./convolutions.h"

namespace StereoVision {
namespace ImageProcessing {

/*!
 * \brief gradientBasedEdges a very simple edge extractor which detect pixels with a large gradient
 * \param input the input image
 * \param propEdges the proportion of pixels to classify as edges
 * \return a list of image coordinates with the pixels with a large gradient, as well as the associated gradient.
 */
template<typename ComputeType>
std::vector<std::tuple<std::array<int,2>, std::array<ComputeType,2>>> gradientBasedEdges(Multidim::Array<ComputeType, 3> const& input, ComputeType propEdges = 0.05) {

    std::array<int,3> inSize = input.shape();

    constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

    constexpr int nGradientDir = 2;
    Multidim::Array<ComputeType,3> coefficients(3,3,nGradientDir);

    coefficients.atUnchecked(0,0,0) = 1;
    coefficients.atUnchecked(1,0,0) = 2;
    coefficients.atUnchecked(2,0,0) = 1;

    coefficients.atUnchecked(0,1,0) = 0;
    coefficients.atUnchecked(1,1,0) = 0;
    coefficients.atUnchecked(2,1,0) = 0;

    coefficients.atUnchecked(0,2,0) = -1;
    coefficients.atUnchecked(1,2,0) = -2;
    coefficients.atUnchecked(2,2,0) = -1;

    coefficients.atUnchecked(0,0,1) = 1;
    coefficients.atUnchecked(0,1,1) = 2;
    coefficients.atUnchecked(0,2,1) = 1;

    coefficients.atUnchecked(1,0,1) = 0;
    coefficients.atUnchecked(1,1,1) = 0;
    coefficients.atUnchecked(1,2,1) = 0;

    coefficients.atUnchecked(2,0,1) = -1;
    coefficients.atUnchecked(2,1,1) = -2;
    coefficients.atUnchecked(2,2,1) = -1;

    using Maxis = Convolution::MovingWindowAxis;
    using BIaxis = Convolution::BatchedInputAxis;
    using BOaxis = Convolution::BatchedOutputAxis;

    Convolution::Filter<ComputeType, Maxis, Maxis, BIaxis, BOaxis> gradientFilter(coefficients, Maxis(), Maxis(), BIaxis(), BOaxis());

    Multidim::Array<ComputeType, 4> gradientsPerChannels = gradientFilter.convolve(input);

    Multidim::Array<ComputeType, 3> aggregatedGradients(inSize[0], inSize[1], nGradientDir);
    Multidim::Array<ComputeType, 2> gradientsAmplitude(inSize[0], inSize[1]);

    std::vector<ComputeType> sortedGradientsAmpl;
    sortedGradientsAmpl.reserve(inSize[0]*inSize[1]);

    for (int i = 0; i < inSize[0]; i++) {
        for (int j = 0; j < inSize[1]; j++) {

            ComputeType d0 = 0;
            ComputeType d1 = 0;

            for (int c = 0; c < 2; c++) {

                ComputeType cd0 = gradientsPerChannels.atUnchecked(i,j,c,0);
                ComputeType cd1 = gradientsPerChannels.atUnchecked(i,j,c,1);

                ComputeType coeff = 1;

                //turn 180° if necessary, so that orientation matter, but direction do not.
                if (d0*cd0 + d1*cd1 < 0) {
                    coeff = -1;
                }

                d0 += coeff*cd0;
                d1 += coeff*cd1;
            }

            aggregatedGradients.atUnchecked(i,j,0) = d0;
            aggregatedGradients.atUnchecked(i,j,1) = d1;

            ComputeType ampl = d0*d0 + d1*d1;

            gradientsAmplitude.atUnchecked(i,j) = ampl;
            sortedGradientsAmpl.push_back(ampl);

        }
    }

    int qant = (sortedGradientsAmpl.size()-1)*(1.-propEdges);

    if (qant < 0) {
        qant = 0;
    }

    if (qant >= sortedGradientsAmpl.size()) {
        qant = sortedGradientsAmpl.size()-1;
    }

    std::nth_element(sortedGradientsAmpl.begin(), sortedGradientsAmpl.begin()+qant, sortedGradientsAmpl.end());

    ComputeType threshold = sortedGradientsAmpl[qant];
    int nElements = sortedGradientsAmpl.size() - qant;

    std::vector<std::tuple<std::array<int,2>, std::array<ComputeType,2>>> ret;
    ret.reserve(nElements);


    for (int i = 0; i < inSize[0]; i++) {
        for (int j = 0; j < inSize[1]; j++) {

            if (gradientsAmplitude.atUnchecked(i,j) >= threshold) {

                ComputeType d0 = aggregatedGradients.atUnchecked(i,j,0);
                ComputeType d1 = aggregatedGradients.atUnchecked(i,j,1);

                ret.push_back(std::make_tuple(std::array<int,2>{i,j}, std::array<ComputeType,2>{d0, d1}));
            }
        }
    }

    return ret;

}

} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_EDGEDETECTIONS_H
