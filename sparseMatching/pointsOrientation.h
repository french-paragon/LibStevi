#ifndef POINTSORIENTATION_H
#define POINTSORIENTATION_H

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

#include <vector>
#include <array>

#include <cmath>

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

namespace StereoVision {
namespace SparseMatching {

template <int nDim>
struct orientedCoordinate {

        orientedCoordinate()
        {

        }

        orientedCoordinate(std::array<int, nDim> const& pos,
                           std::array<float, nDim> const& mainDir) :
            coord(pos),
            main_dir(mainDir)
        {

        }

        inline int& operator[](int idx) {
            return coord[idx];
        }

        std::array<int, nDim> coord;
        std::array<float, nDim> main_dir;
};

template<bool hasFeatureAxis = true,typename T, size_t nDim, Multidim::ArrayDataAccessConstness constNess>
std::vector<orientedCoordinate<nDim>> intensityOrientedCoordinates(std::vector<std::array<int, nDim>> coords,
                                                                   Multidim::Array<T, (hasFeatureAxis) ? nDim+1 : nDim, constNess> const& img,
                                                                   int searchRadius = 3,
                                                                   int featureAxis = nDim) {

    constexpr int imDim = (hasFeatureAxis) ? nDim+1 : nDim;

    std::vector<orientedCoordinate<nDim>> ret;
    ret.reserve(coords.size());

    for (std::array<int, nDim> const& coord : coords) {

        std::array<int, imDim> windowSize;
        std::array<int, imDim> windowOffset;
        std::array<int, imDim> initialCoord;

        for (int i = 0; i < imDim; i++) {

            if (i == featureAxis) {

                windowSize[i] = img.shape()[i];
                windowOffset[i] = 0;
                initialCoord[i] = 0;

            } else {
                windowSize[i] = 2*searchRadius+1;
                windowOffset[i] = -searchRadius;

                int cI = (i > featureAxis) ? i-1 : i;

                initialCoord[i] = coord[cI];
            }

        }

        Multidim::IndexConverter<imDim> converter(windowSize);

        std::array<float, nDim> weigthedSum;

        for (int i = 0; i < nDim; i++) {
            weigthedSum[i] = 0;
        }

        for (int i = 0; i < converter.numberOfPossibleIndices(); i++) {

            std::array<int, imDim> idx = converter.getIndexFromPseudoFlatId(i);
            std::array<int, nDim> s_idx;

            int s = 0;

            for (int i = 0; i < imDim; i++) {

                if (i != featureAxis) {
                    s_idx[s] = idx[i] + windowOffset[i];
                    s++;
                }

                idx[i] += initialCoord[i] + windowOffset[i];
            }

            T weigth = img.valueOrAlt(idx, 0);

            for (int i = 0; i < nDim; i++) {
                weigthedSum[i] += weigth*s_idx[i];
            }

        }

        double norm = 0;

        for (int i = 0; i < nDim; i++) {
            norm += weigthedSum[i]*weigthedSum[i];
        }

        norm = std::sqrt(norm);

        for (int i = 0; i < nDim; i++) {
            weigthedSum[i] /= norm;
        }

        ret.emplace_back(coord, weigthedSum);

    }

    return ret;

}

} // namespace SparseMatching
} // namespace StereoVision

#endif // POINTSORIENTATION_H
