#ifndef POINTSDESCRIPTORS_H
#define POINTSDESCRIPTORS_H

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

#include <random>

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include "./pointsOrientation.h"

#include "../interpolation/interpolation.h"

namespace StereoVision {
namespace SparseMatching {

template <int nDim, typename FT>
struct pointFeatures {

        pointFeatures()
        {

        }

        pointFeatures(std::array<int, nDim> const& pos,
                      FT const& features) :
            coord(pos),
            features(features)
        {

        }

        std::array<int, nDim> coord;
        FT features;
};

template<int nDim>
using ComparisonPair = std::array<std::array<float, nDim>,2>;

/*!
 * \brief generateRandomComparisonPairs generate random comparison pairs for descriptors
 * \param nSamples the number of comparison to generate.
 * \param windowsRadius the size of the window to consider
 * \return randomly generated pairs
 *
 * The pairs are generated indepandently from an isotropic gaussian distribution, as was found to be optimal in
 * Calonder, Michael, et al. "Brief: Binary robust independent elementary features." Computer Vision–ECCV 2010:
 * 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings,
 * Part IV 11. Springer Berlin Heidelberg, 2010.
 */
template<int nDim>
std::vector<ComparisonPair<nDim>> generateRandomComparisonPairs(int nSamples, int windowsRadius) {

    int windowsWidth = windowsRadius;
    float std = static_cast<float>(windowsWidth)/5;

    std::default_random_engine re;
    re.seed(std::random_device{}());
    std::normal_distribution<float> distribution(0.0,std);

    std::vector<ComparisonPair<nDim>> ret(nSamples);

    for (int i = 0; i < nSamples; i++) {
        for (int j = 0; j < nDim; j++) {
            ret[i][0][j] = distribution(re);
            ret[i][1][j] = distribution(re);

            ret[i][0][j] = std::clamp<float>(ret[i][0][j], -windowsRadius, windowsRadius);
            ret[i][1][j] = std::clamp<float>(ret[i][1][j], -windowsRadius, windowsRadius);
        }
    }

    return ret;

}

/*!
 * \brief generateRandomComparisonPairs generate random comparison pairs for descriptors for a multi channel image.
 * \param nSamples the number of samples
 * \param windowsRadius the radius of the search window
 * \param nChannels the number of channels in the image
 * \return comparison pairs
 */
template<int nDim>
std::vector<ComparisonPair<nDim+1>> generateRandomComparisonPairs(int nSamples, int windowsRadius, int nChannels) {

    int windowsWidth = windowsRadius;
    float std = static_cast<float>(windowsWidth)/5;

    std::default_random_engine re;
    re.seed(std::random_device{}());
    std::normal_distribution<float> distribution(0.0,std);

    std::vector<ComparisonPair<nDim+1>> ret(nSamples);

    int channel = 0;

    for (int i = 0; i < nSamples; i++) {
        for (int j = 0; j < nDim; j++) {
            ret[i][0][j] = distribution(re);
            ret[i][1][j] = distribution(re);

            ret[i][0][j] = std::clamp<float>(ret[i][0][j], -windowsRadius, windowsRadius);
            ret[i][1][j] = std::clamp<float>(ret[i][1][j], -windowsRadius, windowsRadius);
        }

        ret[i][0][nDim] = channel;
        ret[i][1][nDim] = channel;

        channel++;
        channel %= nChannels;

    }

    return ret;

}

/*!
 * \brief generateDensePatchCoordinates generate a set of dense coordinates delta for
 * \param shape
 * \return
 */
template<int nDim>
std::vector<std::array<int,nDim>> generateDensePatchCoordinates(std::array<int,nDim> const& shape) {

    int nElements = 1;
    std::array<int,nDim> delta;

    for (int i = 0; i < nDim; i++) {
        nElements *= shape[i];
        delta[i] = shape[i]/2;
    }

    std::vector<std::array<int,nDim>> ret(nElements);

    Multidim::IndexConverter<nDim> idxCvrt(shape);

    for (int i = 0; i < idxCvrt.numberOfPossibleIndices(); i++) {
        std::array<int,nDim> idx = idxCvrt.getIndexFromPseudoFlatId(i);

        for (int j = 0; j < nDim; j++) {
            idx[j] -= delta[j];
        }

        ret[i] = idx;
    }

    return ret;
}

template <bool hasFeatureAxis = true, int nDim, typename T, Multidim::ArrayDataAccessConstness constNess>
std::vector<pointFeatures<nDim, std::vector<uint32_t>>> BriefDescriptor(std::vector<orientedCoordinate<nDim>> const& coords,
                                                                        Multidim::Array<T, (hasFeatureAxis) ? nDim+1 : nDim, constNess> const& img,
                                                                        std::vector<ComparisonPair<(hasFeatureAxis) ? nDim+1 : nDim>> const& comparisonPairs,
                                                                        int featureAxis = nDim) {

    static_assert (nDim == 2, "only nDim == 2 is supported at the moment"); //TODO: change the oriented coordinate class to support at least nDim == 3 as well.

    constexpr int imDim = (hasFeatureAxis) ? nDim+1 : nDim;

    int nPairs = comparisonPairs.size();

    constexpr int nbits = 32;
    int nDWord = nPairs/nbits;
    if (nPairs % nbits != 0) {
        nDWord += 1;
    }

    std::vector<pointFeatures<nDim, std::vector<uint32_t>>> ret;
    ret.reserve(coords.size());

    for (orientedCoordinate<nDim> const& coord : coords) {

        std::vector<uint32_t> feature(nDWord);
        std::fill(feature.begin(), feature.end(), 0);

        int bId = 0;
        int fId = 0;

        float theta = std::atan2(coord.main_dir[0], coord.main_dir[1]);

        float cos = std::cos(theta);
        float sin = std::sin(theta);

        for (ComparisonPair<imDim> const& pair : comparisonPairs) {

            std::array<float, imDim> transformed_coords0;
            std::array<float, imDim> transformed_coords1;

            int coord0_id = (hasFeatureAxis and (featureAxis == 0)) ? 1 : 0;
            int coord1_id = (hasFeatureAxis and (featureAxis >= 1)) ? 2 : 1;

            transformed_coords0[coord0_id] = cos*pair[0][coord0_id] - sin*pair[0][coord1_id];
            transformed_coords1[coord0_id] = cos*pair[1][coord0_id] - sin*pair[1][coord1_id];

            transformed_coords0[coord1_id] = sin*pair[0][coord0_id] + cos*pair[0][coord1_id];
            transformed_coords1[coord1_id] = sin*pair[1][coord0_id] + cos*pair[1][coord1_id];

            if (hasFeatureAxis) {
                transformed_coords0[featureAxis] = pair[0][featureAxis];
                transformed_coords1[featureAxis] = pair[1][featureAxis];
            }

            T val0 = Interpolation::interpolateValue<imDim, T, Interpolation::pyramidFunction<T, imDim>, 0>(img, transformed_coords0);
            T val1 = Interpolation::interpolateValue<imDim, T, Interpolation::pyramidFunction<T, imDim>, 0>(img, transformed_coords1);

            uint32_t m = (val0 > val1) ? 1 : 0; //compute bitmask

            m <<= bId; //shift bit to id

            feature[fId] |= m; //apply bitmask

            bId++;

            if (bId >= nbits) {
                bId = 0;
                fId++;
            }

        }

        ret.emplace_back(coord.coord, feature);

    }

    return ret;

}

template <int nDim, int fDims, typename T, Multidim::ArrayDataAccessConstness constNess>
std::vector<pointFeatures<nDim, std::vector<float>>> WhitenedPixelsDescriptor(std::vector<orientedCoordinate<nDim>> const& coords,
                                                                              Multidim::Array<T, fDims, constNess> const& img,
                                                                              std::vector<std::array<int,std::size_t(fDims)>> const& patchCoordinates,
                                                                              int featureAxis = nDim) {

    static_assert (fDims == nDim or fDims == nDim+1, "image dimension should be equal (grayscale) or one more (color) than coordinates dimensions");

    constexpr bool hasFeatureAxis = fDims == nDim+1;

    std::vector<pointFeatures<nDim, std::vector<float>>> ret;
    ret.reserve(coords.size());

    for (orientedCoordinate<nDim> const& coord : coords) {

        std::vector<float> feature(patchCoordinates.size());
        int f = 0;

        std::array<int, fDims> center;

        for (int i = 0; i < fDims; i++) {

            if (hasFeatureAxis and i == featureAxis) {
                center[i] = 0;
                break;
            }

            int idx = (i < featureAxis or !hasFeatureAxis) ? coord.coord[i] : coord.coord[i-1];
            center[i] = idx;
        }

        std::array<int, fDims> idxs;

        for (std::array<int,fDims> const& pCoord : patchCoordinates) {


            for (int i = 0; i < fDims; i++) {
                idxs[i] = center[i] + pCoord[i];
            }

            feature[f] = img.valueOrAlt(idxs, 0);
            f++;
        }

        //whitening of the features
        float mean = 0;

        for (float const& val : feature) {
            mean += val;
        }

        mean /= feature.size();

        for (float & val : feature) {
            val -= mean;
        }

        float norm = 0;

        for (float const& val : feature) {
            norm += val*val;
        }

        norm /= feature.size();
        norm = std::sqrt(norm);

        for (float & val : feature) {
            val /= norm;
        }

        ret.emplace_back(coord.coord, feature);

    }

    return ret;

}

} // namespace SparseMatching
} // namespace StereoVision

#endif // POINTSDESCRIPTORS_H
