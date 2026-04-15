#ifndef FEATURE_VOLUME_INFOS_H
#define FEATURE_VOLUME_INFOS_H
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2026  Paragon<french.paragon@gmail.com>

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

#include <MultidimArrays/MultidimArrays.h>
#include <Eigen/Core>

namespace StereoVision {
namespace Correlation {

/*!
 * \brief The FeatureVolumeInfos class provide informations about a feature volume type, provided as template parameter
 *
 * The basic implementation is empty, the struct will be re-implemented with the different types that can be used as feature volume
 */
template<typename T>
struct FeatureVolumeInfos {

};

template<typename T, int nDim, Multidim::ArrayDataAccessConstness constness>
struct FeatureVolumeInfos<Multidim::Array<T,nDim,constness>> {
    static constexpr int NDims = nDim;
    static constexpr int FeatureDim = nDim-1;
    using FeatureScalarT = T;
    static constexpr bool isArray = true;

    inline static std::array<int, NDims> shape(Multidim::Array<T,nDim,constness> const& F_V) {
        return F_V.shape();
    }

    inline static Multidim::Array<FeatureScalarT, 1> getFeatureVec (Multidim::Array<T,nDim,constness> const& F_V,
                                                                   std::array<int,NDims-1> const& idx) {
        Multidim::Array<FeatureScalarT, 1> ret(F_V.shape()[FeatureDim]);

        std::array<int,NDims> i;
        for (int j = 0; j < NDims-1; j++) {
            i[j] = idx[j];
        }

        for (int c = 0; c < F_V.shape()[FeatureDim]; c++) {
            i[NDims-1] = c;
            ret.atUnchecked(c) = F_V.valueUnchecked(i);
        }
        return ret;
    }

    Eigen::Matrix<FeatureScalarT, Eigen::Dynamic, 1> getFeatureVecEigen (Multidim::Array<T,nDim,constness> const& F_V,
                                                                        std::array<int,NDims-1> const& idx) {
        Eigen::Matrix<FeatureScalarT, Eigen::Dynamic, 1> ret(F_V.shape()[FeatureDim]);

        std::array<int,NDims> i;
        for (int j = 0; j < NDims-1; j++) {
            i[j] = idx[j];
        }

        for (int c = 0; c < F_V.shape()[FeatureDim]; c++) {
            i[NDims-1] = c;
            ret[c] = F_V.valueUnchecked(i);
        }
        return ret;
    }
};

} // namespace StereoVision
} // namespace Correlation
#endif // FEATURE_VOLUME_INFOS_H
