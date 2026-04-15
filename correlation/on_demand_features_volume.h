#ifndef ON_DEMAND_FEATURES_VOLUME_H
#define ON_DEMAND_FEATURES_VOLUME_H

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
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <vector>
#include <Eigen/Core>

#include "./feature_volume_infos.h"

namespace StereoVision {
namespace Correlation {

template<class T_F, int nD, Multidim::ArrayDataAccessConstness constness,
         int... FeaturesDims>
class OnDemandFeaturesVolume {
    static inline constexpr bool checkFeaturesDims() {
        std::array<int, sizeof...(FeaturesDims)> dims = {FeaturesDims...};

        for (int i = 0; i < dims.size(); i++) {
            if (dims[i] < 0 or dims[i] >= nD) {
                return false;
            }
            for (int j = i+1; j < dims.size(); j++) {
                if (dims[i] == dims[j]) {
                    return false;
                }
            }
        }
        return true;
    }
public:

    static_assert(sizeof...(FeaturesDims) < nD);

    static_assert(checkFeaturesDims(),
                  "Features dimensions should be greather than 0 and smaller than the number of dimensions, all of them should be different!");

    using ArrayType = Multidim::Array<T_F, nD, constness>;

    static constexpr int nInDim = nD;
    typedef T_F ScalarT;
    static constexpr int nFeaturesDim = sizeof...(FeaturesDims);
    static constexpr int nOutDim = nInDim - nFeaturesDim;

    using InIndex = std::array<int,nInDim>;
    using OutIndex = std::array<int,nOutDim>;

    static OutIndex outIdxCorrespondingInAxis() {
        std::array<int, sizeof...(FeaturesDims)> dims = {FeaturesDims...};
        OutIndex ret;
        int rId = 0;
        for (int i = 0; i < nInDim; i++) {
            for (int j = 0; j < dims.size(); j++) {
                if (i == dims[j]) {
                    continue;
                }
            }
            ret[rId] = i;
            rId++;
        }
        return ret;
    }
    static constexpr int inDimIsFeature(int dimId) {
        std::array<int, sizeof...(FeaturesDims)> dims = {FeaturesDims...};
        for (int j = 0; j < dims.size(); j++) {
            if (dimId == dims[j]) {
                return true;;
            }
        }
        return false;
    }

    OnDemandFeaturesVolume(std::vector<InIndex> const& window,
                           ArrayType const& array) :
        _window(window),
        _array(array)
    {

    }

    /*!
     * \brief shape the shape the actual corresponding feature volume would have (assuming feature axis is the last dim)
     * \return the shape
     */
    std::array<int, nOutDim+1> shape() const {
        std::array<int, nOutDim+1> ret;
        OutIndex outCorrespIdxs = outIdxCorrespondingInAxis();
        for (int i = 0; i < outCorrespIdxs.size(); i++) {
            ret[i] = _array.shape()[outCorrespIdxs[i]];
        }
        ret.back() = _window.size();
        return ret;
    }

    Multidim::Array<T_F, 1> getFeatureVec (OutIndex const& idx) const {
        Multidim::Array<T_F, 1> ret(_window.size());

        InIndex in_idx_base;
        std::fill(in_idx_base.begin(), in_idx_base.end(), 0);
        OutIndex outCorrespIdxs = outIdxCorrespondingInAxis();

        for (int i = 0; i < outCorrespIdxs.size(); i++) {
            in_idx_base[outCorrespIdxs[i]] = idx[i];
        }

        for (int f = 0; f < _window.size(); f++) {
            InIndex idx;
            for (int i = 0; i < idx.size(); i++) {
                idx[i] = std::min(_array.shape()[i]-1,std::max(0,in_idx_base[i]+_window[f][i])); //constant border condition
                //TODO: define more border conditions
            }

            ret.atUnchecked(f) = _array.valueUnchecked(idx);
        }

        return ret;
    }

    Eigen::Matrix<T_F, Eigen::Dynamic, 1> getFeatureVecEigen (OutIndex const& idx) const {
        Eigen::Matrix<T_F, Eigen::Dynamic, 1> ret(_window.size());

        InIndex in_idx_base;
        std::fill(in_idx_base.begin(), in_idx_base.end(), 0);
        OutIndex outCorrespIdxs = outIdxCorrespondingInAxis();

        for (int i = 0; i < outCorrespIdxs.size(); i++) {
            in_idx_base[outCorrespIdxs[i]] = idx[i];
        }

        for (int f = 0; f < _window.size(); f++) {
            InIndex idx = in_idx_base;
            for (int i = 0; i < idx.size(); i++) {
                idx[i] += std::min(_array.shape()[i]-1,std::max(0,_window[f][i])); //constant border condition
                //TODO: define more border conditions
            }

            ret[f] = _array.valueUnchecked(idx);
        }

        return ret;
    }

protected:
    std::vector<InIndex> _window;
    ArrayType const& _array;
};

template<bool ZeroMean, bool Normalized>
struct ZNFeaturesVolumeDecorator {
public:
    template<typename T, Multidim::ArrayDataAccessConstness constness>
    static Multidim::Array<T, 1, constness> processFeature(Multidim::Array<T, 1, constness> && f) {

        if (!ZeroMean and !Normalized) {
            return Multidim::Array<T, 1, constness>(f); //try to use move constructor if possible
        }

        int nF = f.shape()[0];

        Multidim::Array<T, 1> ret(nF);

        for (int i = 0; i < nF; i++) {
            ret.atUnchecked(i) = f.valueUnchecked(i);
        }

        if constexpr (ZeroMean) {
            T mean = 0;
            for (int i = 0; i < nF; i++) {
                mean += ret.valueUnchecked(i);
            }
            mean /= nF;
            for (int i = 0; i < nF; i++) {
                ret.atUnchecked(i) -= mean;
            }
        }

        if constexpr (Normalized) {
            T norm = 0;
            for (int i = 0; i < nF; i++) {
                T v = ret.valueUnchecked(i);
                norm += v*v;
            }

            norm /= nF;
            norm = sqrt(norm);

            for (int i = 0; i < nF; i++) {
                ret.atUnchecked(i) /= norm;
            }
        }

        return ret;
    }
};

template<class Decorator, class T_F, int nD, Multidim::ArrayDataAccessConstness constness,
         int... FeaturesDims>
class OnDemandDecoratedFeaturesVolume : public OnDemandFeaturesVolume<T_F, nD, constness, FeaturesDims...> {

    using ParentT = OnDemandFeaturesVolume<T_F, nD, constness, FeaturesDims...>;

public:

    OnDemandDecoratedFeaturesVolume(std::vector<typename ParentT::InIndex> const& window,
                                    typename ParentT::ArrayType const& array) :
        ParentT(window, array)
    {

    }

    Multidim::Array<T_F, 1> getFeatureVec (typename ParentT::OutIndex const& idx) const {
        return Decorator::processFeature(ParentT::getFeatureVec(idx));
    }

    Eigen::Matrix<T_F, Eigen::Dynamic, 1> getFeatureVecEigen (typename ParentT::OutIndex const& idx) const {
        Multidim::Array<T_F, 1> r = getFeatureVec (idx);
        Eigen::Matrix<T_F, Eigen::Dynamic, 1> ret(r.shape()[0]);

        for (int i = 0; i < r.shape()[0]; i++) {
            ret[i] = r.valueUnchecked(i);
        }

        return ret;
    }

protected:
};


template<typename T, int nDim, Multidim::ArrayDataAccessConstness constness, int... FeaturesDims>
struct FeatureVolumeInfos<OnDemandFeaturesVolume<T,nDim, constness, FeaturesDims...>> {
    static constexpr int NDims = OnDemandFeaturesVolume<T,nDim, constness, FeaturesDims...>::nOutDim+1;
    static constexpr int FeatureDim = NDims-1;
    using FeatureScalarT = T;
    static constexpr bool isArray = false;

    inline static std::array<int, NDims> shape(OnDemandFeaturesVolume<T,nDim, constness, FeaturesDims...> const& F_V) {
        return F_V.shape();
    }

    inline static Multidim::Array<FeatureScalarT, 1> getFeatureVec (OnDemandFeaturesVolume<T,nDim, constness, FeaturesDims...> const& F_V,
                                                     std::array<int,NDims-1> const& idx) {
        return F_V.getFeatureVec(idx);
    }

    Eigen::Matrix<FeatureScalarT, Eigen::Dynamic, 1> getFeatureVecEigen (OnDemandFeaturesVolume<T,nDim, constness, FeaturesDims...> const& F_V,
                                                                        std::array<int,NDims-1> const& idx) {
        return F_V.getFeatureVecEigen(idx);
    }
};

template<typename T, class Decorator, int nDim, Multidim::ArrayDataAccessConstness constness, int... FeaturesDims>
struct FeatureVolumeInfos<OnDemandDecoratedFeaturesVolume<Decorator, T,nDim, constness, FeaturesDims...>> {
    static constexpr int NDims = OnDemandDecoratedFeaturesVolume<Decorator, T,nDim, constness, FeaturesDims...>::nOutDim+1;
    static constexpr int FeatureDim = NDims-1;
    using FeatureScalarT = T;
    static constexpr bool isArray = false;

    inline static std::array<int, NDims> shape(OnDemandDecoratedFeaturesVolume<Decorator, T,nDim, constness, FeaturesDims...> const& F_V) {
        return F_V.shape();
    }

    inline static Multidim::Array<FeatureScalarT, 1> getFeatureVec (OnDemandDecoratedFeaturesVolume<Decorator, T,nDim, constness, FeaturesDims...> const& F_V,
                                                                   std::array<int,NDims-1> const& idx) {
        return F_V.getFeatureVec(idx);
    }

    Eigen::Matrix<FeatureScalarT, Eigen::Dynamic, 1> getFeatureVecEigen (OnDemandDecoratedFeaturesVolume<Decorator, T,nDim, constness, FeaturesDims...> const& F_V,
                                                                        std::array<int,NDims-1> const& idx) {
        return F_V.getFeatureVecEigen(idx);
    }
};

} //namespace Correlation
} //namespace StereoVision

#endif // ON_DEMAND_FEATURES_VOLUME_H
