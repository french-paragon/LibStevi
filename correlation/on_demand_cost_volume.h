#ifndef STEREOVISION_ON_DEMAND_COST_VOLUME_H
#define STEREOVISION_ON_DEMAND_COST_VOLUME_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022  Paragon<french.paragon@gmail.com>

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

#include "./correlation_base.h"
#include "./matching_costs.h"
#include "./cross_correlations.h"

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

namespace StereoVision {
namespace Correlation {

template<matchingFunctions matchFunc, class T_CV, class T_S, class T_T,
		 Multidim::ArrayDataAccessConstness constnessS,
		 Multidim::ArrayDataAccessConstness constnessT,
		 typename... Ds>
class GenericOnDemandCostVolume {

public:
	using SearchSpaceType = FixedSearchSpace<Ds...>;

	using T_SF = typename MatchingFuncComputeTypeInfos<matchFunc, T_S>::FeatureType;
	using T_TF = typename MatchingFuncComputeTypeInfos<matchFunc, T_T>::FeatureType;

	static constexpr int nDim = SearchSpaceType::nDim;
	static constexpr int nSearchDim = SearchSpaceType::nDimsOfType(SearchSpaceBase::Search);
	static constexpr int featureDim = SearchSpaceType::featuresDim();
	static constexpr int nCostVolDim = nDim + nSearchDim - 1;

	static_assert (featureDim >= 0 and featureDim < nDim, "Invalid search space");

	explicit GenericOnDemandCostVolume(Multidim::Array<T_S, nDim, constnessS> const& source,
									   Multidim::Array<T_T, nDim, constnessT> const& target,
									   SearchSpaceType const& searchSpace) :
		_source(source),
		_target(target),
		_search_space(searchSpace)
	{

		if (MatchingFunctionTraits<matchFunc>::Normalized or MatchingFunctionTraits<matchFunc>::ZeroMean) {

			_source_processed = getFeatureVolumeForMatchFunc<matchFunc, T_S, Multidim::NonConstView, T_SF>(source);
			_target_processed = getFeatureVolumeForMatchFunc<matchFunc, T_T, Multidim::NonConstView, T_TF>(target);

		}

		std::array<Multidim::array_size_t, nCostVolDim> cvShape{};

		for (int i = 0, s = 0; i < nDim; i++) {
			int d = (i >= featureDim) ? i-1 : i;

			if (_search_space.getDimType(i) == SearchSpaceBase::Search) {

				_searchDims[s] = i;
				s++;
			}

			if (_search_space.getDimType(i) != SearchSpaceBase::Feature) {
				cvShape[d] = source.shape()[i];
			}

		}

		for (int i = 0; i < nSearchDim; i++) {

			int range = _search_space.dimRange(_searchDims[i]);

			cvShape[nDim-1+i] = range;
		}

		_cost_volume = Multidim::Array<T_CV, nCostVolDim>(cvShape);
		_computed = Multidim::Array<bool, nCostVolDim>(cvShape);

		int l = _computed.flatLenght();
		std::fill(&_computed.atUnchecked(0), &_computed.atUnchecked(0)+l, false);
	}

	inline auto shape() const {
		return _cost_volume.shape();
	}

	inline std::optional<T_CV> costValue(std::array<int, nDim-1> pos, std::array<int, nSearchDim> disp) const {

		constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

		std::array<int, nDim-1> sourcePos = pos;
		std::array<int, nDim-1> targetPos = pos;

		std::array<Multidim::array_size_t, nCostVolDim> cvIndex{};

		for(int i = 0; i < nDim-1; i++) {
			cvIndex[i] = pos[i];
		}

		for (int i = 0; i < nSearchDim; i++) {

			cvIndex[nDim+i-1] = _search_space.disp2idx(_searchDims[i], disp[i]);
		}

		for (int i = 0; i < nCostVolDim; i++) {

			if (cvIndex[i] < 0) {
				return std::nullopt;
			}

			if (cvIndex[i] >= _cost_volume.shape()[i]) {
				return std::nullopt;
			}
		}

		if (_computed.template at<Nc>(cvIndex)) {
			return _cost_volume.template at<Nc>(cvIndex);
		}

		for (int i = 0, s = 0; i < nDim; i++) {

			int d = (i >= featureDim) ? i-1 : i;

			if (_search_space.getDimType(i) == SearchSpaceBase::Search) {

				if (disp[s] < _search_space.getDimMinSearchRange(i)) {
					return std::nullopt;
				}

				if (disp[s] > _search_space.getDimMaxSearchRange(i)) {
					return std::nullopt;
				}

				targetPos[d] += disp[s];

				if (targetPos[d] < 0 or targetPos[d] >= _target.shape()[i]) {
					return std::nullopt;
				}
				s++;
			}
		}

		T_CV cost;
		if (MatchingFunctionTraits<matchFunc>::Normalized or MatchingFunctionTraits<matchFunc>::ZeroMean) {

			Multidim::Array<T_SF,1,Multidim::ConstView> source_features = _source_processed.indexDimView(featureDim, sourcePos);
			Multidim::Array<T_TF,1,Multidim::ConstView> target_features = _target_processed.indexDimView(featureDim, targetPos);

			cost = MatchingFunctionTraits<matchFunc>::template featureComparison<T_SF, T_TF, T_CV>(source_features, target_features);

		} else {

			Multidim::Array<T_S,1,Multidim::ConstView> source_features = _source.indexDimView(featureDim, sourcePos);
			Multidim::Array<T_T,1,Multidim::ConstView> target_features = _target.indexDimView(featureDim, targetPos);

			cost = MatchingFunctionTraits<matchFunc>::template featureComparison<T_S, T_T, T_CV>(source_features, target_features);
		}

		_computed.template at<Nc>(cvIndex) = true;
		_cost_volume.template at<Nc>(cvIndex) = cost;

		return cost;
	}

	inline SearchSpace<sizeof... (Ds)> const& searchSpace() const {
		return _search_space;
	}

	template<Multidim::ArrayDataAccessConstness viewConstness>
	Multidim::Array<T_CV, nCostVolDim> truncatedCostVolume(Multidim::Array<disp_t, nDim, viewConstness> const& disp,
														   int radius = 1,
														   T_CV defaultVal = defaultCvValForMatchFunc<matchFunc, T_CV>()) const {

		if (disp.shape()[nDim-1] != nSearchDim) {
			return Multidim::Array<T_CV, nCostVolDim>();
		}

		std::array<int, nCostVolDim> tcv_shape;

		for (int i = 0; i < nDim-1; i++) {
			tcv_shape[i] = _cost_volume.shape()[i];
		}

		for (int i = 0; i < nSearchDim; i++) {
			tcv_shape[nDim - 1 + i] = 2*radius+1;
		}

		Multidim::Array<T_CV, nCostVolDim> tcv(tcv_shape);

		Multidim::IndexConverter idxConv(tcv_shape);

		#pragma omp parallel for
		for (int i = 0; i < idxConv.numberOfPossibleIndices(); i++) {

			std::array<int, nCostVolDim> tcvid = idxConv.getIndexFromPseudoFlatId(i);
			std::array<int, nCostVolDim> cvid = tcvid;

			std::array<int, nDim> dispId;

			for (int j = 0; j < nDim-1; j++) {
				dispId[j] = cvid[j];
			}

			for (int j = 0; j < nSearchDim; j++) {
				dispId[nDim-1] = j;
				disp_t delta = _search_space.disp2idx(_searchDims[j], disp.valueUnchecked(dispId));
				cvid[nDim - 1 + j] = tcvid[nDim - 1 + j]-radius + delta;
			}

			std::array<int, nDim-1> pos;
			std::array<int, nSearchDim> disp;

			for (int j = 0; j < nDim-1; j++) {
				pos[j] = cvid[j];
			}

			for (int j = 0; j < nSearchDim; j++) {
				disp[j] = cvid[nDim-1+j];
			}

			std::optional<T_CV> cand_cost = costValue(pos, disp);

			if (cand_cost.has_value()) {
				tcv.atUnchecked(tcvid) = cand_cost.value();
			} else {
				tcv.atUnchecked(tcvid) = defaultVal;
			}

		}

		return tcv;

	}

	template<Multidim::ArrayDataAccessConstness viewConstness, int nSrchDims = nSearchDim>
	std::enable_if_t<nSrchDims == 1, Multidim::Array<T_CV, nCostVolDim>>
	truncatedCostVolume(Multidim::Array<disp_t, nDim-1, viewConstness> const& disp,
						int radius = 1,
						T_CV defaultVal = defaultCvValForMatchFunc<matchFunc, T_CV>()) const {

		std::array<int, nCostVolDim> tcv_shape;

		for (int i = 0; i < nDim-1; i++) {
			tcv_shape[i] = _cost_volume.shape()[i];
		}

		for (int i = 0; i < nSearchDim; i++) {
			tcv_shape[nDim - 1 + i] = 2*radius+1;
		}

		Multidim::Array<T_CV, nCostVolDim> tcv(tcv_shape);

		Multidim::IndexConverter<nCostVolDim> idxConv(tcv_shape);

		#pragma omp parallel for
		for (int i = 0; i < idxConv.numberOfPossibleIndices(); i++) {

			std::array<int, nCostVolDim> tcvid = idxConv.getIndexFromPseudoFlatId(i);
			std::array<int, nCostVolDim> cvid = tcvid;

			std::array<int, nDim-1> dispId;

			for (int j = 0; j < nDim-1; j++) {
				dispId[j] = cvid[j];
			}

			disp_t delta = _search_space.disp2idx(_searchDims[0], disp.valueUnchecked(dispId));
			cvid[nDim - 1] = tcvid[nDim - 1]-radius + delta;

			std::array<int, nDim-1> pos;
			std::array<int, nSearchDim> disp;

			for (int j = 0; j < nDim-1; j++) {
				pos[j] = cvid[j];
			}

			for (int j = 0; j < nSearchDim; j++) {
				disp[j] = cvid[nDim-1+j];
			}

			std::optional<T_CV> cand_cost = costValue(pos, disp);

			if (cand_cost.has_value()) {
				tcv.atUnchecked(tcvid) = cand_cost.value();
			} else {
				tcv.atUnchecked(tcvid) = defaultVal;
			}

		}

		return tcv;

	}

protected:

	std::array<Multidim::array_size_t, nSearchDim> _searchDims;

	Multidim::Array<T_S, nDim, constnessS> const& _source;
	Multidim::Array<T_S, nDim, constnessT> const& _target;

	Multidim::Array<T_SF, nDim> _source_processed;
	Multidim::Array<T_TF, nDim> _target_processed;

	mutable Multidim::Array<T_CV, nCostVolDim> _cost_volume;
	mutable Multidim::Array<bool, nCostVolDim> _computed;

	SearchSpaceType _search_space;
};

template<matchingFunctions matchFunc, class T_CV, class T_S, class T_T,
		 Multidim::ArrayDataAccessConstness constnessS,
		 Multidim::ArrayDataAccessConstness constnessT>
using OnDemandStereoCostVolume =
GenericOnDemandCostVolume<matchFunc, T_CV, T_S, T_T,
constnessS, constnessT,
SearchSpaceBase::IgnoredDim, SearchSpaceBase::SearchDim, SearchSpaceBase::FeatureDim>;

template<matchingFunctions matchFunc, class T_CV, class T_S, class T_T,
		 Multidim::ArrayDataAccessConstness constnessS,
		 Multidim::ArrayDataAccessConstness constnessT>
using OnDemandImageFlowVolume =
GenericOnDemandCostVolume<matchFunc, T_CV, T_S, T_T,
constnessS, constnessT,
SearchSpaceBase::SearchDim, SearchSpaceBase::SearchDim, SearchSpaceBase::FeatureDim>;

} //namespace Correlation
} //namespace StereoVision

#endif // STEREOVISION_ON_DEMAND_COST_VOLUME_H
