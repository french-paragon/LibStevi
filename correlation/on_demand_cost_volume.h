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

#include <MultidimArrays/MultidimArrays.h>

namespace StereoVision {
namespace Correlation {

template<matchingFunctions matchFunc, class T_L, class T_R, class T_CV, dispDirection dDir = dispDirection::RightToLeft>
class OnDemandStereoCostVolume {

	using T_S = typename condImgRef<T_L, T_R, dDir, 3>::T_S;
	using T_T = typename condImgRef<T_L, T_R, dDir, 3>::T_T;

	static_assert (matchingcosts_details::MatchingFunctionTraitsInfos<matchFunc>::template traitsHasFeatureComparison<T_S, T_T, T_CV>(),
	"Trying to implement an OnDemandCostVolume with incompatible feature and cost volume types.");

public:

	explicit OnDemandStereoCostVolume(Multidim::Array<T_L, 3> const& left, Multidim::Array<T_R, 3> const& right, int searchRange) :
		_feature_volumes(left, right),
		_cv(_feature_volumes.source().shape()[0], _feature_volumes.source().shape()[1], searchRange),
		_computed(_feature_volumes.source().shape()[0], _feature_volumes.source().shape()[1], searchRange)
	{

		for(int i = 0; i < _computed.flatLenght(); i++) {
			(&_computed.atUnchecked(0,0,0))[i] = false;
		}

	}

	inline T_CV costValue(int i, int j, int d) const {

		if (_computed.valueUnchecked(i,j,d)) {
			return _cv.valueUnchecked(i,j,d);
		}

		constexpr int dirSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

		if (d < 0 or d >= _cv.shape()[2]) {
			return defaultCvValForMatchFunc<matchFunc>();
		}

		if (j+dirSign*d >= _feature_volumes.target().shape()[1] or j+dirSign*d < 0) {

			_cv.atUnchecked(i,j,d) = defaultCvValForMatchFunc<matchFunc>();
			_computed.atUnchecked(i,j,d) = true;

		} else {

			auto fSource = const_cast<Multidim::Array<T_S,3>&>(_feature_volumes.source()).subView(Multidim::DimIndex(i), Multidim::DimIndex(j), Multidim::DimSlice());
			auto fTarget = const_cast<Multidim::Array<T_T,3>&>(_feature_volumes.target()).subView(Multidim::DimIndex(i), Multidim::DimIndex(j+dirSign*d), Multidim::DimSlice());

			T_CV val = MatchingFunctionTraits<matchFunc>::template featureComparison<T_S, T_T, T_CV>(fSource, fTarget);

			_cv.atUnchecked(i,j,d) = val;
			_computed.atUnchecked(i,j,d) = true;
		}

		return _cv.valueUnchecked(i,j,d);
	}

private:

	condImgRef<T_L, T_R, dDir, 3> _feature_volumes;

	mutable Multidim::Array<T_CV,3> _cv;
	mutable Multidim::Array<bool,3> _computed;

};

template<matchingFunctions matchFunc, class T_S, class T_T, class T_CV, int searchSpaceDims>
class OnDemandCostVolume {
	static_assert (searchSpaceDims == 1 or searchSpaceDims == 2, "Cannot search in more than 2 dimensions yet");
};

template<matchingFunctions matchFunc, class T_S, class T_T, class T_CV>
class OnDemandCostVolume<matchFunc, T_S, T_T, T_CV, 1> {

public:

	explicit OnDemandCostVolume(Multidim::Array<T_S, 3> const& source, Multidim::Array<T_T, 3> const& target, searchOffset<1> searchRange):
		_searchRange(searchRange),
		_source(source),
		_target(target),
		_cv(_source.shape()[0],_source.shape()[1], searchRange.dimRange<0>()),
		_computed(_source.shape()[0], _source.shape()[1], searchRange.dimRange<0>())
	{

		for(int i = 0; i < _computed.flatLenght(); i++) {
			(&_computed.atUnchecked(0,0,0))[i] = false;
		}
	}

	inline T_CV costValue(int i, int j, std::array<int, 1> d) const {
		return costValue(i, j, d[0]);
	}

	inline T_CV costValue(int i, int j, int d) const {

		int d_id = _searchRange.disp2idx<0>(d);

		if (_computed.valueUnchecked(i,j,d_id)) {
			return _cv.valueUnchecked(i,j,d_id);
		}

		if (!_searchRange.valueInRange<0>(d)) {
			return defaultCvValForMatchFunc<matchFunc, T_CV>();
		}

		if (j+d >= _target.shape()[1] or j+d < 0) {

			_cv.atUnchecked(i,j,d_id) = defaultCvValForMatchFunc<matchFunc, T_CV>();
			_computed.atUnchecked(i,j,d_id) = true;

		} else {

			auto fSource = const_cast<Multidim::Array<T_S,3>&>(_source).subView(Multidim::DimIndex(i), Multidim::DimIndex(j), Multidim::DimSlice());
			auto fTarget = const_cast<Multidim::Array<T_T,3>&>(_target).subView(Multidim::DimIndex(i), Multidim::DimIndex(j+d), Multidim::DimSlice());

			T_CV val = MatchingFunctionTraits<matchFunc>::template featureComparison<T_S, T_T, T_CV>(fSource, fTarget);

			_cv.atUnchecked(i,j,d_id) = val;
			_computed.atUnchecked(i,j,d_id) = true;
		}

		return _cv.valueUnchecked(i,j,d_id);
	}

	auto shape() const {
		return _cv.shape();
	}

	searchOffset<1> const& searchRange() const {
		return _searchRange;
	}

	Multidim::Array<T_S, 3> const& source() const {
		return _source;
	}

	Multidim::Array<T_S, 3> const& target() const {
		return _target;
	}

private:

	searchOffset<1> _searchRange;

	Multidim::Array<T_S, 3> const& _source;
	Multidim::Array<T_T, 3> const& _target;

	mutable Multidim::Array<T_CV,3> _cv;
	mutable Multidim::Array<bool,3> _computed;

};

template<matchingFunctions matchFunc, class T_S, class T_T, class T_CV>
class OnDemandCostVolume<matchFunc, T_S, T_T, T_CV, 2> {

public:

	explicit OnDemandCostVolume(Multidim::Array<T_S, 3> const& source, Multidim::Array<T_T, 3> const& target, searchOffset<1> searchRange):
		_searchRange(searchRange),
		_source(source),
		_target(target),
		_cv(_source.shape()[0],_source.shape()[1], searchRange.dimRange<0>(), searchRange.dimRange<1>()),
		_computed(_source.shape()[0], _source.shape()[1], searchRange.dimRange<0>(), searchRange.dimRange<1>())
	{

		for(int i = 0; i < _computed.flatLenght(); i++) {
			(&_computed.atUnchecked(0,0,0))[i] = false;
		}
	}

	inline T_CV costValue(int i, int j, std::array<int, 2> d) const {
		return costValue(i, j, d[0], d[1]);
	}

	inline T_CV costValue(int i, int j, int d1, int d2) const {

		int d1_id = _searchRange.disp2idx<0>(d1);
		int d2_id = _searchRange.disp2idx<0>(d2);

		if (_computed.valueUnchecked(i,j,d1_id, d2_id)) {
			return _cv.valueUnchecked(i,j,d1_id,d2_id);
		}

		if (!_searchRange.valueInRange<0>(d1)) {
			return defaultCvValForMatchFunc<matchFunc>();
		}

		if (!_searchRange.valueInRange<1>(d2)) {
			return defaultCvValForMatchFunc<matchFunc>();
		}

		if (i+d1 >= _target.shape()[0] or i+d1 < 0 or j+d2 >= _target.shape()[1] or j+d2 < 0) {

			_cv.atUnchecked(i,j,d1_id,d2_id) = defaultCvValForMatchFunc<matchFunc>();
			_computed.atUnchecked(i,j,d1_id,d2_id) = true;

		} else {

			auto fSource = const_cast<Multidim::Array<T_S,3>&>(_source).subView(Multidim::DimIndex(i), Multidim::DimIndex(j), Multidim::DimSlice());
			auto fTarget = const_cast<Multidim::Array<T_T,3>&>(_target).subView(Multidim::DimIndex(i+d1), Multidim::DimIndex(j+d2), Multidim::DimSlice());

			T_CV val = MatchingFunctionTraits<matchFunc>::template featureComparison<T_S, T_T, T_CV>(fSource, fTarget);

			_cv.atUnchecked(i,j,d1_id,d2_id) = val;
			_computed.atUnchecked(i,j,d1_id,d2_id) = true;
		}

		return _cv.valueUnchecked(i,j,d1_id,d2_id);
	}

	auto shape() const {
		return _cv.shape();
	}

	searchOffset<1> const& searchRange() const {
		return _searchRange;
	}

	Multidim::Array<T_S, 3> const& source() const {
		return _source;
	}

	Multidim::Array<T_S, 3> const& target() const {
		return _target;
	}

private:

	searchOffset<1> _searchRange;

	Multidim::Array<T_S, 3> const& _source;
	Multidim::Array<T_T, 3> const& _target;

	mutable Multidim::Array<T_CV,4> _cv;
	mutable Multidim::Array<bool,4> _computed;

};

} //namespace Correlation
} //namespace StereoVision

#endif // STEREOVISION_ON_DEMAND_COST_VOLUME_H
