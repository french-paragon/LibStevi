#ifndef STEREOVISION_FAST_APPROXIMATE_MATCHING_H
#define STEREOVISION_FAST_APPROXIMATE_MATCHING_H
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

#include "./cross_correlations.h"
#include "./on_demand_cost_volume.h"

#include "../utils/propagation_direction.h"

namespace StereoVision {
namespace Correlation {

template<matchingFunctions matchFunc, class T_FV, int searchSpaceDim>
struct FastMatchTraits {
	typedef typename std::conditional<std::is_integral_v<T_FV>, int32_t, float>::type TCV;
	typedef typename MatchingFuncComputeTypeInfos<matchFunc, T_FV>::FeatureType T_FVE;

	template<Multidim::ArrayDataAccessConstness constnessS,
			 Multidim::ArrayDataAccessConstness constnessT>
	using FastMatchOnDemandCV = std::conditional_t<searchSpaceDim == 1,
	OnDemandStereoCostVolume<matchFunc, TCV, T_FV, T_FV, constnessS, constnessT>,
	OnDemandImageFlowVolume<matchFunc, TCV, T_FV, T_FV, constnessS, constnessT>> ;
};

template<matchingFunctions matchFunc, int searchSpaceDim, class T_FV, class T_CV,
		 Multidim::ArrayDataAccessConstness constnessS,
		 Multidim::ArrayDataAccessConstness constnessT>
inline std::array<disp_t, searchSpaceDim> fullDispAtIdx(Multidim::Array<disp_t, 3> & disp,
														typename FastMatchTraits<matchFunc, T_FV, searchSpaceDim>::template FastMatchOnDemandCV<constnessS, constnessT> const& cost,
														searchOffset<searchSpaceDim> searchOffset,
														int i,
														int j) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	std::array<int, searchSpaceDim> dispCoord;

	if (searchSpaceDim == 1) {
		disp_t currentD = 0;
		T_CV currentCost = defaultCvValForMatchFunc<matchFunc, T_CV>();

		for (disp_t d = searchOffset.template lowerOffset<0>(); d < searchOffset.template upperOffset<0>(); d++) {
			dispCoord[0] = d;
			auto opt_c = cost.costValue({i,j}, dispCoord);

			if (!opt_c.has_value()) {
				continue;
			}

			T_CV c = opt_c.value();

			std::tie(currentD, currentCost) = optimalDispAndCost<matchFunc>(currentD, currentCost, d, c);
		}

		disp.at<Nc>(i,j,0) = currentD;

	} else {
		std::array<int, 2> currentD = {0,0};
		T_CV currentCost = defaultCvValForMatchFunc<matchFunc, T_CV>();

		for (disp_t d1 = searchOffset.template lowerOffset<0>(); d1 < searchOffset.template upperOffset<0>(); d1++) {
			for (disp_t d2 = searchOffset.template lowerOffset<1>(); d2 < searchOffset.template upperOffset<1>(); d2++) {

				std::array<disp_t, 2> d = {d1,d2};
				dispCoord[0] = d1;
				dispCoord[1] = d2;

				auto opt_c = cost.costValue({i,j}, dispCoord);

				if (!opt_c.has_value()) {
					continue;
				}

				T_CV c = opt_c.value();

				std::tie(currentD, currentCost) = optimalDispAndCost<matchFunc>(currentD, currentCost, d, c);
			}
		}

		disp.at<Nc>(i,j,0) = currentD[0];
		disp.at<Nc>(i,j,1) = currentD[1];
	}

}

template<matchingFunctions matchFunc, int searchSpaceDim, class T_FV,
		 Multidim::ArrayDataAccessConstness constnessS,
		 Multidim::ArrayDataAccessConstness constnessT>
Multidim::Array<disp_t, 3> fastmatch(Multidim::Array<T_FV, 3, constnessS> const& feature_vol_s,
									 Multidim::Array<T_FV, 3, constnessT> const& feature_vol_t,
									 searchOffset<searchSpaceDim> searchOffset) {

	using TCV = typename FastMatchTraits<matchFunc, T_FV, searchSpaceDim>::TCV;
	using T_FVE = typename FastMatchTraits<matchFunc, T_FV, searchSpaceDim>::T_FVE;

	using CostVolT = typename FastMatchTraits<matchFunc, T_FV, searchSpaceDim>::template FastMatchOnDemandCV<constnessS, constnessT>;
	using SearchSpaceT = typename CostVolT::SearchSpaceType;

	static_assert (CostVolT::nSearchDim == searchSpaceDim, "Error in cv type");

	constexpr int dim0Idx = 0;
	constexpr int dim1Idx = (searchSpaceDim == 2) ? 1 : 0;

	using Dim0TypeT = std::conditional_t<searchSpaceDim == 2, SearchSpaceBase::SearchDim, SearchSpaceBase::IgnoredDim>;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	static_assert (searchSpaceDim == 1 or searchSpaceDim == 2, "patchMatch function can only be used to search in 1 or two dimension !");

	if (feature_vol_s.shape()[2] != feature_vol_t.shape()[2]) {
		return Multidim::Array<disp_t, 3>(); //return empty array
	}

	if (searchSpaceDim == 1) {
		if (feature_vol_s.shape()[0] != feature_vol_t.shape()[0]) {
			return Multidim::Array<disp_t, 3>(); //return empty array
		}
	}

	Multidim::Array<disp_t, 3> disp(feature_vol_s.shape()[0], feature_vol_s.shape()[1], searchSpaceDim);


	SearchSpaceT searchSpace (Dim0TypeT(searchOffset.template lowerOffset<dim0Idx>(), searchOffset.template upperOffset<dim0Idx>()),
							  SearchSpaceBase::SearchDim(searchOffset.template lowerOffset<dim1Idx>(), searchOffset.template upperOffset<dim1Idx>()),
							  SearchSpaceBase::FeatureDim());

	CostVolT cost(feature_vol_s, feature_vol_t, searchOffset);


	for (int i = 0; i < feature_vol_s.shape()[0]; i++) {

		fullDispAtIdx(disp, cost, searchOffset, i, 0);

		std::array<int, searchSpaceDim> dispCoord;

		bool previous_jumped = true;

		for (int j = 1; j < feature_vol_s.shape()[1]; j++) {

			bool current_jumped = false;

			if (searchSpaceDim == 1) {
				disp_t previousD = disp.value<Nc>(i,j-1,0);
				disp_t currentD = previousD;

				dispCoord[0] = previousD;
				TCV noMoveCost = cost.costValue(i,j, dispCoord);
				TCV currentCost = noMoveCost;

				if (searchOffset.valueInRange<0>(previousD-1)) {
					dispCoord[0] = previousD-1;
					TCV backCost = cost.costValue(i,j, dispCoord);

					std::tie(currentD, currentCost) = optimalDispAndCost<matchFunc>(currentD, currentCost, dispCoord[0], backCost);

				}

				if (searchOffset.valueInRange<0>(previousD+1)) {
					dispCoord[0] = previousD+1;
					TCV forwardCost = cost.costValue(i,j, dispCoord);

					std::tie(currentD, currentCost) = optimalDispAndCost<matchFunc>(currentD, currentCost, dispCoord[0], forwardCost);

				}

				current_jumped = (currentD != previousD);

				if (current_jumped and previous_jumped) {
					fullDispAtIdx(disp, cost, searchOffset, i, j);
				} else {
					disp.at<Nc>(i,j,0) = currentD;
				}

				previous_jumped = current_jumped;

			} else {
				std::array<disp_t, 2> previousD = {disp.value<Nc>(i,j-1,0), disp.value<Nc>(i,j-1,1)};
				std::array<disp_t, 2> currentD = previousD;

				dispCoord[0] = previousD[0];
				dispCoord[1] = previousD[1];

				TCV noMoveCost = cost.costValue(i,j, dispCoord);
				TCV currentCost = noMoveCost;

				for (int delta1 = -1; delta1 <= 1; delta1++) {
					for (int delta2 = -1; delta2 <= 1; delta2++) {

						if (delta1 == 0 and delta2 == 0) {
							continue;
						}

						if (searchOffset.valueInRange<0>(previousD[0]+delta1) and
								searchOffset.valueInRange<1>(previousD[1]+delta2)) {
							dispCoord[0] = previousD[0]+delta1;
							dispCoord[1] = previousD[1]+delta2;
							TCV backCost = cost.costValue(i,j, dispCoord);

							std::array<disp_t, 2> cD = {previousD[0]+delta1, previousD[1]+delta2};
							std::tie(currentD, currentCost) = optimalDispAndCost<matchFunc>(currentD, currentCost, cD, backCost);

						}

					}
				}

				current_jumped = (currentD != previousD);

				if (current_jumped and previous_jumped) {
					fullDispAtIdx(disp, cost, searchOffset, i, j);
				} else {
					disp.at<Nc>(i,j,0) = currentD[0];
					disp.at<Nc>(i,j,0) = currentD[1];
				}

				previous_jumped = current_jumped;

			}

		}
	}

	return disp;

}

} //namespace Correlation
} //namespace StereoVision

#endif // STEREOVISION_FAST_APPROXIMATE_MATCHING_H
