#ifndef STEREOVISION_PATCHMATCH_H
#define STEREOVISION_PATCHMATCH_H

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
#include "../utils/randomcache.h"

#include <optional>
#include <random>

namespace StereoVision {
namespace Correlation {

template<matchingFunctions matchFunc, class T_FV, int searchSpaceDim>
struct PatchMatchTraits {
	static_assert (searchSpaceDim == 1 or searchSpaceDim == 2, "invalid searchSpaceDim");

	typedef std::function<Multidim::Array<disp_t, 3>(Multidim::Array<T_FV, 3> const&, Multidim::Array<T_FV, 3> const&)> PatchMatchInitializer;

	typedef typename std::conditional<std::is_integral_v<T_FV>, int32_t, float>::type TCV;
	typedef typename MatchingFuncComputeTypeInfos<matchFunc, T_FV>::FeatureType T_FVE;

	template<Multidim::ArrayDataAccessConstness constnessS,
			 Multidim::ArrayDataAccessConstness constnessT>
	using PatchMatchOnDemandCV = std::conditional_t<searchSpaceDim == 1,
	OnDemandStereoCostVolume<matchFunc, TCV, T_FV, T_FV, constnessS, constnessT>,
	OnDemandImageFlowVolume<matchFunc, TCV, T_FV, T_FV, constnessS, constnessT>> ;
};

template<int searchSpaceDim>
Multidim::Array<disp_t, 3> randomDispInit(std::array<int, 3> s_shape,
										  std::array<int, 3> t_shape,
										  std::optional<searchOffset<searchSpaceDim>> searchOffset,
										  std::optional<StereoVision::Random::NumbersCache<int>> randcache = std::nullopt)
{

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<disp_t, 3> disp (s_shape[0], s_shape[1], searchSpaceDim);

	int tid = 0;

	std::random_device rd;
	int rdseed = rd();

	#pragma omp parallel
	{
		//init a different random engine in each thread.
		int seed;
		#pragma omp critical
		{
			seed = rdseed + (tid++); //different seeds in each threads
		}

		std::default_random_engine re(seed);
		StereoVision::Random::NumbersCache<int> rnc;

		if (randcache.has_value()) {
			rnc = randcache.value();
			rnc.seed(seed);
		}

		std::uniform_int_distribution<disp_t> range1;
		std::uniform_int_distribution<disp_t> range2;

		if (searchOffset.has_value()) {

			range1 = std::uniform_int_distribution<disp_t>(searchOffset.value().lowerOffset(0), searchOffset.value().upperOffset(0));
			if (searchSpaceDim == 2) {
				range2 = std::uniform_int_distribution<disp_t>(searchOffset.value().lowerOffset(1), searchOffset.value().upperOffset(1));
			}
		} else {
			if (searchSpaceDim == 1) {
				range1 = std::uniform_int_distribution<disp_t>(0, t_shape[1]);
			}else if (searchSpaceDim == 2) {
				range1 = std::uniform_int_distribution<disp_t>(0, t_shape[0]);
				range2 = std::uniform_int_distribution<disp_t>(0, t_shape[1]);
			}
		}



		if (randcache.has_value()) {

			#pragma omp for
			for (int i = 0; i < disp.shape()[0]; i++) {
				for (int j = 0; j < disp.shape()[1]; j++) {
					if (searchOffset.has_value()) {

						disp.at<Nc>(i,j,0) = searchOffset.value().template setValueInRange<0>(rnc());
						if (searchSpaceDim == 2) {
							disp.at<Nc>(i,j,1) = searchOffset.value().template setValueInRange<1>(rnc());
						}

					} else {

						if (searchSpaceDim == 1) {
							disp.at<Nc>(i,j,0) = ((searchOffset.value().template setValueInRange<0>(rnc()) + j) % t_shape[1])-j;
						} else {
							disp.at<Nc>(i,j,0) = ((searchOffset.value().template setValueInRange<0>(rnc()) + i) % t_shape[0])-i;
							disp.at<Nc>(i,j,1) = ((searchOffset.value().template setValueInRange<1>(rnc()) + j) % t_shape[1])-j;
						}

					}
				}
			}

		} else {

			#pragma omp for
			for (int i = 0; i < disp.shape()[0]; i++) {
				for (int j = 0; j < disp.shape()[1]; j++) {
					if (searchOffset.has_value()) {

						disp.at<Nc>(i,j,0) = range1(re);
						if (searchSpaceDim == 2) {
							disp.at<Nc>(i,j,1) = range2(re);
						}

					} else {

						if (searchSpaceDim == 1) {
							disp.at<Nc>(i,j,0) = ((range1(re) + j) % t_shape[1])-j;
						} else {
							disp.at<Nc>(i,j,0) = ((range1(re) + i) % t_shape[0])-i;
							disp.at<Nc>(i,j,1) = ((range2(re) + j) % t_shape[1])-j;
						}

					}
				}
			}

		}
	}

	return disp;
}

template<matchingFunctions matchFunc, int searchSpaceDim, class T_FV, class TCV,
		 Multidim::ArrayDataAccessConstness constnessS,
		 Multidim::ArrayDataAccessConstness constnessT>
inline int patchMatchTestCost(Multidim::Array<disp_t, 3> & solution,
							  typename PatchMatchTraits<matchFunc, T_FV, searchSpaceDim>::template PatchMatchOnDemandCV<constnessS, constnessT> & cost,
							  int i,
							  int j,
							  disp_t disp_i,
							  disp_t disp_j) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int n_changed = 0;

	std::array<int, searchSpaceDim> d_c;
	if (searchSpaceDim == 1) {
		d_c[0] = disp_j;
	} else {
		d_c[0] = disp_i;
		d_c[1] = disp_j;
	}

	std::array<int, searchSpaceDim> d_a;
	if (searchSpaceDim == 1) {
		d_a[0] = solution.at<Nc>(i,j,0);
	} else {
		d_a[0] = solution.at<Nc>(i,j,0);
		d_a[1] = solution.at<Nc>(i,j,1);
	}

	std::array<int, 2> pos{i,j};

	auto opt_candidateCost = cost.costValue(pos,d_c);

	if (!opt_candidateCost.has_value()) {
		return 0;
	}

	TCV candidateCost = opt_candidateCost.value();

	bool keepNew = false;

	if (MatchingFunctionTraits<matchFunc>::extractionStrategy == dispExtractionStartegy::Score) {
		if (candidateCost >= cost.costValue(pos,d_a)) {
			keepNew = true;
		}
	} else if (MatchingFunctionTraits<matchFunc>::extractionStrategy == dispExtractionStartegy::Cost) {
		if (candidateCost <= cost.costValue(pos,d_a)) {
			keepNew = true;
		}
	}

	if (keepNew) {
		n_changed++;

		if (searchSpaceDim == 1) {
			solution.at<Nc>(i,j,0) = disp_j;
		} else if (searchSpaceDim == 2) {
			solution.at<Nc>(i,j,0) = disp_i;
			solution.at<Nc>(i,j,1) = disp_j;
		}
	}

	return n_changed;

}

template<matchingFunctions matchFunc, int searchSpaceDim, class T_FV, class TCV,
		 Multidim::ArrayDataAccessConstness constnessS,
		 Multidim::ArrayDataAccessConstness constnessT>
int patchMatchSearch(Multidim::Array<disp_t, 3> & solution,
					 typename PatchMatchTraits<matchFunc, T_FV, searchSpaceDim>::template PatchMatchOnDemandCV<constnessS, constnessT> & cost,
					 int nRandomSearch,
					 std::optional<StereoVision::Random::NumbersCache<int>> randcache = std::nullopt) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int n_changed = 0;

	auto shape = cost.shape();

	int tid = 0;

	std::random_device rd;
	int rdseed = rd();

	#pragma omp parallel
	{
		int n_chang_l = 0;

		//init a different random engine in each thread.
		int seed;
		#pragma omp critical
		{
			seed = rdseed + (tid++); //different seeds in each threads
		}

		std::default_random_engine re(seed);
		StereoVision::Random::NumbersCache<int> rnc;

		if (randcache.has_value()) {
			rnc = randcache.value();
			rnc.seed(seed);
		}

		std::uniform_int_distribution<disp_t> range1;
		std::uniform_int_distribution<disp_t> range2;

		range1 = std::uniform_int_distribution<disp_t>(cost.searchSpace().getDimMinSearchRange(0), cost.searchSpace().getDimMaxSearchRange(0));
		if (searchSpaceDim == 2) {
			range2 = std::uniform_int_distribution<disp_t>(cost.searchSpace().getDimMinSearchRange(1), cost.searchSpace().getDimMaxSearchRange(1));
		}

		#pragma omp for
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {

				disp_t disp_i_b;
				disp_t disp_j_b;

				if (searchSpaceDim == 1) {
					disp_i_b = 0;
					disp_j_b = solution.at<Nc>(i,j,0);
				} else if (searchSpaceDim == 2) {
					disp_i_b = solution.at<Nc>(i,j,0);
					disp_j_b = solution.at<Nc>(i,j,1);
				}

				int n_chang = 0;

				disp_t disp_i;
				disp_t disp_j;


				for (int k = 0; k < nRandomSearch; k++) {

					if (searchSpaceDim == 1) {
						disp_i = 0;

						if (randcache.has_value()) {
							disp_j = cost.searchSpace().template setValueInRange<0>(rnc());
						} else {
							disp_j = range1(re);
						}

					} else if (searchSpaceDim == 2) {

						if (randcache.has_value()) {
							disp_i = cost.searchSpace().template setValueInRange<0>(rnc());
							disp_j = cost.searchSpace().template setValueInRange<1>(rnc());
						} else {
							disp_i = range1(re);
							disp_j = range2(re);
						}
					}

					//scale the disp to favor exploration near the current solution
					disp_t delta_i = disp_i - disp_i_b;
					disp_t delta_j = disp_j - disp_j_b;

					delta_j *= k+1;
					delta_j /= nRandomSearch+1;
					if (searchSpaceDim == 2) {
						delta_i *= k+1;
						delta_i /= nRandomSearch+1;
					}

					disp_i = disp_i_b + delta_i;
					disp_j = disp_j_b + delta_j;

					if (searchSpaceDim == 1) {
						if (disp_j == disp_j_b) {
							disp_j = disp_j_b+1;
						}
					} else if (searchSpaceDim == 2) {
						if (disp_i == disp_i_b and disp_j == disp_j_b) {
							disp_i = disp_i_b+1;
							disp_j = disp_j_b+1;
						}
					}

					n_chang = patchMatchTestCost
							<matchFunc, searchSpaceDim, T_FV, TCV, constnessS, constnessT>
							(solution,
							 cost,
							 i,
							 j,
							 disp_i,
							 disp_j);
				}

				n_chang_l += n_chang;

			}
		}

		#pragma omp critical
		{
			n_changed += n_chang_l; //different seeds in each threads
		}

	}

	return n_changed;

}

template<PropagationDirection::Direction direction, matchingFunctions matchFunc, int searchSpaceDim, class T_FV, class TCV,
		 Multidim::ArrayDataAccessConstness constnessS,
		 Multidim::ArrayDataAccessConstness constnessT>
int patchMatchPropagate(Multidim::Array<disp_t, 3> & solution,
						typename PatchMatchTraits<matchFunc, T_FV, searchSpaceDim>::template PatchMatchOnDemandCV<constnessS, constnessT> & cost) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	constexpr std::array<int, 2> increments = PropagationDirection::Traits<direction>::increments;

	static_assert (increments[0] == 1 or increments[0] == -1, "Wrong direction template: increments have to be either 1 or -1");
	static_assert (increments[1] == 1 or increments[1] == -1, "Wrong direction template: increments have to be either 1 or -1");

	auto shape = cost.shape();

	int n_changed = 0;

	PropagationDirection::IndexRange irange = PropagationDirection::initialAndFinalPos<increments[0]>(shape[0]);
	PropagationDirection::IndexRange jrange = PropagationDirection::initialAndFinalPos<increments[1]>(shape[1]);


	//lines scans
	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = jrange.initial; j != jrange.final; j += increments[1]) {

			int p_j = j - increments[1];

			if (p_j >= 0 and p_j < shape[1]) {
				Multidim::Array<disp_t, 1> dispVec = solution.subView(Multidim::DimIndex(i), Multidim::DimIndex(p_j), Multidim::DimSlice());

				disp_t disp_i = (searchSpaceDim == 2) ? dispVec.at<Nc>(0) : 0;
				disp_t disp_j = (searchSpaceDim == 2) ? dispVec.at<Nc>(1) : dispVec.at<Nc>(0);

				n_changed += patchMatchTestCost
						<matchFunc, searchSpaceDim, T_FV, TCV, constnessS, constnessT>
						(solution,
						 cost,
						 i,
						 j,
						 disp_i,
						 disp_j);

			}
		}
	}

	//columns scans
	#pragma omp parallel for
	for (int j = 0; j < shape[1]; j++) {

		for (int i = irange.initial; i != irange.final; i += increments[0]) {

			int p_i = i - increments[0];

			if (p_i >= 0 and p_i < shape[0]) {
				Multidim::Array<disp_t, 1> dispVec = solution.subView(Multidim::DimIndex(p_i), Multidim::DimIndex(j), Multidim::DimSlice());

				disp_t disp_i = (searchSpaceDim == 2) ? dispVec.at<Nc>(0) : 0;
				disp_t disp_j = (searchSpaceDim == 2) ? dispVec.at<Nc>(1) : dispVec.at<Nc>(0);

				n_changed += patchMatchTestCost
						<matchFunc, searchSpaceDim, T_FV, TCV, constnessS, constnessT>
						(solution,
						 cost,
						 i,
						 j,
						 disp_i,
						 disp_j);

			}
		}

	}

	return n_changed;
}

template<matchingFunctions matchFunc, int searchSpaceDim, class T_FV>
Multidim::Array<disp_t, 3> patchMatch(Multidim::Array<T_FV, 3> const& feature_vol_s_p,
									  Multidim::Array<T_FV, 3> const& feature_vol_t_p,
									  searchOffset<searchSpaceDim> searchOffset,
									  int nIter = 5,
									  int nRandomSearch = 4,
									  std::optional<typename PatchMatchTraits<matchFunc, T_FV, searchSpaceDim>::PatchMatchInitializer> initializer = std::nullopt,
									  std::optional<StereoVision::Random::NumbersCache<int>> randcache = std::nullopt) {

	constexpr Multidim::ArrayDataAccessConstness C_S = Multidim::NonConstView;
	constexpr Multidim::ArrayDataAccessConstness C_T = Multidim::NonConstView;

	using TCV = typename PatchMatchTraits<matchFunc, T_FV, searchSpaceDim>::TCV;
	using T_FVE = typename PatchMatchTraits<matchFunc, T_FV, searchSpaceDim>::T_FVE;

	using CostVolT = typename PatchMatchTraits<matchFunc, T_FVE, searchSpaceDim>::template PatchMatchOnDemandCV<C_S, C_T>;
	using SearchSpaceT = typename CostVolT::SearchSpaceType;

	static_assert (CostVolT::nSearchDim == searchSpaceDim, "Error in cv type");

	constexpr int dim0Idx = 0;
	constexpr int dim1Idx = (searchSpaceDim == 2) ? 1 : 0;

	using Dim0TypeT = std::conditional_t<searchSpaceDim == 2, SearchSpaceBase::SearchDim, SearchSpaceBase::IgnoredDim>;

	Multidim::Array<T_FVE, 3> feature_vol_s = getFeatureVolumeForMatchFunc<matchFunc>(feature_vol_s_p);
	Multidim::Array<T_FVE, 3> feature_vol_t = getFeatureVolumeForMatchFunc<matchFunc>(feature_vol_t_p);

	static_assert (searchSpaceDim == 1 or searchSpaceDim == 2, "patchMatch function can only be used to search in 1 or two dimension !");

	Multidim::Array<disp_t, 3> disp;

	if (feature_vol_s.shape()[2] != feature_vol_t.shape()[2]) {
		return disp; //return empty array
	}

	if (searchSpaceDim == 1) {
		if (feature_vol_s.shape()[0] != feature_vol_t.shape()[0]) {
			return disp; //return empty array
		}
	}

	if (initializer.has_value()) {
		disp = initializer.value()(feature_vol_s_p, feature_vol_t_p);
	} else {
		disp = randomDispInit<searchSpaceDim>({feature_vol_s.shape()[0], feature_vol_s.shape()[1]},
											  {feature_vol_t.shape()[0], feature_vol_t.shape()[1]},
											  searchOffset,
											  randcache);
	}

	SearchSpaceT searchSpace (Dim0TypeT(searchOffset.template lowerOffset<dim0Idx>(), searchOffset.template upperOffset<dim0Idx>()),
							  SearchSpaceBase::SearchDim(searchOffset.template lowerOffset<dim1Idx>(), searchOffset.template upperOffset<dim1Idx>()),
							  SearchSpaceBase::FeatureDim());

	CostVolT cost(feature_vol_s, feature_vol_t, searchSpace);

	for (int i = 0; i < nIter; i++) {

		int nChanges = 0;

		switch (i % 4) {
		case 0:
			nChanges += patchMatchPropagate<PropagationDirection::TopLeftToBottomRight, matchFunc, searchSpaceDim, T_FVE, TCV, C_S, C_T>
					(disp, cost);
			break;
		case 1:
			nChanges += patchMatchPropagate<PropagationDirection::TopRightToBottomLeft, matchFunc, searchSpaceDim, T_FVE, TCV, C_S, C_T>
					(disp, cost);
			break;
		case 2:
			nChanges += patchMatchPropagate<PropagationDirection::BottomLeftToTopRight, matchFunc, searchSpaceDim, T_FVE, TCV, C_S, C_T>
					(disp, cost);
			break;
		default:
			nChanges += patchMatchPropagate<PropagationDirection::BottomRightToTopLeft, matchFunc, searchSpaceDim, T_FVE, TCV, C_S, C_T>
					(disp, cost);
			break;
		}

		nChanges += patchMatchSearch<matchFunc, searchSpaceDim, T_FVE, TCV, C_S, C_T>(disp,
																			cost,
																			nRandomSearch,
																			randcache);

		if (nChanges == 0) { //no changes mean we can break early
			break;
		}
	}

	return disp;
}

} //namespace Correlation
} //namespace StereoVision

#endif // STEREOVISION_PATCHMATCH_H
