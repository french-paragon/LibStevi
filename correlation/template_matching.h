#ifndef STEREOVISION_TEMPLATE_MATCHING_H
#define STEREOVISION_TEMPLATE_MATCHING_H

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
#include "./cross_correlations.h"
#include "./matching_costs.h"

#include "MultidimArrays/MultidimArrays.h"
#include "MultidimArrays/MultidimIndexManipulators.h"

namespace StereoVision {
namespace Correlation {


template<matchingFunctions matchFunc,  typename T_O = float, int matchDim = -1, typename  T_S, typename T_T,
		 Multidim::ArrayDataAccessConstness C_S, Multidim::ArrayDataAccessConstness C_T>
Multidim::Array<T_O, 2> matchPattern(Multidim::Array<T_S,1, C_S> const& features_template, Multidim::Array<T_T,3, C_T> const& search_feature_volume) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;
	constexpr int nDim = 3;

	static_assert((matchDim >= 0 and matchDim < nDim) or (matchDim < 0 and nDim+matchDim >= 0), "invalid matchDim template argument provided!");

	constexpr int matching_dim = (matchDim >= 0) ? matchDim : nDim+matchDim;

	static_assert(matching_dim >= 0 and matching_dim < nDim, "Now, this should not happen and is a bug!");

	using T_FS = typename MatchingFuncComputeTypeInfos<matchFunc, T_S>::FeatureType;
	using T_FT = typename MatchingFuncComputeTypeInfos<matchFunc, T_T>::FeatureType;


	if (search_feature_volume.shape()[matching_dim] != features_template.shape()[0]) {
		return Multidim::Array<T_O, nDim-1>(); //invalid template size.
	}

	Multidim::Array<T_FS,1> tmpl = getFeatureVectorForMatchFunc<matchFunc>(features_template);
	Multidim::Array<T_FT,3> search_features = getFeatureVolumeForMatchFunc<matchFunc>(search_feature_volume);

	int nFeatures = features_template.shape()[0];

	Multidim::IndexConverter<nDim> indexConverter(search_feature_volume.shape(),
												  Multidim::DimsExclusionSet<nDim>(matching_dim));

	typename Multidim::Array<T_O, nDim-1>::ShapeBlock shape;

	for (int i = 0; i < nDim; i++) {
		int s_id = i;

		if (i > matching_dim) {
			s_id -= 1;
		}

		if (i == matching_dim) {
			continue;
		}

		shape[s_id] = search_feature_volume.shape()[i];
	}

	Multidim::Array<T_O, nDim-1> out(shape);

	for (int i = 0; i < indexConverter.numberOfPossibleIndices(); i++) {

		auto fullIdx = indexConverter.getIndexFromPseudoFlatId(i);

		typename Multidim::Array<T_O, nDim-1>::IndexBlock subIdx;

		for (int id = 0; id <nDim; id++) {
			int s_id = id;

			if (id > matching_dim) {
				s_id -= 1;
			}

			if (id == matching_dim) {
				continue;
			}

			subIdx[s_id] = fullIdx[id];
		}

		Multidim::Array<T_FT,1> target = search_features.indexDimView(2, subIdx);


		out.template at<Nc>(subIdx) =
				MatchingFunctionTraits<matchFunc>::template featureComparison<T_FS, T_FT, T_O>(target, tmpl);

	}

	return out;

}

}
}

#endif // STEREOVISION_TEMPLATE_MATCHING_H
