#ifndef STEREOVISION_CROSS_CORRELATIONS_H
#define STEREOVISION_CROSS_CORRELATIONS_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021  Paragon<french.paragon@gmail.com>

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

#include "./matching_costs.h"
#include "./correlation_base.h"
#include "./cost_based_refinement.h"
#include "./unfold.h"
#include "./census.h"

#include "../utils/contiguity.h"
#include "../utils/types_manipulations.h"

namespace StereoVision {
namespace Correlation {

template<class T_I, class T_M, class T_O = float>
T_O channelsZeroMeanNorm (Multidim::Array<T_I, 1> const& in_data,
									   T_M const& mean) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int f = in_data.shape()[0];

	T_O norm = 0;

	for (int c = 0; c < f; c++) {
		T_O tmp = static_cast<T_M>(in_data.template value<Nc>(c)) - mean;
		if ((std::is_integral<T_O>::value)) {
			norm = std::max(norm, static_cast<T_O>(std::abs(tmp)));
		} else {
			norm += tmp*tmp;
		}
	}

	if ((std::is_integral<T_O>::value)) {
		return norm;
	}

	return sqrtf(norm);
}

template<class T_I, class T_M, class T_O = float>
Multidim::Array<T_O, 2> channelsZeroMeanNorm (Multidim::Array<T_I, 3> const& in_data,
									   Multidim::Array<T_M, 2> const& mean) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = in_data.shape()[0];
	int w = in_data.shape()[1];
	int f = in_data.shape()[2];

	Multidim::Array<T_O, 2> norm(h, w);

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {

			norm.template at<Nc>(i,j) = 0;

			for (int c = 0; c < f; c++) {
				T_O tmp = static_cast<T_M>(in_data.template value<Nc>(i,j,c)) - mean.template value<Nc>(i,j);
				if ((std::is_integral<T_O>::value)) {
					norm.template at<Nc>(i,j) = std::max(norm.template value<Nc>(i,j), static_cast<T_O>(std::abs(tmp)));
				} else {
					norm.template at<Nc>(i,j) += tmp*tmp;
				}
			}
		}
	}

	if ((std::is_integral<T_O>::value)) {
		return norm;
	}

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {
			norm.template at<Nc>(i,j) = sqrtf(norm.template value<Nc>(i,j));
		}
	}

	return norm;
}

template<class T_I, class T_M = float, class T_O = float>
T_O channelsZeroMeanNorm (Multidim::Array<T_I, 1> const& in_data) {

	T_M mean = channelsMean<T_I, T_M>(in_data);

	return channelsZeroMeanNorm<T_I, T_M, T_O>(in_data, mean);

}

template<class T_I, class T_M = float, class T_O = float>
Multidim::Array<T_O, 2> channelsZeroMeanNorm (Multidim::Array<T_I, 3> const& in_data) {

	Multidim::Array<T_M, 2> mean = channelsMean<T_I, T_M>(in_data);

	return channelsZeroMeanNorm<T_I, T_M, T_O>(in_data, mean);

}

template<class T_I, class T_O = float>
T_O channelsNorm (Multidim::Array<T_I, 1> const& in_data) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int f = in_data.shape()[0];

	T_O norm = 0;

	for (int c = 0; c < f; c++) {
		T_O tmp = static_cast<T_O>(in_data.template value<Nc>(c));
		if ((std::is_integral<T_O>::value)) {
			norm = std::max(norm, static_cast<T_O>(std::abs(tmp)));
		} else {
			norm += tmp*tmp;
		}
	}

	if ((std::is_integral<T_O>::value)) {
		return norm;
	}

	return sqrtf(norm);
}

template<class T_I, class T_O = float>
Multidim::Array<T_O, 2> channelsNorm (Multidim::Array<T_I, 3> const& in_data) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = in_data.shape()[0];
	int w = in_data.shape()[1];
	int f = in_data.shape()[2];

	Multidim::Array<T_O, 2> norm(h, w);

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {

			norm.template at<Nc>(i,j) = 0;

			for (int c = 0; c < f; c++) {
				T_O tmp = static_cast<T_O>(in_data.template value<Nc>(i,j,c));
				if ((std::is_integral<T_O>::value)) {
					norm.template at<Nc>(i,j) = std::max(norm.template value<Nc>(i,j), static_cast<T_O>(std::abs(tmp)));
				} else {
					norm.template at<Nc>(i,j) += tmp*tmp;
				}
			}
		}
	}

	if ((std::is_integral<T_O>::value)) {
		return norm;
	}

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {
			norm.template at<Nc>(i,j) = sqrtf(norm.template value<Nc>(i,j));
		}
	}

	return norm;
}


template<matchingFunctions matchFunc, class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, typename TCV = float>
inline Multidim::Array<TCV, 3> aggregateCost(Multidim::Array<T_L, 3> const& feature_vol_l,
											 Multidim::Array<T_R, 3> const& feature_vol_r,
											 disp_t disp_width) {

	condImgRef<T_L, T_R, dDir, 3> dirInfos(feature_vol_l, feature_vol_r);
	using T_S = typename condImgRef<T_L, T_R, dDir>::T_S;
	using T_T = typename condImgRef<T_L, T_R, dDir>::T_T;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;
	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

	auto l_shape = feature_vol_l.shape();
	auto r_shape = feature_vol_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<TCV, 3>(0,0,0);
	}

	Multidim::Array<T_S, 3> & source_feature_volume = const_cast<Multidim::Array<T_S, 3> &>(dirInfos.source());
	Multidim::Array<T_T, 3> & target_feature_volume = const_cast<Multidim::Array<T_T, 3> &>(dirInfos.target());

	int h = source_feature_volume.shape()[0];
	int w = source_feature_volume.shape()[1];
	int f = source_feature_volume.shape()[2];

	Multidim::Array<TCV, 3> costVolume({h,w,disp_width}, {w*disp_width, 1, w});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {

			Multidim::Array<T_S, 1> source_feature_vector =
					source_feature_volume.subView(Multidim::DimIndex(i), Multidim::DimIndex(j), Multidim::DimSlice());

			for (int d = 0; d < disp_width; d++) {

				Multidim::Array<T_T, 1> target_feature_vector(f);

				for (int c = 0; c < f; c++) {
					float t = target_feature_volume.valueOrAlt({i,j+deltaSign*d,c}, 0);
					target_feature_vector.template at<Nc>(c) = t;
				}

				costVolume.template at<Nc>(i,j,d) =
						MatchingFunctionTraits<matchFunc>::template featureComparison<T_S, T_T, TCV>(source_feature_vector, target_feature_vector);

			}
		}
	}

	return costVolume;


}

template<matchingFunctions matchFunc, class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, typename TCV = float>
inline Multidim::Array<TCV, 4> aggregateCost(Multidim::Array<T_L, 3> const& feature_vol_l,
											 Multidim::Array<T_R, 3> const& feature_vol_r,
											 searchOffset<2> searchRange) {

	condImgRef<T_L, T_R, dDir, 3> dirInfos(feature_vol_l, feature_vol_r);
	using T_S = typename condImgRef<T_L, T_R, dDir>::T_S;
	using T_T = typename condImgRef<T_L, T_R, dDir>::T_T;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto l_shape = feature_vol_l.shape();
	auto r_shape = feature_vol_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 4>();
	}

	Multidim::Array<T_S, 3> & source_feature_volume = const_cast<Multidim::Array<T_S, 3> &>(dirInfos.source());
	Multidim::Array<T_T, 3> & target_feature_volume = const_cast<Multidim::Array<T_T, 3> &>(dirInfos.target());

	int h = source_feature_volume.shape()[0];
	int w = source_feature_volume.shape()[1];
	int f = source_feature_volume.shape()[2];

	int disp_height = searchRange.upperOffset<0>() - searchRange.lowerOffset<0>() + 1;
	int disp_width = searchRange.upperOffset<1>() - searchRange.lowerOffset<1>() + 1;

	if (disp_width <= 0 or disp_height <= 0) {
		return Multidim::Array<float, 4>();
	}

	Multidim::Array<TCV, 4> costVolume({h,w,disp_height,disp_width}, {w*disp_width*disp_height, 1, w*disp_width, w}); //TODO: check if those stides are otpimal

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {

			Multidim::Array<T_S, 1> source_feature_vector =
					source_feature_volume.subView(Multidim::DimIndex(i), Multidim::DimIndex(j), Multidim::DimSlice());

			for (int dh = 0; dh < disp_height; dh++) {

				for (int dw = 0; dw < disp_width; dw++) {

					Multidim::Array<T_T, 1> target_feature_vector(f);

					for (int c = 0; c < f; c++) {
						float t = target_feature_volume.valueOrAlt({i+dh+searchRange.lowerOffset<0>(),j+dw+searchRange.lowerOffset<1>(),c}, 0);
						target_feature_vector.template at<Nc>(c) = t;
					}

					costVolume.template at<Nc>(i,j,dh,dw) =
							MatchingFunctionTraits<matchFunc>::template featureComparison<T_S, T_T, TCV>(source_feature_vector, target_feature_vector);
				}

			}
		}
	}

	return costVolume;


}

template<class T_I, class T_M, class T_N, class T_O = float>
inline Multidim::Array<T_O, 1> zeromeanNormalizedFeatureVector(Multidim::Array<T_I, 1> const& feature_vol,
															   T_M const& mean,
															   T_N const& norm) {
	using T_E = TypesManipulations::accumulation_extended_t<T_I>;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int f = feature_vol.shape()[0];

	Multidim::Array<T_O, 1> normalized_feature_volume({f},{1});

	for (int c = 0; c < f; c++) {

		T_O val = 0;

		if (std::is_integral<T_O>::value) {
			T_E v = TypesManipulations::equivalentOneForNormalizing<T_E>();

			v *= static_cast<T_E>(feature_vol.template value<Nc>(c)) - static_cast<T_E>(mean);
			v /= norm;

			constexpr int diff = static_cast<int>(sizeof (T_E)) - static_cast<int>(sizeof (T_O));

			if (diff > 0) {
				v /= (1 << diff*8); //fit back into T_O
			}

			val = static_cast<T_O>(v);

		} else {
			val = static_cast<T_O>(feature_vol.template value<Nc>(c) - mean)/norm;
		}

		normalized_feature_volume.template at<Nc>(c) = val;
	}

	return normalized_feature_volume;

}

template<class T_I, class T_M, class T_N, class T_O = float>
inline Multidim::Array<T_O, 3> zeromeanNormalizedFeatureVolume(Multidim::Array<T_I, 3> const& feature_vol,
															   Multidim::Array<T_M, 2> const& mean,
															   Multidim::Array<T_N, 2> const& norm) {
	using T_E = TypesManipulations::accumulation_extended_t<T_I>;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = feature_vol.shape()[0];
	int w = feature_vol.shape()[1];
	int f = feature_vol.shape()[2];

	Multidim::Array<T_O, 3> normalized_feature_volume({h,w,f},{w*f,f,1});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {
			for (int c = 0; c < f; c++) {

				T_O val = 0;

				if (std::is_integral<T_O>::value) {
					T_E v = TypesManipulations::equivalentOneForNormalizing<T_E>();

					v *= static_cast<T_E>(feature_vol.template value<Nc>(i,j,c)) - static_cast<T_E>(mean.template value<Nc>(i,j));
					v /= norm.template value<Nc>(i,j);

					constexpr int diff = static_cast<int>(sizeof (T_E)) - static_cast<int>(sizeof (T_O));

					if (diff > 0) {
						v /= (1 << diff*8); //fit back into T_O
					}

					val = static_cast<T_O>(v);

				} else {
					val = static_cast<T_O>(feature_vol.template value<Nc>(i,j,c) - mean.template value<Nc>(i,j))/norm.template value<Nc>(i,j);
				}

				normalized_feature_volume.template at<Nc>(i,j,c) = val;
			}
		}
	}

	return normalized_feature_volume;

}

template<class T_I, class T_N, class T_O = float>
inline Multidim::Array<T_O, 1> normalizedFeatureVector(Multidim::Array<T_I, 1> const& feature_vol,
													   T_N const& norm) {

	using T_E = TypesManipulations::accumulation_extended_t<T_I>;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int f = feature_vol.shape()[0];

	Multidim::Array<T_O, 1> normalized_feature_vector({f},{1});

	for (int c = 0; c < f; c++) {

		T_O val = 0;

		if (std::is_integral<T_O>::value) {
			T_E v = TypesManipulations::equivalentOneForNormalizing<T_E>();

			v *= feature_vol.template value<Nc>(c);
			v /= norm;

			constexpr int diff = static_cast<int>(sizeof (T_E)) - static_cast<int>(sizeof (T_O));

			if (diff > 0) {
				v /= (2 << diff*8); //fit back into T_O
			}

			val = static_cast<T_O>(v);

		} else {
			val = static_cast<T_O>(feature_vol.template value<Nc>(c))/norm;
		}

		normalized_feature_vector.template at<Nc>(c) = val;
	}

	return normalized_feature_vector;

}

template<class T_I, class T_N, class T_O = float>
inline Multidim::Array<T_O, 3> normalizedFeatureVolume(Multidim::Array<T_I, 3> const& feature_vol,
													   Multidim::Array<T_N, 2> const& norm) {

	using T_E = TypesManipulations::accumulation_extended_t<T_I>;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = feature_vol.shape()[0];
	int w = feature_vol.shape()[1];
	int f = feature_vol.shape()[2];

	Multidim::Array<T_O, 3> normalized_feature_volume({h,w,f},{w*f,f,1});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {
			for (int c = 0; c < f; c++) {

				T_O val = 0;

				if (std::is_integral<T_O>::value) {
					T_E v = TypesManipulations::equivalentOneForNormalizing<T_E>();

					v *= feature_vol.template value<Nc>(i,j,c);
					v /= norm.template value<Nc>(i,j);

					constexpr int diff = static_cast<int>(sizeof (T_E)) - static_cast<int>(sizeof (T_O));

					if (diff > 0) {
						v /= (2 << diff*8); //fit back into T_O
					}

					val = static_cast<T_O>(v);

				} else {
					val = static_cast<T_O>(feature_vol.template value<Nc>(i,j,c))/norm.template value<Nc>(i,j);
				}

				normalized_feature_volume.template at<Nc>(i,j,c) = val;
			}
		}
	}

	return normalized_feature_volume;

}

template<class T_I, class T_M, class T_O = float>
inline Multidim::Array<T_O, 1> zeromeanFeatureVector(Multidim::Array<T_I, 1> const& feature_vol,
													 T_M const& mean) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int f = feature_vol.shape()[0];

	Multidim::Array<T_O, 1> zeromean_feature_vector({f},{1});

	for (int c = 0; c < f; c++) {
		zeromean_feature_vector.template at<Nc>(c) = static_cast<T_O>(feature_vol.template value<Nc>(c)) - static_cast<T_O>(mean);
	}

	return zeromean_feature_vector;

}

template<class T_I, class T_M, class T_O = float>
inline Multidim::Array<T_O, 3> zeromeanFeatureVolume(Multidim::Array<T_I, 3> const& feature_vol,
													   Multidim::Array<T_M, 2> const& mean) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = feature_vol.shape()[0];
	int w = feature_vol.shape()[1];
	int f = feature_vol.shape()[2];

	Multidim::Array<T_O, 3> zeromean_feature_volume({h,w,f},{w*f,f,1});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {
			for (int c = 0; c < f; c++) {
				zeromean_feature_volume.template at<Nc>(i,j,c) = static_cast<T_O>(feature_vol.template value<Nc>(i,j,c)) - static_cast<T_O>(mean.template value<Nc>(i,j));
			}
		}
	}

	return zeromean_feature_volume;

}

template<matchingFunctions matchFunc, class T_I>
Multidim::Array<FeatureTypeForMatchFunc<matchFunc, T_I>,1> getFeatureVectorForMatchFunc(Multidim::Array<T_I, 1> const& feature_vec) {

	using T_E = typename TypesManipulations::accumulation_extended_t<T_I>;

	constexpr bool CensusFeatures = MatchingFunctionTraits<matchFunc>::isCensusBased;

	using FType = FeatureTypeForMatchFunc<matchFunc, T_I>;

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		T_I mean = channelsMean<T_I, T_I>(feature_vec);
		T_E sigma = channelsZeroMeanNorm<T_I, T_I, T_E>(feature_vec, mean);

		if (CensusFeatures) {
			return censusFeatures(zeromeanNormalizedFeatureVector<T_I, T_I, T_E, T_E>(feature_vec, mean, sigma)).template cast<FType>();
		}

		return zeromeanNormalizedFeatureVector<T_I, T_I, T_E, FType>(feature_vec, mean, sigma);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		T_I mean = channelsMean<T_I, T_I>(feature_vec);

		if (CensusFeatures) {
			return censusFeatures(zeromeanFeatureVector<T_I, T_I, T_I>(feature_vec, mean)).template cast<FType>();
		}

		return zeromeanFeatureVector<T_I, T_I, FType>(feature_vec, mean);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		T_E norm = channelsNorm<T_I, T_E>(feature_vec);

		if (CensusFeatures) {
			return censusFeatures(normalizedFeatureVector<T_I, T_E, T_E>(feature_vec, norm)).template cast<FType>();
		}

		return normalizedFeatureVector<T_I, T_E, FType>(feature_vec, norm);
	}

	if (CensusFeatures) {
		return censusFeatures(feature_vec).template cast<FType>();
	}

	return feature_vec.template cast<FType>();

}

template<matchingFunctions matchFunc, class T_I>
Multidim::Array<FeatureTypeForMatchFunc<matchFunc, T_I>,3> getFeatureVolumeForMatchFunc(Multidim::Array<T_I, 3> const& feature_vol) {

	using T_E = typename TypesManipulations::accumulation_extended_t<T_I>;

	constexpr bool CensusFeatures = MatchingFunctionTraits<matchFunc>::isCensusBased;

	using FType = FeatureTypeForMatchFunc<matchFunc, T_I>;

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<T_I, 2> mean = channelsMean<T_I, T_I>(feature_vol);
		Multidim::Array<T_E, 2> sigma = channelsZeroMeanNorm<T_I, T_I, T_E>(feature_vol, mean);

		if (CensusFeatures) {
			return censusFeatures(zeromeanNormalizedFeatureVolume<T_I, T_I, T_E, T_E>(feature_vol, mean, sigma)).template cast<FType>();
		}

		return zeromeanNormalizedFeatureVolume<T_I, T_I, T_E, FType>(feature_vol, mean, sigma);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		Multidim::Array<T_I, 2> mean = channelsMean<T_I, T_I>(feature_vol);

		if (CensusFeatures) {
			return censusFeatures(zeromeanFeatureVolume<T_I, T_I, T_I>(feature_vol, mean)).template cast<FType>();
		}

		return zeromeanFeatureVolume<T_I, T_I, FType>(feature_vol, mean);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<T_E, 2> norm = channelsNorm<T_I, T_E>(feature_vol);

		if (CensusFeatures) {
			return censusFeatures(normalizedFeatureVolume<T_I, T_E, T_E>(feature_vol, norm)).template cast<FType>();
		}

		return normalizedFeatureVolume<T_I, T_E, FType>(feature_vol, norm);
	}

	if (CensusFeatures) {
		return censusFeatures(feature_vol).template cast<FType>();
	}

	return feature_vol.template cast<FType>();

}

template<matchingFunctions matchFunc, class T_L, class T_R, typename SearchRangeType, dispDirection dDir = dispDirection::RightToLeft, typename TCV = float>
inline Multidim::Array<TCV, searchRangeTypeInfos<SearchRangeType>::CostVolumeDims>
featureVolume2CostVolume(Multidim::Array<T_L, 3> const& feature_vol_l,
						 Multidim::Array<T_R, 3> const& feature_vol_r,
						 SearchRangeType searchRange) {

	using FTypeL = FeatureTypeForMatchFunc<matchFunc, T_L>;
	using FTypeR = FeatureTypeForMatchFunc<matchFunc, T_R>;

	return aggregateCost<matchFunc, FTypeL, FTypeR, dDir, TCV>
			(getFeatureVolumeForMatchFunc<matchFunc>(feature_vol_l),
			 getFeatureVolumeForMatchFunc<matchFunc>(feature_vol_r),
			 searchRange);

}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, typename TCV = float>
Multidim::Array<TCV, 3> unfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
											  Multidim::Array<T_R, nImDim> const& img_r,
											  uint8_t h_radius,
											  uint8_t v_radius,
											  disp_t disp_width)
{

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<TCV, 3>(0,0,0);
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<TCV, 3>(0,0,0);
		}
	}

	Multidim::Array<T_L, 3> left_feature_volume = unfold<T_L, T_L>(h_radius, v_radius, img_l);
	Multidim::Array<T_R, 3> right_feature_volume = unfold<T_R, T_R>(h_radius, v_radius, img_r);

	return featureVolume2CostVolume<matchFunc, T_L, T_R, disp_t, dDir, TCV>(left_feature_volume, right_feature_volume, disp_width);
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, typename TCV = float>
Multidim::Array<TCV, 3> unfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
											  Multidim::Array<T_R, nImDim> const& img_r,
											  UnFoldCompressor const& compressor,
											  disp_t disp_width)
{

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<TCV, 3>(0,0,0);
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<TCV, 3>(0,0,0);
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(compressor, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(compressor, img_r);

	return featureVolume2CostVolume<matchFunc, float, float, disp_t, dDir, TCV>(left_feature_volume, right_feature_volume, disp_width);
}


template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, typename TCV = float>
Multidim::Array<TCV, 4> unfoldBased2dDisparityCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
														 Multidim::Array<T_R, nImDim> const& img_r,
														 uint8_t h_radius,
														 uint8_t v_radius,
														 searchOffset<2> const& searchWindows) {

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<TCV, 4>();
	}

	if (l_shape[1] != r_shape[1]) {
		return Multidim::Array<TCV, 4>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<TCV, 4>();
		}
	}

	Multidim::Array<T_L, 3> left_feature_volume = unfold<T_L, T_L>(h_radius, v_radius, img_l);
	Multidim::Array<T_R, 3> right_feature_volume = unfold<T_R, T_R>(h_radius, v_radius, img_r);

	return featureVolume2CostVolume<matchFunc, T_L, T_R, searchOffset<2>, dDir, TCV>(left_feature_volume, right_feature_volume, searchWindows);
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, typename TCV = float>
Multidim::Array<TCV, 4> unfoldBased2dDisparityCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
														 Multidim::Array<T_R, nImDim> const& img_r,
														 UnFoldCompressor const& compressor,
														 searchOffset<2> const& searchWindows) {

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<TCV, 4>();
	}

	if (l_shape[1] != r_shape[1]) {
		return Multidim::Array<TCV, 4>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<TCV, 4>();
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(compressor, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(compressor, img_r);

	return featureVolume2CostVolume<matchFunc, float, float, searchOffset<2>, dDir, TCV>(left_feature_volume, right_feature_volume, searchWindows);
}


} // namespace Correlation
} // namespace StereoVision


#endif // STEREOVISION_CROSS_CORRELATIONS_H
