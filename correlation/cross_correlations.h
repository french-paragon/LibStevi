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

#include "./correlation_base.h"
#include "./cost_based_refinement.h"
#include "./unfold.h"

namespace StereoVision {
namespace Correlation {

enum class matchingFunctions{
	NCC = 0,
	SSD = 1,
	SAD = 2,
	ZNCC = 3,
	ZSSD = 4,
	ZSAD = 5
};

template<matchingFunctions func>
class MatchingFunctionTraits{
};

template<class T_S, class T_T>
inline float dotProduct(Multidim::Array<T_S,1> const& source,
						Multidim::Array<T_T,1> const& target) {

	float score = 0;

	for (int i = 0; i < source.shape()[0]; i++) {
		score += source.valueUnchecked(i)*target.valueUnchecked(i);
	}

	return score;

}

template<class T_S, class T_T>
inline float SumSquareDiff(Multidim::Array<T_S,1> const& source,
						   Multidim::Array<T_T,1> const& target) {

	float score = 0;

	for (int i = 0; i < source.shape()[0]; i++) {
		float tmp = source.valueUnchecked(i) - target.valueUnchecked(i);
		score += tmp*tmp;
	}

	return score;

}

template<class T_S, class T_T>
inline float SumAbsDiff(Multidim::Array<T_S,1> const& source,
						   Multidim::Array<T_T,1> const& target) {

	float score = 0;

	for (int i = 0; i < source.shape()[0]; i++) {
		float tmp = source.valueUnchecked(i) - target.valueUnchecked(i);
		score += tmp*tmp;
	}

	return score;

}

template<>
class MatchingFunctionTraits<matchingFunctions::NCC>{
public:
	static const std::string Name;
	static constexpr bool ZeroMean = false;
	static constexpr bool Normalized = true;
	static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Score;

	template<class T_S, class T_T>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return dotProduct(source, target);
	}
};
template<>
class MatchingFunctionTraits<matchingFunctions::SSD>{
public:
	static const std::string Name;
	static constexpr bool ZeroMean = false;
	static constexpr bool Normalized = false;
	static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

	template<class T_S, class T_T>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return SumSquareDiff(source, target);
	}
};
template<>
class MatchingFunctionTraits<matchingFunctions::SAD>{
public:
	static const std::string Name;
	static constexpr bool ZeroMean = false;
	static constexpr bool Normalized = false;
	static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

	template<class T_S, class T_T>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return SumAbsDiff(source, target);
	}
};

template<>
class MatchingFunctionTraits<matchingFunctions::ZNCC>{
public:
	static const std::string Name;
	static constexpr bool ZeroMean = true;
	static constexpr bool Normalized = true;
	static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Score;

	template<class T_S, class T_T>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return dotProduct(source, target);
	}
};
template<>
class MatchingFunctionTraits<matchingFunctions::ZSSD>{
public:
	static const std::string Name;
	static constexpr bool ZeroMean = true;
	static constexpr bool Normalized = false;
	static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

	template<class T_S, class T_T>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return SumSquareDiff(source, target);
	}
};
template<>
class MatchingFunctionTraits<matchingFunctions::ZSAD>{

	static const std::string Name;
	static constexpr bool ZeroMean = true;
	static constexpr bool Normalized = false;
	static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

	template<class T_S, class T_T>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return SumAbsDiff(source, target);
	}
};

template<class T_I>
Multidim::Array<float, 2> channelsSigma (Multidim::Array<T_I, 3> const& in_data,
									   Multidim::Array<float, 2> const& mean) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = in_data.shape()[0];
	int w = in_data.shape()[1];
	int f = in_data.shape()[2];

	Multidim::Array<float, 2> std(h, w);

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {

			std.at<Nc>(i,j) = 0;

			for (int c = 0; c < f; c++) {
				float tmp = static_cast<float>(in_data.template value<Nc>(i,j,c)) - mean.value<Nc>(i,j);
				std.at<Nc>(i,j) += tmp*tmp;
			}
		}
	}

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {
			std.at<Nc>(i,j) = sqrtf(std.value<Nc>(i,j));
		}
	}

	return std;
}

template<class T_I>
Multidim::Array<float, 2> channelsSigma (Multidim::Array<T_I, 3> const& in_data) {

	Multidim::Array<float, 2> mean = channelsMean(in_data);

	return channelsSigma(in_data, mean);

}


template<class T_I>
Multidim::Array<float, 2> channelsNorm (Multidim::Array<T_I, 3> const& in_data) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = in_data.shape()[0];
	int w = in_data.shape()[1];
	int f = in_data.shape()[2];

	Multidim::Array<float, 2> std(h, w);

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {

			std.at<Nc>(i,j) = 0;

			for (int c = 0; c < f; c++) {
				float tmp = static_cast<float>(in_data.template value<Nc>(i,j,c));
				std.at<Nc>(i,j) += tmp*tmp;
			}
		}
	}

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {
			std.at<Nc>(i,j) = sqrtf(std.value<Nc>(i,j));
		}
	}

	return std;
}

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft>
inline Multidim::Array<float, 3> aggregateCost(Multidim::Array<float, 3> const& feature_vol_l,
											   Multidim::Array<float, 3> const& feature_vol_r,
											   disp_t disp_width) {


	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;
	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

	auto l_shape = feature_vol_l.shape();
	auto r_shape = feature_vol_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>(0,0,0);
	}

	constexpr bool r2l = dDir == dispDirection::RightToLeft;
	Multidim::Array<float, 3> & source_feature_volume = const_cast<Multidim::Array<float, 3> &>((r2l) ? feature_vol_r : feature_vol_l);
	Multidim::Array<float, 3> & target_feature_volume = const_cast<Multidim::Array<float, 3> &>((r2l) ? feature_vol_l : feature_vol_r);

	int h = source_feature_volume.shape()[0];
	int w = source_feature_volume.shape()[1];
	int f = source_feature_volume.shape()[2];

	Multidim::Array<float, 3> costVolume({h,w,disp_width}, {w*disp_width, 1, w});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {

			for (int d = 0; d < disp_width; d++) {

				Multidim::Array<float, 1> source_feature_vector(f);
				Multidim::Array<float, 1> target_feature_vector(f);

				for (int c = 0; c < f; c++) {
					float s = source_feature_volume.value<Nc>(i,j,c);
					float t = target_feature_volume.valueOrAlt({i,j+deltaSign*d,c}, 0);

					source_feature_vector.at<Nc>(c) = s;
					target_feature_vector.at<Nc>(c) = t;
				}

				costVolume.at<Nc>(i,j,d) = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector);

			}
		}
	}

	return costVolume;


}

inline Multidim::Array<float, 3> zeromeanNormalizedFeatureVolume(Multidim::Array<float, 3> const& feature_vol,
																 Multidim::Array<float, 2> const& mean,
																 Multidim::Array<float, 2> const& sigma) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = feature_vol.shape()[0];
	int w = feature_vol.shape()[1];
	int f = feature_vol.shape()[2];

	Multidim::Array<float, 3> normalized_feature_volume({h,w,f},{w*f,f,1});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {
			for (int c = 0; c < f; c++) {
				normalized_feature_volume.at<Nc>(i,j,c) = (feature_vol.value<Nc>(i,j,c) - mean.value<Nc>(i,j))/sigma.value<Nc>(i,j);
			}
		}
	}

	return normalized_feature_volume;

}

inline Multidim::Array<float, 3> normalizedFeatureVolume(Multidim::Array<float, 3> const& feature_vol,
														 Multidim::Array<float, 2> const& norm) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = feature_vol.shape()[0];
	int w = feature_vol.shape()[1];
	int f = feature_vol.shape()[2];

	Multidim::Array<float, 3> normalized_feature_volume({h,w,f},{w*f,f,1});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {
			for (int c = 0; c < f; c++) {
				normalized_feature_volume.at<Nc>(i,j,c) = feature_vol.value<Nc>(i,j,c)/norm.value<Nc>(i,j);
			}
		}
	}

	return normalized_feature_volume;



}

inline Multidim::Array<float, 3> zeromeanFeatureVolume(Multidim::Array<float, 3> const& feature_vol,
														  Multidim::Array<float, 2> const& mean) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = feature_vol.shape()[0];
	int w = feature_vol.shape()[1];
	int f = feature_vol.shape()[2];

	Multidim::Array<float, 3> zeromean_feature_volume({h,w,f},{w*f,f,1});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {
			for (int c = 0; c < f; c++) {
				zeromean_feature_volume.at<Nc>(i,j,c) = feature_vol.value<Nc>(i,j,c) - mean.value<Nc>(i,j);
			}
		}
	}

	return zeromean_feature_volume;

}

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft>
inline Multidim::Array<float, 3> featureVolume2CostVolume(Multidim::Array<float, 3> const& feature_vol_l,
														  Multidim::Array<float, 3> const& feature_vol_r,
														  disp_t disp_width) {

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 2> sigma_left = channelsSigma(feature_vol_l, mean_left);
		Multidim::Array<float, 2> sigma_right = channelsSigma(feature_vol_r, mean_right);

		Multidim::Array<float, 3> normalized_feature_volume_l = zeromeanNormalizedFeatureVolume(feature_vol_l, mean_left, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = zeromeanNormalizedFeatureVolume(feature_vol_r, mean_right, sigma_right);

		return aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp_width);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		return aggregateCost<matchFunc, dDir>(feature_vol_l, feature_vol_r, disp_width);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> sigma_left = channelsNorm(feature_vol_l);
		Multidim::Array<float, 2> sigma_right = channelsNorm(feature_vol_r);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(feature_vol_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(feature_vol_r, sigma_right);

		return aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp_width);

	} else {
		return aggregateCost<matchFunc, dDir>(feature_vol_l, feature_vol_r, disp_width);
	}

	return Multidim::Array<float, 3>();

}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> unfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
												Multidim::Array<T_R, nImDim> const& img_r,
												uint8_t h_radius,
												uint8_t v_radius,
												disp_t disp_width)
{

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>(0,0,0);
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 3>(0,0,0);
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(h_radius, v_radius, img_r);

	return featureVolume2CostVolume<matchFunc, dDir>(left_feature_volume, right_feature_volume, disp_width);
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> unfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
												Multidim::Array<T_R, nImDim> const& img_r,
												UnFoldCompressor const& compressor,
												disp_t disp_width)
{

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>(0,0,0);
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 3>(0,0,0);
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(compressor, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(compressor, img_r);

	return featureVolume2CostVolume<matchFunc, dDir>(left_feature_volume, right_feature_volume, disp_width);
}

} // namespace Correlation
} // namespace StereoVision


#endif // STEREOVISION_CROSS_CORRELATIONS_H
