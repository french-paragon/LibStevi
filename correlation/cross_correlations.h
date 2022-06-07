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

#include "../utils/contiguity.h"

namespace StereoVision {
namespace Correlation {

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

			Multidim::Array<float, 1> source_feature_vector(f);

			for (int c = 0; c < f; c++) {
				float s = source_feature_volume.value<Nc>(i,j,c);
				source_feature_vector.at<Nc>(c) = s;
			}

			for (int d = 0; d < disp_width; d++) {

				Multidim::Array<float, 1> target_feature_vector(f);

				for (int c = 0; c < f; c++) {
					float t = target_feature_volume.valueOrAlt({i,j+deltaSign*d,c}, 0);
					target_feature_vector.at<Nc>(c) = t;
				}

				costVolume.at<Nc>(i,j,d) = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector);

			}
		}
	}

	return costVolume;


}

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft>
inline Multidim::Array<float, 4> aggregateCost(Multidim::Array<float, 3> const& feature_vol_l,
											   Multidim::Array<float, 3> const& feature_vol_r,
											   searchOffset<2> searchRange) {


	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto l_shape = feature_vol_l.shape();
	auto r_shape = feature_vol_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 4>();
	}

	constexpr bool r2l = dDir == dispDirection::RightToLeft;
	Multidim::Array<float, 3> & source_feature_volume = const_cast<Multidim::Array<float, 3> &>((r2l) ? feature_vol_r : feature_vol_l);
	Multidim::Array<float, 3> & target_feature_volume = const_cast<Multidim::Array<float, 3> &>((r2l) ? feature_vol_l : feature_vol_r);

	int h = source_feature_volume.shape()[0];
	int w = source_feature_volume.shape()[1];
	int f = source_feature_volume.shape()[2];

	int disp_height = searchRange.upperOffset<0>() - searchRange.lowerOffset<0>() + 1;
	int disp_width = searchRange.upperOffset<1>() - searchRange.lowerOffset<1>() + 1;

	if (disp_width <= 0 or disp_height <= 0) {
		return Multidim::Array<float, 4>();
	}

	Multidim::Array<float, 4> costVolume({h,w,disp_height,disp_width}, {w*disp_width*disp_height, 1, w*disp_width, w}); //TODO: check if those stides are otpimal

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {

			Multidim::Array<float, 1> source_feature_vector(f);

			for (int c = 0; c < f; c++) {
				float s = source_feature_volume.value<Nc>(i,j,c);
				source_feature_vector.at<Nc>(c) = s;
			}

			for (int dh = 0; dh < disp_height; dh++) {

				for (int dw = 0; dw < disp_width; dw++) {

					Multidim::Array<float, 1> target_feature_vector(f);

					for (int c = 0; c < f; c++) {
						float t = target_feature_volume.valueOrAlt({i+dh+searchRange.lowerOffset<0>(),j+dw+searchRange.lowerOffset<1>(),c}, 0);
						target_feature_vector.at<Nc>(c) = t;
					}

					costVolume.at<Nc>(i,j,dh,dw) = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector);
				}

			}
		}
	}

	return costVolume;


}

template<class T_I>
inline Multidim::Array<float, 3> zeromeanNormalizedFeatureVolume(Multidim::Array<T_I, 3> const& feature_vol,
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
				normalized_feature_volume.at<Nc>(i,j,c) = (feature_vol.template value<Nc>(i,j,c) - mean.value<Nc>(i,j))/sigma.value<Nc>(i,j);
			}
		}
	}

	return normalized_feature_volume;

}

template<class T_I>
inline Multidim::Array<float, 3> normalizedFeatureVolume(Multidim::Array<T_I, 3> const& feature_vol,
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
				normalized_feature_volume.at<Nc>(i,j,c) = static_cast<float>(feature_vol.template value<Nc>(i,j,c))/norm.value<Nc>(i,j);
			}
		}
	}

	return normalized_feature_volume;



}

template<class T_I>
inline Multidim::Array<float, 3> zeromeanFeatureVolume(Multidim::Array<T_I, 3> const& feature_vol,
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
				zeromean_feature_volume.at<Nc>(i,j,c) = feature_vol.template value<Nc>(i,j,c) - mean.value<Nc>(i,j);
			}
		}
	}

	return zeromean_feature_volume;

}

template<matchingFunctions matchFunc, class T_I>
Multidim::Array<float,3> getFeatureVolumeForMatchFunc(Multidim::Array<T_I, 3> const& feature_vol) {

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> mean = channelsMean(feature_vol);
		Multidim::Array<float, 2> sigma = channelsSigma(feature_vol, mean);

		return zeromeanNormalizedFeatureVolume(feature_vol, mean, sigma);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		Multidim::Array<float, 2> mean = channelsMean(feature_vol);

		return zeromeanFeatureVolume(feature_vol, mean);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> sigma = channelsNorm(feature_vol);

		return normalizedFeatureVolume(feature_vol, sigma);
	}

	Multidim::Array<float,3> fv(feature_vol.shape());

	#pragma omp parallel for
	for (int i = 0; i < fv.shape()[0]; i++) {
		for (int j = 0; j < fv.shape()[1]; j++) {
			for (int k = 0; k < fv.shape()[2]; k++) {
				fv.atUnchecked(i,j,k) = static_cast<float>(feature_vol.valueUnchecked(i,j,k));
			}
		}
	}

	return fv;

}

template<matchingFunctions matchFunc, typename SearchRangeType, dispDirection dDir = dispDirection::RightToLeft>
inline Multidim::Array<float, searchRangeTypeInfos<SearchRangeType>::CostVolumeDims>
featureVolume2CostVolume(Multidim::Array<float, 3> const& feature_vol_l,
						 Multidim::Array<float, 3> const& feature_vol_r,
						 SearchRangeType searchRange) {

	return aggregateCost<matchFunc, dDir>(getFeatureVolumeForMatchFunc<matchFunc>(feature_vol_l),
										  getFeatureVolumeForMatchFunc<matchFunc>(feature_vol_r),
										  searchRange);

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

	return featureVolume2CostVolume<matchFunc, disp_t, dDir>(left_feature_volume, right_feature_volume, disp_width);
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

	return featureVolume2CostVolume<matchFunc, disp_t, dDir>(left_feature_volume, right_feature_volume, disp_width);
}


template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 4> unfoldBased2dDisparityCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
														   Multidim::Array<T_R, nImDim> const& img_r,
														   uint8_t h_radius,
														   uint8_t v_radius,
														   searchOffset<2> const& searchWindows) {

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 4>();
	}

	if (l_shape[1] != r_shape[1]) {
		return Multidim::Array<float, 4>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 4>();
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(h_radius, v_radius, img_r);

	return featureVolume2CostVolume<matchFunc, searchOffset<2>, dDir>(left_feature_volume, right_feature_volume, searchWindows);
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 4> unfoldBased2dDisparityCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
														   Multidim::Array<T_R, nImDim> const& img_r,
														   UnFoldCompressor const& compressor,
														   searchOffset<2> const& searchWindows) {

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 4>();
	}

	if (l_shape[1] != r_shape[1]) {
		return Multidim::Array<float, 4>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 4>();
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(compressor, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(compressor, img_r);

	return featureVolume2CostVolume<matchFunc, searchOffset<2>, dDir>(left_feature_volume, right_feature_volume, searchWindows);
}

template<matchingFunctions matchFunc, int refineRadius = 1, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refineBarycentricSymmetricDisp(Multidim::Array<float, 3> const& feature_vol_l,
														 Multidim::Array<float, 3> const& feature_vol_r,
														 Multidim::Array<disp_t, 2> const& selectedIndex,
														 disp_t disp_width) {
	static_assert (refineRadius > 0, "refineBarycentricSymmetricDisp cannot proceed with a refinement radius smaller than 1.");

	typedef Eigen::Matrix<float, Eigen::Dynamic, 2*refineRadius+1> TypeMatrixA;
	typedef Eigen::Matrix<float, Eigen::Dynamic, 2*refineRadius> TypeMatrixM;

	typedef Eigen::Matrix<float, 2*refineRadius+1, 1> TypeVectorAlpha;

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 3> const& source_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_r : feature_vol_l;
	Multidim::Array<float, 3> const& target_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_l : feature_vol_r;

	auto d_shape = selectedIndex.shape();
	auto t_shape = target_feature_volume.shape();

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t d = selectedIndex.value<Nc>(i,j);

			int jd = j + deltaSign*d;

			if (j < 0 or j + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (jd - refineRadius < 0 or jd + 1 > d_shape[1] - refineRadius) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (d == 0 or d+1 >= disp_width) {
				refinedDisp.at<Nc>(i,j) = d;
			} else {

				Eigen::VectorXf source(t_shape[2]);

				for (int c = 0 ; c < t_shape[2]; c++) {
					source(c) = source_feature_volume.value<Nc>(i,j,c);
				}

				TypeMatrixA A(t_shape[2],2*refineRadius+1);

				for (int p = -refineRadius; p <= refineRadius; p++) {
					for (int c = 0 ; c < t_shape[2]; c++) {
						A(c, p+refineRadius) = target_feature_volume.value<Nc>(i,jd+p,c);
					}
				}

				TypeVectorAlpha coeffs = MatchingFunctionTraits<matchFunc>::barycentricBestApproximation(A, source);

				float delta_d = 0;
				for (int p = -refineRadius; p <= refineRadius; p++) {
					delta_d += coeffs(p+refineRadius)*float(p);
				}

				if (std::fabs(delta_d) < 1) { //subpixel adjustement is in the interval
					refinedDisp.at<Nc>(i,j) = d + delta_d;
				} else {
					refinedDisp.at<Nc>(i,j) = d;
				}
			}

		}

	}

	return refinedDisp;
}

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refineBarycentricDisp(Multidim::Array<float, 3> const& feature_vol_l,
												Multidim::Array<float, 3> const& feature_vol_r,
												Multidim::Array<disp_t, 2> const& selectedIndex) {

	typedef Eigen::Matrix<float, Eigen::Dynamic, 2> TypeMatrixA;
	typedef Eigen::Matrix<float, 2, 1> TypeVectorAlpha;

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 3> const& source_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_r : feature_vol_l;
	Multidim::Array<float, 3> const& target_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_l : feature_vol_r;

	auto d_shape = selectedIndex.shape();
	auto t_shape = target_feature_volume.shape();

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t d = selectedIndex.value<Nc>(i,j);

			int jd = j + deltaSign*d;

			if (jd < 1 or jd + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else {

				int f = t_shape[2];

				Eigen::VectorXf source(f);

				for (int c = 0 ; c < f; c++) {
					source(c) = source_feature_volume.value<Nc>(i,j,c);
				}

				TypeMatrixA Ap(f,2);
				TypeMatrixA Am(f,2);

				for (int c = 0 ; c < f; c++) {
					Ap(c, 0) = target_feature_volume.value<Nc>(i,jd,c);
					Ap(c, 1) = target_feature_volume.value<Nc>(i,jd+1,c);
				}

				for (int c = 0 ; c < f; c++) {
					Am(c, 0) = target_feature_volume.value<Nc>(i,jd-1,c);
					Am(c, 1) = target_feature_volume.value<Nc>(i,jd,c);
				}

				Multidim::Array<float, 1> source_feature_vector(f);
				float norm = source.norm();

				for (int c = 0; c < f; c++) {
					float s = source_feature_volume.value<Nc>(i,j,c);
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						source_feature_vector.at<Nc>(c) = s/norm;
					}
					else {
						source_feature_vector.at<Nc>(c) = s;
					}
				}

				Multidim::Array<float, 1> target_feature_vector0(f);
				norm = Ap.col(0).norm();

				for (int c = 0; c < f; c++) {
					float s = target_feature_volume.value<Nc>(i,jd,c);
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						target_feature_vector0.at<Nc>(c) = s/norm;
					}
					else {
						target_feature_vector0.at<Nc>(c) = s;
					}
				}

				TypeVectorAlpha coeffsP = MatchingFunctionTraits<matchFunc>::barycentricBestApproximation(Ap, source);
				TypeVectorAlpha coeffsM = MatchingFunctionTraits<matchFunc>::barycentricBestApproximation(Am, source);

				float DeltaD_plus = coeffsP(1);
				float DeltaD_minus = coeffsM(0);

				Eigen::VectorXf src = source;

				if (MatchingFunctionTraits<matchFunc>::Normalized) {
					src.normalize();
				}

				Multidim::Array<float,1> srcArr(&src(0),{int(src.rows())},{int(src.stride())});

				float score = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector0);

				float DeltaD = 0;

				if (DeltaD_plus > 0 and DeltaD_plus < 1) {

					Eigen::VectorXf interpFeaturesPlus = Ap*coeffsP;
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						interpFeaturesPlus.normalize();
					}

					Multidim::Array<float, 1> target_feature_vector_interp(&interpFeaturesPlus(0),{f},{1});

					float tmpScore = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector_interp);

					if (MatchingFunctionTraits<matchFunc>::extractionStrategy == dispExtractionStartegy::Score) {
						if (tmpScore > score) {
							score = tmpScore;
							DeltaD = DeltaD_plus;
						}
					} else {
						if (tmpScore < score) {
							score = tmpScore;
							DeltaD = DeltaD_plus;
						}
					}

				}

				if (DeltaD_minus > 0 and DeltaD_minus < 1) {

					Eigen::VectorXf interpFeaturesMinus = Am*coeffsM;
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						interpFeaturesMinus.normalize();
					}

					Multidim::Array<float, 1> target_feature_vector_interp(&interpFeaturesMinus(0),{f},{1});

					float tmpScore = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector_interp);

					if (MatchingFunctionTraits<matchFunc>::extractionStrategy == dispExtractionStartegy::Score) {
						if (tmpScore > score) {
							score = tmpScore;
							DeltaD = -DeltaD_minus;
						}
					} else {
						if (tmpScore < score) {
							score = tmpScore;
							DeltaD = -DeltaD_minus;
						}
					}

				}

				refinedDisp.at<Nc>(i,j) = d + deltaSign*DeltaD;

			}

		}
	}

	return refinedDisp;
}

template<matchingFunctions matchFunc,
		 Contiguity::bidimensionalContiguity contiguity = Contiguity::Queen,
		 dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> refineBarycentric2dDisp(Multidim::Array<float, 3> const& feature_vol_l,
												  Multidim::Array<float, 3> const& feature_vol_r,
												  Multidim::Array<disp_t, 3> const& selectedIndices,
												  searchOffset<2> const& searchWindows) {

	constexpr int nDirs = Contiguity::nCornerDirections(contiguity)+1;

	typedef Eigen::Matrix<float, Eigen::Dynamic, nDirs> TypeMatrixA;
	typedef Eigen::Matrix<float, nDirs, 1> TypeVectorAlpha;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 3> const& source_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_r : feature_vol_l;
	Multidim::Array<float, 3> const& target_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_l : feature_vol_r;

	auto d_shape = selectedIndices.shape();
	auto t_shape = target_feature_volume.shape();

	Multidim::Array<float, 3> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t d0 = selectedIndices.value<Nc>(i,j,0);
			disp_t d1 = selectedIndices.value<Nc>(i,j,1);

			int id = i + d0;
			int jd = j + d1;

			if (id < 1 or id + 1 >= d_shape[0]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j,0) = d0;
				refinedDisp.at<Nc>(i,j,1) = d1;
			} else if (jd < 1 or jd + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j,0) = d0;
				refinedDisp.at<Nc>(i,j,1) = d1;
			}  else if (d0 < searchWindows.lowerOffset<0>() or d0 > searchWindows.upperOffset<0>()) {
				refinedDisp.at<Nc>(i,j,0) = d0;
				refinedDisp.at<Nc>(i,j,1) = d1;
			}   else if (d1 < searchWindows.lowerOffset<1>() or d1 > searchWindows.upperOffset<1>()) {
				refinedDisp.at<Nc>(i,j,0) = d0;
				refinedDisp.at<Nc>(i,j,1) = d1;
			} else {

				int f = t_shape[2];

				Eigen::VectorXf source(f);

				for (int c = 0 ; c < f; c++) {
					source(c) = source_feature_volume.value<Nc>(i,j,c);
				}

				Multidim::Array<float, 1> source_feature_vector(f);
				float norm = source.norm();

				for (int c = 0; c < f; c++) {
					float s = source_feature_volume.value<Nc>(i,j,c);
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						source_feature_vector.at<Nc>(c) = s/norm;
					}
					else {
						source_feature_vector.at<Nc>(c) = s;
					}
				}

				Multidim::Array<float, 1> target_feature_vector0(f);
				norm = 1;
				if (MatchingFunctionTraits<matchFunc>::Normalized) {
					norm = 0;
					for (int c = 0; c < f; c++) {
						float s = target_feature_volume.value<Nc>(id,jd,c);
						norm += s*s;
					}
					norm = std::sqrt(norm);
				}

				for (int c = 0; c < f; c++) {
					float s = target_feature_volume.value<Nc>(id,jd,c);
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						target_feature_vector0.at<Nc>(c) = s/norm;
					}
					else {
						target_feature_vector0.at<Nc>(c) = s;
					}
				}

				float score = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector0);

				constexpr std::array<int,2> dirShifts = {1,-1};
				constexpr auto searchDirs = Contiguity::getCornerDirections<contiguity>();

				float deltaD0 = 0;
				float deltaD1 = 0;

				for (int dir_x : dirShifts) {
					for (int dir_y : dirShifts) {

						TypeMatrixA A(f,nDirs);

						int i = 0;
						for (auto sDir : searchDirs) {
							int di = sDir[0]*dir_x;
							int dj = sDir[1]*dir_y;

							for (int c = 0 ; c < f; c++) {
								A(c,i) = target_feature_volume.value<Nc>(id+di,jd+dj,c);
							}

							i++;
						}

						for (int c = 0 ; c < f; c++) {
							A(c,i) = target_feature_volume.value<Nc>(id,jd,c);
						}

						TypeVectorAlpha alphas = MatchingFunctionTraits<matchFunc>::barycentricBestApproximation(A, source);

						float tmp_deltaD0 = 0;
						float tmp_deltaD1 = 0;

						int pos = 0;
						for (auto sDir : searchDirs) {
							int di = sDir[0]*dir_x;
							int dj = sDir[1]*dir_y;

							tmp_deltaD0 += alphas(pos)*di;
							tmp_deltaD1 += alphas(pos)*dj;

							pos++;
						}

						if ( std::fabs(tmp_deltaD0) <= 1. and std::fabs(tmp_deltaD1) <= 1. ) { // consider the interpolation only if the interpolated position is in the area of interest.
							//Note that if the search space is a 2d simplex, this is true if all barycentric coordinates are greather than or equal 0, but in the general case this is not true.

							Eigen::VectorXf interpFeatures = A*alphas;
							if (MatchingFunctionTraits<matchFunc>::Normalized) {
								interpFeatures.normalize();
							}

							Multidim::Array<float, 1> target_feature_vector_interp(&interpFeatures(0),{f},{1});

							float tmpScore = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector_interp);

							bool better = false;

							if (MatchingFunctionTraits<matchFunc>::extractionStrategy == dispExtractionStartegy::Score) {
								if (tmpScore > score) {
									better = true;
								}
							} else {
								if (tmpScore < score) {
									better = true;
								}
							}

							if (better) {

								deltaD0 = tmp_deltaD0;
								deltaD1 = tmp_deltaD1;

								score = tmpScore;

							}
						}
					}

				}

				refinedDisp.at<Nc>(i,j,0) = d0 + deltaD0;
				refinedDisp.at<Nc>(i,j,1) = d1 + deltaD1;

			}

		}
	}

	return refinedDisp;
}

template<matchingFunctions matchFunc,
		 Contiguity::bidimensionalContiguity contiguity = Contiguity::Queen,
		 dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> refineBarycentricSymmetric2dDisp(Multidim::Array<float, 3> const& feature_vol_l,
														   Multidim::Array<float, 3> const& feature_vol_r,
														   Multidim::Array<disp_t, 3> const& selectedIndices,
														   searchOffset<2> const& searchWindows) {

	constexpr int nDirs = Contiguity::nDirections(contiguity)+1;

	typedef Eigen::Matrix<float, Eigen::Dynamic, nDirs> TypeMatrixA;
	typedef Eigen::Matrix<float, nDirs, 1> TypeVectorAlpha;

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 3> const& source_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_r : feature_vol_l;
	Multidim::Array<float, 3> const& target_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_l : feature_vol_r;

	auto d_shape = selectedIndices.shape();
	auto t_shape = target_feature_volume.shape();

	Multidim::Array<float, 3> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t d0 = selectedIndices.value<Nc>(i,j,0);
			disp_t d1 = selectedIndices.value<Nc>(i,j,1);

			int id = i + d0;
			int jd = j + d1;

			if (id < 1 or id + 1 >= d_shape[0]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j,0) = d0;
				refinedDisp.at<Nc>(i,j,1) = d1;
			} else if (jd < 1 or jd + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j,0) = d0;
				refinedDisp.at<Nc>(i,j,1) = d1;
			}  else if (d0 < searchWindows.lowerOffset<0>() or d0 > searchWindows.upperOffset<0>()) {
				refinedDisp.at<Nc>(i,j,0) = d0;
				refinedDisp.at<Nc>(i,j,1) = d1;
			}   else if (d1 < searchWindows.lowerOffset<1>() or d1 > searchWindows.upperOffset<1>()) {
				refinedDisp.at<Nc>(i,j,0) = d0;
				refinedDisp.at<Nc>(i,j,1) = d1;
			} else {

				int f = t_shape[2];

				Eigen::VectorXf source(f);

				for (int c = 0 ; c < f; c++) {
					source(c) = source_feature_volume.value<Nc>(i,j,c);
				}

				Multidim::Array<float, 1> source_feature_vector(f);
				float norm = source.norm();

				for (int c = 0; c < f; c++) {
					float s = source_feature_volume.value<Nc>(i,j,c);
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						source_feature_vector.at<Nc>(c) = s/norm;
					}
					else {
						source_feature_vector.at<Nc>(c) = s;
					}
				}

				Multidim::Array<float, 1> target_feature_vector0(f);
				norm = 1;
				if (MatchingFunctionTraits<matchFunc>::Normalized) {
					norm = 0;
					for (int c = 0; c < f; c++) {
						float s = target_feature_volume.value<Nc>(id,jd,c);
						norm += s*s;
					}
					norm = std::sqrt(norm);
				}

				for (int c = 0; c < f; c++) {
					float s = target_feature_volume.value<Nc>(id,jd,c);
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						target_feature_vector0.at<Nc>(c) = s/norm;
					}
					else {
						target_feature_vector0.at<Nc>(c) = s;
					}
				}

				float score = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector0);

				constexpr auto searchDirs = Contiguity::getDirections<contiguity>();

				float deltaD0 = 0;
				float deltaD1 = 0;

				TypeMatrixA A(f,nDirs);

				int col = 0;
				for (auto sDir : searchDirs) {
					int di = sDir[0];
					int dj = sDir[1];

					for (int c = 0 ; c < f; c++) {
						A(c,col) = target_feature_volume.value<Nc>(id+di,jd+dj,c);
					}

					col++;
				}

				for (int c = 0 ; c < f; c++) {
					A(c,col) = target_feature_volume.value<Nc>(id,jd,c);
				}

				TypeVectorAlpha alphas = MatchingFunctionTraits<matchFunc>::barycentricBestApproximation(A, source);

				float tmp_deltaD0 = 0;
				float tmp_deltaD1 = 0;

				int pos = 0;
				for (auto sDir : searchDirs) {
					int di = sDir[0];
					int dj = sDir[1];

					tmp_deltaD0 += alphas(pos)*di;
					tmp_deltaD1 += alphas(pos)*dj;

					pos++;
				}

				if ( std::fabs(tmp_deltaD0) <= 1. and std::fabs(tmp_deltaD1) <= 1. ) { // consider the interpolation only if the barycentric coordinates are all greather than 0.
					Eigen::VectorXf interpFeatures = A*alphas;
					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						interpFeatures.normalize();
					}

					Multidim::Array<float, 1> target_feature_vector_interp(&interpFeatures(0),{f},{1});

					float tmpScore = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector_interp);

					bool better = false;

					if (MatchingFunctionTraits<matchFunc>::extractionStrategy == dispExtractionStartegy::Score) {
						if (tmpScore > score) {
							better = true;
						}
					} else {
						if (tmpScore < score) {
							better = true;
						}
					}

					if (better) {

						deltaD0 = tmp_deltaD0;
						deltaD1 = tmp_deltaD1;

						score = tmpScore;

					}
				}

				refinedDisp.at<Nc>(i,j,0) = d0 + deltaD0;
				refinedDisp.at<Nc>(i,j,1) = d1 + deltaD1;

			}

		}
	}

	return refinedDisp;
}

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refineCostSymmetricDisp(Multidim::Array<float, 3> const& feature_vol_l,
												  Multidim::Array<float, 3> const& feature_vol_r,
												  Multidim::Array<disp_t, 2> const& selectedIndex,
												  Multidim::Array<float, 3> const& cost_volume) {

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 3> const& source_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_r : feature_vol_l;
	Multidim::Array<float, 3> const& target_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_l : feature_vol_r;

	auto d_shape = selectedIndex.shape();
	auto t_shape = target_feature_volume.shape();
	auto cv_shape = cost_volume.shape();

	int f = t_shape[2];

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t d = selectedIndex.value<Nc>(i,j);

			int jd = j + deltaSign*d;

			float delta = 0;

			if (j > 1 and j + 1 < cv_shape[1] and d > 0 and d+1 < cv_shape[2]) {

				float cm1 = cost_volume.value<Nc>(i,j,d-1);
				float c0 = cost_volume.value<Nc>(i,j,d);
				float c1 = cost_volume.value<Nc>(i,j,d+1);

				delta = (cm1 - c1)/(2*(c1 - 2*c0 + cm1));

				disp_t dir = 1;

				if (delta > 0) {
					dir = -1;
				}

				if (jd + 1 < cv_shape[1] and jd > 1) {

					Eigen::VectorXf sourceInterp(f);

					for (int c = 0 ; c < f; c++) {
						sourceInterp(c) = 0.5*source_feature_volume.value<Nc>(i,j,c) + 0.5*source_feature_volume.value<Nc>(i,j+dir,c);
					}

					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						sourceInterp.normalize();
					}

					Eigen::VectorXf targetm1(f);
					Eigen::VectorXf target0(f);
					Eigen::VectorXf target1(f);


					for (int c = 0 ; c < f; c++) {
						targetm1(c) = target_feature_volume.value<Nc>(i,jd-1,c);
						target0(c) = target_feature_volume.value<Nc>(i,jd,c);
						target1(c) = target_feature_volume.value<Nc>(i,jd+1,c);
					}

					if (MatchingFunctionTraits<matchFunc>::Normalized) {
						targetm1.normalize();
						target0.normalize();
						target1.normalize();
					}


					Multidim::Array<float, 1> source_feature_vector(&sourceInterp(0),{f},{1});
					Multidim::Array<float, 1> target_feature_vectorm1(&targetm1(0),{f},{1});
					Multidim::Array<float, 1> target_feature_vector0(&target0(0),{f},{1});
					Multidim::Array<float, 1> target_feature_vector1(&target1(0),{f},{1});

					float fm1 = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vectorm1);
					float f0 = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector0);
					float f1 = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector1);

					float delta2 = (fm1 - f1)/(2*(f1 - 2*f0 + fm1)) - dir*0.5;

					if (std::fabs(delta2) < 1.) {
						delta = (delta + delta2)/2;
					}

				}
			}

			refinedDisp.at<Nc>(i,j) = d + delta;
		}
	}

	return refinedDisp;
}

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft, int refineRadius = 1>
Multidim::Array<float, 2> refinedBarycentricSymmetricDispFeatureVol(Multidim::Array<float, 3> const& feature_vol_l,
																	Multidim::Array<float, 3> const& feature_vol_r,
																	disp_t searchRange,
																	bool preNormalize = false) {

	typedef MatchingFunctionTraits<matchFunc> mFTraits;

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 2> sigma_left = channelsSigma(feature_vol_l, mean_left);
		Multidim::Array<float, 2> sigma_right = channelsSigma(feature_vol_r, mean_right);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(zeroMean_feature_volume_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(zeroMean_feature_volume_r, sigma_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		if (preNormalize) {
			return refineBarycentricSymmetricDisp<matchFunc, refineRadius, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp, searchRange);
		}

		return refineBarycentricSymmetricDisp<matchFunc, refineRadius, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp, searchRange);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		return refineBarycentricSymmetricDisp<matchFunc, refineRadius, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp, searchRange);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> sigma_left = channelsNorm(feature_vol_l);
		Multidim::Array<float, 2> sigma_right = channelsNorm(feature_vol_r);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(feature_vol_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(feature_vol_r, sigma_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		if (preNormalize) {
			return refineBarycentricSymmetricDisp<matchFunc, refineRadius, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp, searchRange);
		}

		return refineBarycentricSymmetricDisp<matchFunc, refineRadius, dDir>(feature_vol_l, feature_vol_r, disp, searchRange);

	} else {
		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		return refineBarycentricSymmetricDisp<matchFunc, refineRadius, dDir>(feature_vol_l, feature_vol_r, disp, searchRange);
	}

	return Multidim::Array<float, 2>();
}

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedBarycentricDispFeatureVol(Multidim::Array<float, 3> const& feature_vol_l,
														   Multidim::Array<float, 3> const& feature_vol_r,
														   disp_t searchRange,
														   bool preNormalize = false) {

	typedef MatchingFunctionTraits<matchFunc> mFTraits;

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 2> sigma_left = channelsSigma(feature_vol_l, mean_left);
		Multidim::Array<float, 2> sigma_right = channelsSigma(feature_vol_r, mean_right);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(zeroMean_feature_volume_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(zeroMean_feature_volume_r, sigma_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		if (preNormalize) {
			return refineBarycentricDisp<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp);
		}

		return refineBarycentricDisp<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		return refineBarycentricDisp<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> sigma_left = channelsNorm(feature_vol_l);
		Multidim::Array<float, 2> sigma_right = channelsNorm(feature_vol_r);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(feature_vol_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(feature_vol_r, sigma_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		if (preNormalize) {
			return refineBarycentricDisp<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp);
		}

		return refineBarycentricDisp<matchFunc, dDir>(feature_vol_l, feature_vol_r, disp);

	} else {
		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		return refineBarycentricDisp<matchFunc, dDir>(feature_vol_l, feature_vol_r, disp);
	}

	return Multidim::Array<float, 2>();
}

template<matchingFunctions matchFunc,
		 Contiguity::bidimensionalContiguity contiguity = Contiguity::Queen,
		 dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> refinedBarycentric2dDispFeatureVol(Multidim::Array<float, 3> const& feature_vol_l,
															 Multidim::Array<float, 3> const& feature_vol_r,
															 searchOffset<2> const& searchWindows,
															 bool preNormalize = false) {

	typedef MatchingFunctionTraits<matchFunc> mFTraits;

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 2> sigma_left = channelsSigma(feature_vol_l, mean_left);
		Multidim::Array<float, 2> sigma_right = channelsSigma(feature_vol_r, mean_right);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(zeroMean_feature_volume_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(zeroMean_feature_volume_r, sigma_right);

		Multidim::Array<float, 4> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchWindows);

		Multidim::Array<disp_t, 3> disp = selected2dIndexToDisp(extractSelected2dIndex<mFTraits::extractionStrategy>(CV), searchWindows);

		if (preNormalize) {
			return refineBarycentric2dDisp<matchFunc, contiguity, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp, searchWindows);
		}

		return refineBarycentric2dDisp<matchFunc, contiguity, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp, searchWindows);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 4> CV = aggregateCost<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, searchWindows);

		Multidim::Array<disp_t, 3> disp = selected2dIndexToDisp(extractSelected2dIndex<mFTraits::extractionStrategy>(CV), searchWindows);

		return refineBarycentric2dDisp<matchFunc, contiguity, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp, searchWindows);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> sigma_left = channelsNorm(feature_vol_l);
		Multidim::Array<float, 2> sigma_right = channelsNorm(feature_vol_r);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(feature_vol_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(feature_vol_r, sigma_right);

		Multidim::Array<float, 4> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchWindows);

		Multidim::Array<disp_t, 3> disp = selected2dIndexToDisp(extractSelected2dIndex<mFTraits::extractionStrategy>(CV), searchWindows);

		if (preNormalize) {
			return refineBarycentric2dDisp<matchFunc, contiguity, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp, searchWindows);
		}

		return refineBarycentric2dDisp<matchFunc, contiguity, dDir>(feature_vol_l, feature_vol_r, disp, searchWindows);

	} else {
		Multidim::Array<float, 4> CV = aggregateCost<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchWindows);

		Multidim::Array<disp_t, 3> disp = selected2dIndexToDisp(extractSelected2dIndex<mFTraits::extractionStrategy>(CV), searchWindows);

		return refineBarycentric2dDisp<matchFunc, contiguity, dDir>(feature_vol_l, feature_vol_r, disp, searchWindows);
	}

	return Multidim::Array<float, 3>();
}

template<matchingFunctions matchFunc,
		 Contiguity::bidimensionalContiguity contiguity = Contiguity::Queen,
		 dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> refinedBarycentricSymmetric2dDispFeatureVol(Multidim::Array<float, 3> const& feature_vol_l,
																	  Multidim::Array<float, 3> const& feature_vol_r,
																	  searchOffset<2> const& searchWindows,
																	  bool preNormalize = false) {

	typedef MatchingFunctionTraits<matchFunc> mFTraits;

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 2> sigma_left = channelsSigma(feature_vol_l, mean_left);
		Multidim::Array<float, 2> sigma_right = channelsSigma(feature_vol_r, mean_right);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(zeroMean_feature_volume_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(zeroMean_feature_volume_r, sigma_right);

		Multidim::Array<float, 4> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchWindows);

		Multidim::Array<disp_t, 3> disp = selected2dIndexToDisp(extractSelected2dIndex<mFTraits::extractionStrategy>(CV), searchWindows);

		if (preNormalize) {
			return refineBarycentricSymmetric2dDisp<matchFunc, contiguity, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp, searchWindows);
		}

		return refineBarycentricSymmetric2dDisp<matchFunc, contiguity, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp, searchWindows);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 4> CV = aggregateCost<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, searchWindows);

		Multidim::Array<disp_t, 3> disp = selected2dIndexToDisp(extractSelected2dIndex<mFTraits::extractionStrategy>(CV), searchWindows);

		return refineBarycentricSymmetric2dDisp<matchFunc, contiguity, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp, searchWindows);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> sigma_left = channelsNorm(feature_vol_l);
		Multidim::Array<float, 2> sigma_right = channelsNorm(feature_vol_r);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(feature_vol_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(feature_vol_r, sigma_right);

		Multidim::Array<float, 4> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchWindows);

		Multidim::Array<disp_t, 3> disp = selected2dIndexToDisp(extractSelected2dIndex<mFTraits::extractionStrategy>(CV), searchWindows);

		if (preNormalize) {
			return refineBarycentricSymmetric2dDisp<matchFunc, contiguity, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp, searchWindows);
		}

		return refineBarycentricSymmetric2dDisp<matchFunc, contiguity, dDir>(feature_vol_l, feature_vol_r, disp, searchWindows);

	} else {
		Multidim::Array<float, 4> CV = aggregateCost<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchWindows);

		Multidim::Array<disp_t, 3> disp = selected2dIndexToDisp(extractSelected2dIndex<mFTraits::extractionStrategy>(CV), searchWindows);

		return refineBarycentricSymmetric2dDisp<matchFunc, contiguity, dDir>(feature_vol_l, feature_vol_r, disp, searchWindows);
	}

	return Multidim::Array<float, 3>();
}

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedCostSymmetricDispFeatureVol(Multidim::Array<float, 3> const& feature_vol_l,
															 Multidim::Array<float, 3> const& feature_vol_r,
															 disp_t searchRange,
															 bool preNormalize = false) {

	typedef MatchingFunctionTraits<matchFunc> mFTraits;

	if (MatchingFunctionTraits<matchFunc>::ZeroMean and MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 2> sigma_left = channelsSigma(feature_vol_l, mean_left);
		Multidim::Array<float, 2> sigma_right = channelsSigma(feature_vol_r, mean_right);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(zeroMean_feature_volume_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(zeroMean_feature_volume_r, sigma_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		if (preNormalize) {
			return refineCostSymmetricDisp<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp, CV);
		}

		return refineCostSymmetricDisp<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp, CV);

	} else if (MatchingFunctionTraits<matchFunc>::ZeroMean) {

		Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
		Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

		Multidim::Array<float, 3> zeroMean_feature_volume_l = zeromeanFeatureVolume(feature_vol_l, mean_left);
		Multidim::Array<float, 3> zeroMean_feature_volume_r = zeromeanFeatureVolume(feature_vol_r, mean_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		return refineCostSymmetricDisp<matchFunc, dDir>(zeroMean_feature_volume_l, zeroMean_feature_volume_r, disp, CV);

	} else if (MatchingFunctionTraits<matchFunc>::Normalized) {

		Multidim::Array<float, 2> sigma_left = channelsNorm(feature_vol_l);
		Multidim::Array<float, 2> sigma_right = channelsNorm(feature_vol_r);

		Multidim::Array<float, 3> normalized_feature_volume_l = normalizedFeatureVolume(feature_vol_l, sigma_left);
		Multidim::Array<float, 3> normalized_feature_volume_r = normalizedFeatureVolume(feature_vol_r, sigma_right);

		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		if (preNormalize) {
			return refineCostSymmetricDisp<matchFunc, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp, CV);
		}

		return refineCostSymmetricDisp<matchFunc, dDir>(feature_vol_l, feature_vol_r, disp, CV);

	} else {
		Multidim::Array<float, 3> CV = aggregateCost<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchRange);

		Multidim::Array<disp_t, 2> disp = extractSelectedIndex<mFTraits::extractionStrategy>(CV);

		return refineCostSymmetricDisp<matchFunc, dDir>(feature_vol_l, feature_vol_r, disp, CV);
	}

	return Multidim::Array<float, 2>();
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, int refineRadius = 1>
Multidim::Array<float, 2> refinedBarycentricSymmetricDisp(Multidim::Array<T_L, nImDim> const& img_l,
														  Multidim::Array<T_R, nImDim> const& img_r,
														  uint8_t h_radius,
														  uint8_t v_radius,
														  disp_t searchRange,
														  bool preNormalize = false)
{
	static_assert (refineRadius > 0, "Barycentric symmetric refinement cannot run with a refinement radius smaller than 1 !");

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(h_radius, v_radius, img_r);

	return refinedBarycentricSymmetricDispFeatureVol<matchFunc, dDir, refineRadius>(feature_vol_l, feature_vol_r, searchRange, preNormalize);
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, int refineRadius = 1>
Multidim::Array<float, 2> refinedBarycentricSymmetricDisp(Multidim::Array<T_L, nImDim> const& img_l,
														  Multidim::Array<T_R, nImDim> const& img_r,
														  UnFoldCompressor const& compressor,
														  disp_t searchRange,
														  bool preNormalize = false)
{
	static_assert (refineRadius > 0, "Barycentric symmetric refinement cannot run with a refinement radius smaller than 1 !");

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(compressor, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(compressor, img_r);

	return refinedBarycentricSymmetricDispFeatureVol<matchFunc, dDir, refineRadius>(feature_vol_l, feature_vol_r, searchRange, preNormalize);
}



template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedBarycentricDisp(Multidim::Array<T_L, nImDim> const& img_l,
												 Multidim::Array<T_R, nImDim> const& img_r,
												 uint8_t h_radius,
												 uint8_t v_radius,
												 disp_t searchRange,
												 bool preNormalize = false)
{

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(h_radius, v_radius, img_r);

	return refinedBarycentricDispFeatureVol<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchRange, preNormalize);
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedBarycentricDisp(Multidim::Array<T_L, nImDim> const& img_l,
												 Multidim::Array<T_R, nImDim> const& img_r,
												 UnFoldCompressor const& compressor,
												 disp_t searchRange,
												 bool preNormalize = false)
{

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(compressor, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(compressor, img_r);

	return refinedBarycentricDispFeatureVol<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchRange, preNormalize);
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedCostSymmetricDisp(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   uint8_t h_radius,
												   uint8_t v_radius,
												   disp_t searchRange,
												   bool preNormalize = false)
{
	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(h_radius, v_radius, img_r);

	return refinedCostSymmetricDispFeatureVol<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchRange, preNormalize);
}

template<matchingFunctions matchFunc, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedCostSymmetricDisp(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   UnFoldCompressor const& compressor,
												   disp_t searchRange,
												   bool preNormalize = false)
{
	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(compressor, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(compressor, img_r);

	return refinedCostSymmetricDispFeatureVol<matchFunc, dDir>(feature_vol_l, feature_vol_r, searchRange, preNormalize);
}

template<matchingFunctions matchFunc,
		 class T_L,
		 class T_R,
		 int nImDim = 2,
		 Contiguity::bidimensionalContiguity contiguity = Contiguity::Queen,
		 dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> refinedBarycentric2dDisp(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   uint8_t h_radius,
												   uint8_t v_radius,
												   searchOffset<2> const& searchWindows,
												   bool preNormalize = false) {

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 3>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(h_radius, v_radius, img_r);

	return refinedBarycentric2dDispFeatureVol<matchFunc, contiguity, dDir>(feature_vol_l, feature_vol_r, searchWindows, preNormalize);

}

template<matchingFunctions matchFunc,
		 class T_L,
		 class T_R,
		 int nImDim = 2,
		 Contiguity::bidimensionalContiguity contiguity = Contiguity::Queen,
		 dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> refinedBarycentric2dDisp(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   UnFoldCompressor const& compressor,
												   searchOffset<2> const& searchWindows,
												   bool preNormalize = false) {

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 3>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(compressor, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(compressor, img_r);

	return refinedBarycentric2dDispFeatureVol<matchFunc, contiguity, dDir>(feature_vol_l, feature_vol_r, searchWindows, preNormalize);

}
template<matchingFunctions matchFunc,
		 class T_L,
		 class T_R,
		 int nImDim = 2,
		 Contiguity::bidimensionalContiguity contiguity = Contiguity::Queen,
		 dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> refinedBarycentricSymmetric2dDisp(Multidim::Array<T_L, nImDim> const& img_l,
															Multidim::Array<T_R, nImDim> const& img_r,
															uint8_t h_radius,
															uint8_t v_radius,
															searchOffset<2> const& searchWindow,
															bool preNormalize = false) {

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 3>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(h_radius, v_radius, img_r);

	return refinedBarycentricSymmetric2dDispFeatureVol<matchFunc, contiguity, dDir>(feature_vol_l, feature_vol_r, searchWindow, preNormalize);

}

template<matchingFunctions matchFunc,
		 class T_L,
		 class T_R,
		 int nImDim = 2,
		 Contiguity::bidimensionalContiguity contiguity = Contiguity::Queen,
		 dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> refinedBarycentricSymmetric2dDisp(Multidim::Array<T_L, nImDim> const& img_l,
															Multidim::Array<T_R, nImDim> const& img_r,
															UnFoldCompressor const& compressor,
															searchOffset<2> const& searchWindow,
															bool preNormalize = false) {

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 3>();
		}
	}

	Multidim::Array<float, 3> feature_vol_l = unfold(compressor, img_l);
	Multidim::Array<float, 3> feature_vol_r = unfold(compressor, img_r);

	return refinedBarycentricSymmetric2dDispFeatureVol<matchFunc, contiguity, dDir>(feature_vol_l, feature_vol_r, searchWindow, preNormalize);

}

} // namespace Correlation
} // namespace StereoVision


#endif // STEREOVISION_CROSS_CORRELATIONS_H
