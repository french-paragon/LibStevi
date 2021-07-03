#ifndef SSD_H
#define SSD_H

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

inline float squareDiff(float p1, float p2) {
	return (p1-p2)*(p1-p2);
}

template<class T_S, class T_T, int nImDims = 2, disp_t deltaSign = 1, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> ZSSDCostVolume(Multidim::Array<T_S, nImDims> const& img_s,
										 Multidim::Array<T_T, nImDims> const& img_t,
										 Multidim::Array<float, 2> const& s_mean,
										 Multidim::Array<float, 2> const& t_mean,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	return buildZeroMeanCostVolume<T_S,
			T_T,
			squareDiff,
			dispExtractionStartegy::Cost,
			deltaSign,
			rmIncompleteRanges>(img_s,
								img_t,
								s_mean,
								t_mean,
								h_radius,
								v_radius,
								disp_width,
								disp_offset);

}


template<class T_L, class T_R, int nImDims = 2, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> ZSSDCostVolume(Multidim::Array<T_L, nImDims> const& img_l,
										 Multidim::Array<T_R, nImDims> const& img_r,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	return buildZeroMeanCostVolume<T_L,
			T_R,
			squareDiff,
			dispExtractionStartegy::Cost,
			nImDims,
			dDir,
			rmIncompleteRanges>(img_l,
								img_r,
								h_radius,
								v_radius,
								disp_width,
								disp_offset);

}

template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refinedSSDDisp(Multidim::Array<T_L, 2> const& img_l,
										 Multidim::Array<T_R, 2> const& img_r,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	condImgRef<T_L, T_R, dDir> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
				return Multidim::Array<float, 2>(0,0);
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_l);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 2> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 2> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;

	Multidim::Array<float, 3> costVolume = ZSSDCostVolume<typename condImgRef<T_L, T_R, dDir>::T_S,
				typename condImgRef<T_L, T_R, dDir>::T_T,
				2,
				deltaSign,
				rmIncompleteRanges>(s_img,
									t_img,
									s_mean,
									t_mean,
									h_radius,
									v_radius,
									disp_width,
									disp_offset);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Cost>(costVolume);

	auto d_shape = disp.shape();
	auto t_shape = t_img.shape();

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		for (int j = 0; j < d_shape[1]; j++) {

			disp_t const& d = disp.at<Nc>(i,j);

			if (i < v_radius or i + v_radius >= d_shape[0]) { // if we are too high or too low
				refinedDisp.at<Nc>(i,j) = d;
			} else if (j < h_radius+1 or j + h_radius + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (d == 0 or d+1 >= disp_width) {
				refinedDisp.at<Nc>(i,j) = d;
			} else if (!rmIncompleteRanges and (j + disp_offset + deltaSign*d < h_radius + 1 or
					   j + disp_offset + deltaSign*d + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset < h_radius + 1 or
					   j + disp_offset + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset + deltaSign*disp_width < h_radius + 1 or
						j + disp_offset + deltaSign*disp_width + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			}  else {

				int jd = j + d;

				float a1_plus = 0;
				float a1_minus = 0;

				float a2_plus = 0;
				float a2_minus = 0;

				for(int k = -h_radius; k <= h_radius; k++) {

					for (int l = -v_radius; l <= v_radius; l++) {

						float tc_m1 = t_img.template value<Nc>(i+k, jd+l-1) - t_mean.value<Nc>(i,jd-1);
						float tc_0 = t_img.template value<Nc>(i+k, jd+l) - t_mean.value<Nc>(i,jd);
						float tc_1 = t_img.template value<Nc>(i+k, jd+l+1) - t_mean.value<Nc>(i,jd+1);

						float sc = s_img.template value<Nc>(i+k, j+l) - s_mean.value<Nc>(i,j);

						float b1_plus = tc_1 - tc_0;
						float b1_minus = tc_0 - tc_m1;

						float b2_plus = tc_0 - sc;
						float b2_minus = tc_m1 - sc;

						a1_plus += b1_plus*b1_plus;
						a1_minus += b1_minus*b1_minus;

						a2_plus += b1_plus*b2_plus;
						a2_minus += b1_minus*b2_minus;
					}
				}

				float a3_plus = costVolume.value<Nc>(i,j,d);
				float a3_minus = costVolume.value<Nc>(i,j,d-1);

				float DeltaD_plus = -a2_plus/a1_plus;
				float DeltaD_minus = -a2_minus/a1_minus;

				float cost = a3_plus;

				float DeltaD = 0;

				if (DeltaD_plus > 0 and DeltaD_plus < 1) {

					float interpSSD = a1_plus*DeltaD_plus*DeltaD_plus + 2*a2_plus*DeltaD_plus + a3_plus;

					if (interpSSD < cost) {
						cost = interpSSD;
						DeltaD = DeltaD_plus;
					}

				}

				if (DeltaD_minus > 0 and DeltaD_minus < 1) {

					float interpSSD = a1_minus*DeltaD_minus*DeltaD_minus + 2*a2_minus*DeltaD_minus + a3_minus;

					if (interpSSD < cost) {
						cost = interpSSD;
						DeltaD = DeltaD_minus - 1;
					}

				}

				refinedDisp.at<Nc>(i,j) = d + DeltaD;

			}

		}

	}

	return refinedDisp;
}

template<dispDirection dDir = dispDirection::RightToLeft>
inline Multidim::Array<float, 3> zssdFeatureVolume2CostVolume(Multidim::Array<float, 3> const& feature_vol_l,
															  Multidim::Array<float, 3> const& feature_vol_r,
															  Multidim::Array<float, 3> const& mean_l,
															  Multidim::Array<float, 3> const& mean_r,
															  disp_t disp_width) {

	condImgRef<float, float, dDir, 3> cfvr(feature_vol_l, feature_vol_r);
	condImgRef<float, float, dDir, 2> cmr(mean_l, mean_r);

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;
	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

	auto l_shape = feature_vol_l.shape();
	auto r_shape = feature_vol_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>(0,0,0);
	}

	Multidim::Array<float, 3> const& source_feature_volume = cfvr.source();
	Multidim::Array<float, 3> const& target_feature_volume = cfvr.target();

	Multidim::Array<float, 2> const& source_mean = cmr.source();
	Multidim::Array<float, 2> const& target_mean = cmr.target();

	int h = source_feature_volume.shape()[0];
	int w = source_feature_volume.shape()[1];
	int f = source_feature_volume.shape()[2];

	Multidim::Array<float, 3> costVolume({h,w,disp_width}, {w*disp_width, 1, w});

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {

			for (int d = 0; d < disp_width; d++) {

				costVolume.at<Nc>(i,j,d) = 0.0;

				for (int c = 0; c < f; c++) {
					float s = source_feature_volume.value<Nc>(i,j,c) - source_mean.value<Nc>(i,j);
					float t = target_feature_volume.valueOrAlt({i,j+deltaSign*d,c}, 0) - target_mean.value<Nc>(i,j);
					costVolume.at<Nc>(i,j,d) += (s - t)*(s - t);
				}
			}
		}
	}

	return costVolume;

}

template<dispDirection dDir = dispDirection::RightToLeft>
inline Multidim::Array<float, 3> zssdFeatureVolume2CostVolume(Multidim::Array<float, 3> const& feature_vol_l,
															  Multidim::Array<float, 3> const& feature_vol_r,
															  disp_t disp_width) {

	Multidim::Array<float, 2> mean_left = channelsMean(feature_vol_l);
	Multidim::Array<float, 2> mean_right = channelsMean(feature_vol_r);

	return zssdFeatureVolume2CostVolume<dDir>(feature_vol_l, feature_vol_r, mean_left, mean_right, disp_width);

}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> zssdUnfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
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

	return zssdFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, disp_width);
}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> zssdUnfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
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

	return zssdFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, disp_width);
}

template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refinedSSDDisp(Multidim::Array<T_L, 3> const& img_l,
										 Multidim::Array<T_R, 3> const& img_r,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	condImgRef<T_L, T_R, dDir, 3> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0] or l_shape[2] != r_shape[2]) {
		return Multidim::Array<float, 2>(0,0);
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_l);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 3> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 3> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;

	Multidim::Array<float, 3> costVolume = ZSSDCostVolume<typename condImgRef<T_L, T_R, dDir>::T_S,
				typename condImgRef<T_L, T_R, dDir>::T_T,
				3,
				deltaSign,
				rmIncompleteRanges>(s_img,
									t_img,
									s_mean,
									t_mean,
									h_radius,
									v_radius,
									disp_width,
									disp_offset);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Cost>(costVolume);

	auto d_shape = disp.shape();
	auto t_shape = t_img.shape();

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		for (int j = 0; j < d_shape[1]; j++) {

			disp_t const& d = disp.at<Nc>(i,j);

			if (i < v_radius or i + v_radius >= d_shape[0]) { // if we are too high or too low
				refinedDisp.at<Nc>(i,j) = d;
			} else if (j < h_radius+1 or j + h_radius + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (d == 0 or d+1 >= disp_width) {
				refinedDisp.at<Nc>(i,j) = d;
			} else if (!rmIncompleteRanges and (j + disp_offset + deltaSign*d < h_radius + 1 or
					   j + disp_offset + deltaSign*d + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset < h_radius + 1 or
					   j + disp_offset + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset + deltaSign*disp_width < h_radius + 1 or
						j + disp_offset + deltaSign*disp_width + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			}  else {

				int jd = j + d;

				float a1_plus = 0;
				float a1_minus = 0;

				float a2_plus = 0;
				float a2_minus = 0;

				for(int k = -h_radius; k <= h_radius; k++) {

					for (int l = -v_radius; l <= v_radius; l++) {

						for (int c = 0; c < l_shape[2]; c++) {

							float tc_m1 = t_img.template value<Nc>(i+k, jd+l-1, c) - t_mean.value<Nc>(i,jd-1);
							float tc_0 = t_img.template value<Nc>(i+k, jd+l, c) - t_mean.value<Nc>(i,jd);
							float tc_1 = t_img.template value<Nc>(i+k, jd+l+1, c) - t_mean.value<Nc>(i,jd+1);

							float sc = s_img.template value<Nc>(i+k, j+l, c) - s_mean.value<Nc>(i,j);

							float b1_plus = tc_1 - tc_0;
							float b1_minus = tc_0 - tc_m1;

							float b2_plus = tc_0 - sc;
							float b2_minus = tc_m1 - sc;

							a1_plus += b1_plus*b1_plus;
							a1_minus += b1_minus*b1_minus;

							a2_plus += b1_plus*b2_plus;
							a2_minus += b1_minus*b2_minus;
						}
					}
				}

				float a3_plus = costVolume.value<Nc>(i,j,d);
				float a3_minus = costVolume.value<Nc>(i,j,d-1);

				float DeltaD_plus = -a2_plus/a1_plus;
				float DeltaD_minus = -a2_minus/a1_minus;

				float cost = a3_plus;

				float DeltaD = 0;

				if (DeltaD_plus > 0 and DeltaD_plus < 1) {

					float interpSSD = a1_plus*DeltaD_plus*DeltaD_plus + 2*a2_plus*DeltaD_plus + a3_plus;

					if (interpSSD < cost) {
						cost = interpSSD;
						DeltaD = DeltaD_plus;
					}

				}

				if (DeltaD_minus > 0 and DeltaD_minus < 1) {

					float interpSSD = a1_minus*DeltaD_minus*DeltaD_minus + 2*a2_minus*DeltaD_minus + a3_minus;

					if (interpSSD < cost) {
						cost = interpSSD;
						DeltaD = DeltaD_minus - 1;
					}

				}

				refinedDisp.at<Nc>(i,j) = d + DeltaD;

			}

		}

	}

	return refinedDisp;
}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refinedSSDCostSymmetricDisp(Multidim::Array<T_L, nImDim> const& img_l,
													  Multidim::Array<T_R, nImDim> const& img_r,
													  uint8_t h_radius,
													  uint8_t v_radius,
													  disp_t disp_width,
													  disp_t disp_offset = 0) {


	condImgRef<T_L, T_R, dDir, nImDim> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>(0,0);
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>(0,0);
		}
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_l);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, nImDim> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, nImDim> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;

	Multidim::Array<float, 3> cv = buildZeroMeanCostVolume<typename condImgRef<T_L, T_R, dDir>::T_S,
			typename condImgRef<T_L, T_R, dDir>::T_T,
			squareDiff,
			dispExtractionStartegy::Cost,
			deltaSign,
			rmIncompleteRanges>(s_img,
								t_img,
								s_mean,
								t_mean,
								h_radius,
								v_radius,
								disp_width,
								disp_offset);

	Multidim::Array<disp_t, 2> raw_disp = extractSelectedIndex<dispExtractionStartegy::Cost, float>(cv);

	Multidim::Array<float, 3> tcv = truncatedCostVolume(cv, raw_disp, h_radius, v_radius, 1);

	return refineDispParabolaSymmetricCostInterpolation<typename condImgRef<T_L, T_R, dDir>::T_S,
			typename condImgRef<T_L, T_R, dDir>::T_T,
			squareDiff,
			dispExtractionStartegy::Cost,
			deltaSign,
			rmIncompleteRanges> (s_img,
								 t_img,
								 s_mean,
								 t_mean,
								 tcv,
								 raw_disp,
								 h_radius,
								 v_radius,
								 disp_offset);

}


template<dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedZSSDCostSymmetricFeatureVolumeDisp(Multidim::Array<float, 3> const& left_feature_volume,
																	Multidim::Array<float, 3> const& right_feature_volume,
																	disp_t disp_width) {

	auto l_shape = left_feature_volume.shape();
	auto r_shape = right_feature_volume.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>(0,0,0);
	}

	Multidim::Array<float, 2> mean_left = channelsMean(left_feature_volume);
	Multidim::Array<float, 2> mean_right = channelsMean(right_feature_volume);

	Multidim::Array<float, 3> CV = zssdFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, disp_width);

	Multidim::Array<disp_t, 2> raw_disp = extractSelectedIndex<dispExtractionStartegy::Score>(CV);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;


	condImgRef<float, float, dDir, 3> cfvr(left_feature_volume, right_feature_volume);
	condImgRef<float, float, dDir, 2> cmr(mean_left, mean_right);

	Multidim::Array<float, 3> const& source_feature_volume = cfvr.source();
	Multidim::Array<float, 3> const& target_feature_volume = cfvr.target();

	Multidim::Array<float, 2> const& source_mean = cmr.source();
	Multidim::Array<float, 2> const& target_mean = cmr.target();

	Multidim::Array<float, 3> tcv = truncatedCostVolume(CV, raw_disp, 0, 0, 1);

	return refineDispParabolaSymmetricCostInterpolation<float,
			float,
			squareDiff,
			dispExtractionStartegy::Cost,
			deltaSign> (source_feature_volume,
						target_feature_volume,
						source_mean,
						target_mean,
						tcv,
						raw_disp,
						0,
						0);
}


template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedZSSDCostSymmetricUnfoldDisp(Multidim::Array<T_L, nImDim> const& img_l,
															 Multidim::Array<T_R, nImDim> const& img_r,
															 uint8_t h_radius,
															 uint8_t v_radius,
															 disp_t disp_width) {


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

	return refinedZSSDCostSymmetricFeatureVolumeDisp<dDir>(left_feature_volume, right_feature_volume, disp_width);

}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedZSSDCostSymmetricUnfoldDisp(Multidim::Array<T_L, nImDim> const& img_l,
															 Multidim::Array<T_R, nImDim> const& img_r,
															 UnFoldCompressor compressor,
															 disp_t disp_width) {


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

	return refinedZSSDCostSymmetricFeatureVolumeDisp<dDir>(left_feature_volume, right_feature_volume, disp_width);

}

template<dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refineFeatureVolumeZSSDDisp(Multidim::Array<float, 3> const& feature_vol_l,
													  Multidim::Array<float, 3> const& feature_vol_r,
													  Multidim::Array<float, 2> const& mean_l,
													  Multidim::Array<float, 2> const& mean_r,
													  Multidim::Array<disp_t, 2> const& selectedIndex,
													  disp_t disp_width)
{

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	condImgRef<float, float, dDir, 3> cfvr(feature_vol_l, feature_vol_r);
	condImgRef<float, float, dDir, 2> cmr(mean_l, mean_r);

	Multidim::Array<float, 3> const& source_feature_volume = cfvr.source();
	Multidim::Array<float, 3> const& target_feature_volume = cfvr.target();

	Multidim::Array<float, 2> const& source_mean = cmr.source();
	Multidim::Array<float, 2> const& target_mean = cmr.target();

	auto d_shape = selectedIndex.shape();
	auto t_shape = target_feature_volume.shape();

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t d = selectedIndex.value<Nc>(i,j);

			int jd = j + deltaSign*d;

			if (j < 1 or j + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (jd < 1 or jd + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (d == 0 or d+1 >= disp_width) {
				refinedDisp.at<Nc>(i,j) = d;
			} else {

				int jd = j + d;

				float a1_plus = 0;
				float a1_minus = 0;

				float a2_plus = 0;
				float a2_minus = 0;

				float a3_plus = 0;
				float a3_minus = 0;

				for (int c = 0; c < t_shape[2]; c++) {

					float tc_m1 = target_feature_volume.value<Nc>(i, jd-1, c) - target_mean.value<Nc>(i,jd-1);
					float tc_0 = target_feature_volume.value<Nc>(i, jd, c) - target_mean.value<Nc>(i,jd);
					float tc_1 = target_feature_volume.value<Nc>(i, jd+1, c) - target_mean.value<Nc>(i,jd+1);

					float sc = source_feature_volume.value<Nc>(i, j, c) - source_mean.value<Nc>(i,j);

					float b1_plus = tc_1 - tc_0;
					float b1_minus = tc_0 - tc_m1;

					float b2_plus = tc_0 - sc;
					float b2_minus = tc_m1 - sc;

					a1_plus += b1_plus*b1_plus;
					a1_minus += b1_minus*b1_minus;

					a2_plus += b1_plus*b2_plus;
					a2_minus += b1_minus*b2_minus;

					a3_plus += b2_plus*b2_plus;
					a3_minus += b2_minus*b2_minus;
				}

				float DeltaD_plus = -a2_plus/a1_plus;
				float DeltaD_minus = -a2_minus/a1_minus;

				float cost = a3_plus;

				float DeltaD = 0;

				if (DeltaD_plus > 0 and DeltaD_plus < 1) {

					float interpSSD = a1_plus*DeltaD_plus*DeltaD_plus + 2*a2_plus*DeltaD_plus + a3_plus;

					if (interpSSD < cost) {
						cost = interpSSD;
						DeltaD = DeltaD_plus;
					}

				}

				if (DeltaD_minus > 0 and DeltaD_minus < 1) {

					float interpSSD = a1_minus*DeltaD_minus*DeltaD_minus + 2*a2_minus*DeltaD_minus + a3_minus;

					if (interpSSD < cost) {
						cost = interpSSD;
						DeltaD = DeltaD_minus - 1;
					}

				}

				refinedDisp.at<Nc>(i,j) = d + DeltaD;

			}

		}
	}

	return refinedDisp;

}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedUnfoldZSSDDisp(Multidim::Array<T_L, nImDim> const& img_l,
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

	Multidim::Array<float, 2> mean_left = channelsMean(left_feature_volume);
	Multidim::Array<float, 2> mean_right = channelsMean(right_feature_volume);

	Multidim::Array<float, 3> CV = zssdFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, disp_width);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Score>(CV);

	return refineFeatureVolumeZSSDDisp(left_feature_volume, right_feature_volume, mean_left, mean_right, disp, disp_width);
}


template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedUnfoldZSSDDisp(Multidim::Array<T_L, nImDim> const& img_l,
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

	Multidim::Array<float, 2> mean_left = channelsMean(left_feature_volume);
	Multidim::Array<float, 2> mean_right = channelsMean(right_feature_volume);

	Multidim::Array<float, 3> CV = zssdFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, disp_width);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Score>(CV);

	return refineFeatureVolumeZSSDDisp(left_feature_volume, right_feature_volume, mean_left, mean_right, disp, disp_width);
}


} //namespace Correlation
} //namespace StereoVision

#endif // SSD_H
