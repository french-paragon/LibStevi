#ifndef STEREOVISION_CORRELATION_HIEARCHICAL_H
#define STEREOVISION_CORRELATION_HIEARCHICAL_H

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

#include "./cross_correlations.h"
#include "../interpolation/downsampling.h"

#include <set>
#include <iostream>

namespace StereoVision {
namespace Correlation {

struct OffsetedCostVolume {
	Multidim::Array<float,3> truncated_cost_volume;
	Multidim::Array<disp_t,2> disp_estimate;
};

template<matchingFunctions matchFunc, dispDirection dDir = dispDirection::RightToLeft>
OffsetedCostVolume computeGuidedCV(Multidim::Array<float,3> const& feature_vol_l,
								   Multidim::Array<float,3> const& feature_vol_r,
								   Multidim::Array<disp_t,2> disp_guide,
								   disp_t upscale_disp_radius)
{

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto l_shape = feature_vol_l.shape();
	auto r_shape = feature_vol_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return {Multidim::Array<float, 3>(), Multidim::Array<disp_t,2>()};
	}

	constexpr bool r2l = dDir == dispDirection::RightToLeft;
	Multidim::Array<float, 3> & source_feature_volume = const_cast<Multidim::Array<float, 3> &>((r2l) ? feature_vol_r : feature_vol_l);
	Multidim::Array<float, 3> & target_feature_volume = const_cast<Multidim::Array<float, 3> &>((r2l) ? feature_vol_l : feature_vol_r);

	int h = source_feature_volume.shape()[0];
	int w = source_feature_volume.shape()[1];
	int f = source_feature_volume.shape()[2];
	int tcv_depth = 2*upscale_disp_radius + 1;

	int h_guide = disp_guide.shape()[0];
	int w_guide = disp_guide.shape()[1];

	OffsetedCostVolume ret = {
		Multidim::Array<float, 3>({h,w,tcv_depth}, {w*tcv_depth, tcv_depth, 1}),
		Multidim::Array<disp_t,2>(h,w)
	};

	//Multidim::Array<float, 3>& costVolume = ret.truncated_cost_volume;

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {

		float v_pos = float(i*(h_guide-1))/(h-1);

		int v0 = static_cast<int>(std::floor(v_pos));
		int v1 = static_cast<int>(std::ceil(v_pos));

		if (v0 == v1) {
			v1 += 1;
		}

		if (v1 == h_guide) {
			v0 -= 1;
			v1 -= 1;
		}

		for (int j = 0; j < w; j++) {

			float h_pos = float(j*(w_guide-1))/(w-1);

			int h0 = static_cast<int>(std::floor(h_pos));
			int h1 = static_cast<int>(std::ceil(h_pos));

			if (h0 == h1) {
				h1 += 1;
			}

			if (h1 == w_guide) {
				h0 -= 1;
				h1 -= 1;
			}

			float interpolatedDisp = 0;

			//interpolate disparity
			interpolatedDisp += (v_pos - v0)*(h_pos - h0)*disp_guide.value<Nc>(v1, h1);
			interpolatedDisp += (v1 - v_pos)*(h_pos - h0)*disp_guide.value<Nc>(v0, h1);
			interpolatedDisp += (v_pos - v0)*(h1 - h_pos)*disp_guide.value<Nc>(v1, h0);
			interpolatedDisp += (v1 - v_pos)*(h1 - h_pos)*disp_guide.value<Nc>(v0, h0);

			//upscale disparity
			interpolatedDisp *= 2;

			//baseDisp
			disp_t d0 = static_cast<disp_t>(std::round(interpolatedDisp));

			Multidim::Array<float, 1> source_feature_vector(f);

			for (int c = 0; c < f; c++) {
				float s = source_feature_volume.value<Nc>(i,j,c);
				source_feature_vector.at<Nc>(c) = s;
			}

			float score = (MatchingFunctionTraits<matchFunc>::extractionStrategy == dispExtractionStartegy::Cost) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
			disp_t d_r = d0;

			for (int delta_d = -upscale_disp_radius; delta_d <= upscale_disp_radius; delta_d++) {

				Multidim::Array<float, 1> target_feature_vector(f);

				for (int c = 0; c < f; c++) {
					float t = target_feature_volume.valueOrAlt({i,j+d0+delta_d,c}, 0);
					target_feature_vector.at<Nc>(c) = t;
				}

				float cmp = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector);

				ret.truncated_cost_volume.at<Nc>(i,j,delta_d+upscale_disp_radius) = cmp;

				if (MatchingFunctionTraits<matchFunc>::extractionStrategy == dispExtractionStartegy::Cost) {
					if (cmp < score) {
						score = cmp;
						d_r = d0+delta_d;
					}
				} else {
					if (cmp > score) {
						score = cmp;
						d_r = d0+delta_d;
					}
				}

			}

			ret.disp_estimate.at<Nc>(i,j) = d_r;

			if (d_r != d0) {
				int delta = d0 - d_r;

				int startempty;
				int endempty;

				if (delta > 0) {
					for (int dd = tcv_depth-1; dd >= delta ; dd--) {
						ret.truncated_cost_volume.at<Nc>(i,j,dd) = ret.truncated_cost_volume.at<Nc>(i,j,dd-delta);
					}
					startempty = 0;
					endempty = delta;
				} else {
					for (int dd = 0; dd < tcv_depth+delta; dd++) {
						ret.truncated_cost_volume.at<Nc>(i,j,dd) = ret.truncated_cost_volume.at<Nc>(i,j,dd-delta);
					}
					startempty = tcv_depth+delta;
					endempty = tcv_depth;
				}

				for (int dd = startempty; dd < endempty; dd++) {

					Multidim::Array<float, 1> target_feature_vector(f);

					for (int c = 0; c < f; c++) {
						float t = target_feature_volume.valueOrAlt({i,j+d_r+dd-upscale_disp_radius,c}, 0);
						target_feature_vector.at<Nc>(c) = t;
					}

					ret.truncated_cost_volume.at<Nc>(i,j,dd) = MatchingFunctionTraits<matchFunc>::featureComparison(source_feature_vector, target_feature_vector);
				}
			}

		}
	}

	return ret;

}

template<matchingFunctions matchFunc, int depth, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
OffsetedCostVolume hiearchicalTruncatedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
												  Multidim::Array<T_R, nImDim> const& img_r,
												  std::array<uint8_t, depth+1> h_radiuses,
												  std::array<uint8_t, depth+1>  v_radiuses,
												  disp_t disp_width,
												  disp_t upscale_disp_radius = 2) {

	static_assert (depth > 0, "Minimum depth is 1");

	Multidim::Array<float, nImDim> downscaled_l = Interpolation::averagePoolingDownsample(img_l, Interpolation::DownSampleWindows(2));
	Multidim::Array<float, nImDim> downscaled_r = Interpolation::averagePoolingDownsample(img_r, Interpolation::DownSampleWindows(2));

	if (depth == 1) {

		Multidim::Array<float, 3> level_0_cv = unfoldBasedCostVolume<matchFunc, float, float, nImDim, dDir>
				(downscaled_l,
				 downscaled_r,
				 h_radiuses[0],
				 v_radiuses[0],
				 (disp_width + 1)/2);

		Multidim::Array<disp_t, 2> level_0_disp = selectedIndexToDisp<disp_t, dDir>(extractSelectedIndex<MatchingFunctionTraits<matchFunc>::extractionStrategy>(level_0_cv),0);

		Multidim::Array<float,3> feature_vol_l = getFeatureVolumeForMatchFunc<matchFunc>(unfold(h_radiuses[1], v_radiuses[1], img_l));
		Multidim::Array<float,3> feature_vol_r = getFeatureVolumeForMatchFunc<matchFunc>(unfold(h_radiuses[1], v_radiuses[1], img_r));

		return computeGuidedCV<matchFunc, dDir>(feature_vol_l, feature_vol_r, level_0_disp, upscale_disp_radius);

	} else {

		constexpr int nextDepth = std::max(1,depth-1);

		auto truncate_radiuses = [] (std::array<uint8_t, depth+1> const& previous) -> std::array<uint8_t, nextDepth+1> {
			std::array<uint8_t, nextDepth+1> r;
			for (int i = 0; i < nextDepth+1; i++) {
				r[i] = previous[i];
			}
			return r;
		};

		auto previous_level = hiearchicalTruncatedCostVolume<matchFunc, nextDepth, float, float, nImDim, dDir>
				(downscaled_l,
				 downscaled_r,
				 truncate_radiuses(h_radiuses),
				 truncate_radiuses(v_radiuses),
				 (disp_width + 1)/2,
				 upscale_disp_radius);

		Multidim::Array<float,3> feature_vol_l = getFeatureVolumeForMatchFunc<matchFunc>(unfold(h_radiuses.back(), v_radiuses.back(), img_l));
		Multidim::Array<float,3> feature_vol_r = getFeatureVolumeForMatchFunc<matchFunc>(unfold(h_radiuses.back(), v_radiuses.back(), img_r));

		return computeGuidedCV<matchFunc, dDir>(feature_vol_l, feature_vol_r, previous_level.disp_estimate, upscale_disp_radius);
	}

}

template<matchingFunctions matchFunc, int depth, class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
OffsetedCostVolume hiearchicalTruncatedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
												  Multidim::Array<T_R, nImDim> const& img_r,
												  uint8_t h_radius,
												  uint8_t  v_radius,
												  disp_t disp_width,
												  disp_t upscale_disp_radius = 2) {
	std::array<uint8_t, depth+1> h_radiuses;
	std::array<uint8_t, depth+1>  v_radiuses;

	for (int i = 0; i < depth+1; i++) {
		h_radiuses[i] = h_radius;
		v_radiuses[i] = v_radius;
	}

	return hiearchicalTruncatedCostVolume<matchFunc, depth, T_L, T_R, nImDim, dDir>
			(img_l, img_r, h_radiuses, v_radiuses, disp_width, upscale_disp_radius);
}

} // namespace Correlation
} // namespace StereoVision

#endif // HIEARCHICAL_H
