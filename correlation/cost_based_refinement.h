#ifndef COST_BASED_REFINEMENT_H
#define COST_BASED_REFINEMENT_H

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

namespace StereoVision {
namespace Correlation {

Multidim::Array<float, 2> refineDispEquiangularCostInterpolation(Multidim::Array<float, 3> const& truncatedCostVolume,
																 Multidim::Array<disp_t, 2> const& rawDisparity);

Multidim::Array<float, 2> refineDispParabolaCostInterpolation(Multidim::Array<float, 3> const& truncatedCostVolume,
															  Multidim::Array<disp_t, 2> const& rawDisparity);

template<class T_S, class T_T, CostFunctionType costFunction, dispExtractionStartegy strat, disp_t deltaSign = 1, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refineDispParabolaSymmetricCostInterpolation(Multidim::Array<T_S, 2> const& img_s,
																	   Multidim::Array<T_T, 2> const& img_t,
																	   Multidim::Array<float, 2> const& s_mean,
																	   Multidim::Array<float, 2> const& t_mean,
																	   Multidim::Array<float, 3> const& truncatedCostVolume,
																	   Multidim::Array<disp_t, 2> const& rawDisparity,
																	   uint8_t h_radius,
																	   uint8_t v_radius,
																	   disp_t disp_offset = 0) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = rawDisparity.shape();

	Multidim::Array<float, 2> refined(shape);

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float cm1 = truncatedCostVolume.value<Nc>(i,j,0);
			float c0 = truncatedCostVolume.value<Nc>(i,j,1);
			float c1 = truncatedCostVolume.value<Nc>(i,j,2);

			float delta = (cm1 - c1)/(2*(c1 - 2*c0 + cm1));

			disp_t d = rawDisparity.value<Nc>(i,j);

			if (j + disp_offset + std::abs(deltaSign*d) + h_radius + 1 < img_t.shape()[1] and j + disp_offset + std::abs(deltaSign*d) - h_radius - 1 > 0 and
				i - v_radius > 0 and i + v_radius < img_t.shape()[0] and std::isfinite(cm1) and std::isfinite(c0) and std::isfinite(c1)) {

				float fm1 = 0;
				float f0 = 0;
				float f1 = 0;

				disp_t dir = 1;

				if (delta > 0) {
					dir = -1;
				}

				for(int k = -h_radius; k <= h_radius; k++) {

					for (int l = -v_radius; l <= v_radius; l++) {
						float source = (img_s.template value<Nc>(i+k, j+l) - s_mean.value<Nc>(i,j) + img_s.template value<Nc>(i+k, j+l+dir) - s_mean.value<Nc>(i,j+dir))/2.;

						float targetm1 = img_t.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l-1) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d-1);
						float target0 = img_t.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d);
						float target1 = img_t.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l+1) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d+1);

						fm1 += costFunction(source, targetm1);
						f0 += costFunction(source, target0);
						f1 += costFunction(source, target1);
					}
				}

				float delta2 = (fm1 - f1)/(2*(f1 - 2*f0 + fm1)) - dir*0.5;

				if (std::fabs(delta2) < 1.) {
					delta = (delta + delta2)/2;
				}
			}

			refined.at<Nc>(i,j) = d + delta;

		}

	}

	return refined;

}

template<class T_S, class T_T, CostFunctionType costFunction, dispExtractionStartegy strat, disp_t deltaSign = 1, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refineDispParabolaSymmetricCostInterpolation(Multidim::Array<T_S, 3> const& img_s,
																	   Multidim::Array<T_T, 3> const& img_t,
																	   Multidim::Array<float, 2> const& s_mean,
																	   Multidim::Array<float, 2> const& t_mean,
																	   Multidim::Array<float, 3> const& truncatedCostVolume,
																	   Multidim::Array<disp_t, 2> const& rawDisparity,
																	   uint8_t h_radius,
																	   uint8_t v_radius,
																	   disp_t disp_offset = 0) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = rawDisparity.shape();

	auto s_shape = img_s.shape();
	auto t_shape = img_t.shape();

	if (s_shape[2] != t_shape[2]) {
		return Multidim::Array<float, 2>(0,0);
	}

	Multidim::Array<float, 2> refined(shape);

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float cm1 = truncatedCostVolume.value<Nc>(i,j,0);
			float c0 = truncatedCostVolume.value<Nc>(i,j,1);
			float c1 = truncatedCostVolume.value<Nc>(i,j,2);

			float delta = (cm1 - c1)/(2*(c1 - 2*c0 + cm1));

			disp_t d = rawDisparity.value<Nc>(i,j);

			if (j + disp_offset + std::abs(deltaSign*d) + h_radius + 1 < img_t.shape()[1] and j + disp_offset + std::abs(deltaSign*d) - h_radius - 1 > 0 and
				i - v_radius > 0 and i + v_radius < img_t.shape()[0] and std::isfinite(cm1) and std::isfinite(c0) and std::isfinite(c1)) {

				float fm1 = 0;
				float f0 = 0;
				float f1 = 0;

				disp_t dir = 1;

				if (delta > 0) {
					dir = -1;
				}

				for(int k = -h_radius; k <= h_radius; k++) {

					for (int l = -v_radius; l <= v_radius; l++) {

						for (int c = 0; c < s_shape[2]; c++) {

							float source = (img_s.template value<Nc>(i+k, j+l, c) - s_mean.value<Nc>(i,j) + img_s.template value<Nc>(i+k, j+l+dir, c) - s_mean.value<Nc>(i,j+dir))/2.;

							float targetm1 = img_t.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l-1, c) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d-1);
							float target0 = img_t.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l, c) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d);
							float target1 = img_t.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l+1, c) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d+1);

							fm1 += costFunction(source, targetm1);
							f0 += costFunction(source, target0);
							f1 += costFunction(source, target1);
						}
					}
				}

				float delta2 = (fm1 - f1)/(2*(f1 - 2*f0 + fm1)) - dir*0.5;

				if (std::fabs(delta2) < 1.) {
					delta = (delta + delta2)/2;
				}
			}

			refined.at<Nc>(i,j) = d + delta;

		}

	}

	return refined;

}


} //namespace Correlation
} //namespace StereoVision

#endif // COST_BASED_REFINEMENT_H
