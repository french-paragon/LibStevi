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

enum class InterpolationKernel
{
	Equiangular,
	Parabola,
	Gaussian
};

enum class IsotropyHypothesis
{
	Isotropic,
	Anisotropic
};

template<InterpolationKernel kernel>
inline float refineCostTriplet(float cm1, float c0, float c1) {

	float val = 0;
	switch (kernel) {
	case InterpolationKernel::Equiangular:
	{
		float alpha = std::copysign(1.f,c0-cm1);
		alpha *= std::max(std::fabs(c0-cm1), std::fabs(c1-c0));

		val = (c1 - cm1)/(2*alpha);
	}
		break;
	case InterpolationKernel::Parabola:
	{
		val = (cm1 - c1)/(2*(c1 - 2*c0 + cm1));
	}
		break;
	case InterpolationKernel::Gaussian:
	{
		val = (std::log(cm1) - std::log(c1))/(2*(std::log(c1) - 2*std::log(c0) + std::log(cm1)));
	}
		break;
	}
	return val;

}

template<InterpolationKernel kernel>
Multidim::Array<float, 2> refineDispCostInterpolation(Multidim::Array<float, 3> const& truncatedCostVolume,
													  Multidim::Array<disp_t, 2> const& rawDisparity) {
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = rawDisparity.shape();

	Multidim::Array<float, 2> refined(shape);

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float cm1 = truncatedCostVolume.value<Nc>(i,j,0);
			float c0 = truncatedCostVolume.value<Nc>(i,j,1);
			float c1 = truncatedCostVolume.value<Nc>(i,j,2);

			float delta = refineCostTriplet<kernel>(cm1, c0, c1);

			refined.at<Nc>(i,j) = rawDisparity.value<Nc>(i,j) + delta;

		}

	}

	return refined;
}


template<InterpolationKernel kernel, IsotropyHypothesis isotropHypothesis = IsotropyHypothesis::Isotropic>
Multidim::Array<float, 3> refineDisp2dCostInterpolation(Multidim::Array<float, 4> const& truncatedCostVolume,
														Multidim::Array<disp_t, 3> const& rawDisparity) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = rawDisparity.shape();

	Multidim::Array<float, 3> refined(shape);

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float delta0 = 0;
			float delta1 = 0;

			if (isotropHypothesis == IsotropyHypothesis::Isotropic) {

				float c0_m1 = truncatedCostVolume.value<Nc>(i,j,0,1);
				float c0_0 = truncatedCostVolume.value<Nc>(i,j,1,1);
				float c0_1 = truncatedCostVolume.value<Nc>(i,j,2,1);

				float c1_m1 = truncatedCostVolume.value<Nc>(i,j,1,0);
				float c1_0 = truncatedCostVolume.value<Nc>(i,j,1,1);
				float c1_1 = truncatedCostVolume.value<Nc>(i,j,1,2);

				delta0 = refineCostTriplet<kernel>(c0_m1, c0_0, c0_1);
				delta1 = refineCostTriplet<kernel>(c1_m1, c1_0, c1_1);

			} else {

				float c0_0_m1 = truncatedCostVolume.value<Nc>(i,j,0,0);
				float c0_0_0 = truncatedCostVolume.value<Nc>(i,j,1,0);
				float c0_0_1 = truncatedCostVolume.value<Nc>(i,j,2,0);

				float c0_1_m1 = truncatedCostVolume.value<Nc>(i,j,0,1);
				float c0_1_0 = truncatedCostVolume.value<Nc>(i,j,1,1);
				float c0_1_1 = truncatedCostVolume.value<Nc>(i,j,2,1);

				float c0_2_m1 = truncatedCostVolume.value<Nc>(i,j,0,2);
				float c0_2_0 = truncatedCostVolume.value<Nc>(i,j,1,2);
				float c0_2_1 = truncatedCostVolume.value<Nc>(i,j,2,2);

				float delta0_0 = refineCostTriplet<kernel>(c0_0_m1, c0_0_0, c0_0_1);
				float delta0_1 = refineCostTriplet<kernel>(c0_1_m1, c0_1_0, c0_1_1);
				float delta0_2 = refineCostTriplet<kernel>(c0_2_m1, c0_2_0, c0_2_1);

				// fit a line delta0 = a0*delta1 + b0
				float a0 = (delta0_2 - delta0_0)/2;
				float b0 = (delta0_0 + delta0_1 + delta0_2)/3;


				float c1_0_m1 = truncatedCostVolume.value<Nc>(i,j,0,0);
				float c1_0_0 = truncatedCostVolume.value<Nc>(i,j,0,1);
				float c1_0_1 = truncatedCostVolume.value<Nc>(i,j,0,2);

				float c1_1_m1 = truncatedCostVolume.value<Nc>(i,j,1,0);
				float c1_1_0 = truncatedCostVolume.value<Nc>(i,j,1,1);
				float c1_1_1 = truncatedCostVolume.value<Nc>(i,j,1,2);

				float c1_2_m1 = truncatedCostVolume.value<Nc>(i,j,2,0);
				float c1_2_0 = truncatedCostVolume.value<Nc>(i,j,2,1);
				float c1_2_1 = truncatedCostVolume.value<Nc>(i,j,2,2);

				float delta1_0 = refineCostTriplet<kernel>(c1_0_m1, c1_0_0, c1_0_1);
				float delta1_1 = refineCostTriplet<kernel>(c1_1_m1, c1_1_0, c1_1_1);
				float delta1_2 = refineCostTriplet<kernel>(c1_2_m1, c1_2_0, c1_2_1);

				// fit a line delta1 = a1*delta0 + b1
				float a1 = (delta1_2 - delta1_0)/2;
				float b1 = (delta1_0 + delta1_1 + delta1_2)/3;


				// solve for delta0, delta1, such that both delta0 = a0*delta1 + b0 and delta1 = a1*delta0 + b1 are true
				delta0 = (a0*b1 + b0)/(1-a0*a1);
				delta1 = a1*delta0 + b1;

			}

			if (std::abs(delta0) > 1 or std::abs(delta1) > 1
					or std::isnan(delta0) or std::isnan(delta1)) {
				delta0 = 0;
				delta1 = 0;
			}

			refined.at<Nc>(i,j,0) = rawDisparity.value<Nc>(i,j,0) + delta0;
			refined.at<Nc>(i,j,1) = rawDisparity.value<Nc>(i,j,1) + delta1;

		}

	}

	return refined;
}

} //namespace Correlation
} //namespace StereoVision

#endif // COST_BASED_REFINEMENT_H
