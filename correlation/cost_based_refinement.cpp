#include "cost_based_refinement.h"

namespace StereoVision {
namespace Correlation {

Multidim::Array<float, 2> refineDispEquiangularCostInterpolation(Multidim::Array<float, 3> const& truncatedCostVolume,
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

			float alpha = std::copysign(1.f,c0-cm1);
			alpha *= std::max(std::fabs(c0-cm1), std::fabs(c1-c0));

			float delta = (c1 - cm1)/(2*alpha);

			refined.at<Nc>(i,j) = rawDisparity.value<Nc>(i,j) + delta;

		}

	}

	return refined;

}

Multidim::Array<float, 2> refineDispParabolaCostInterpolation(Multidim::Array<float, 3> const& truncatedCostVolume,
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

			float delta = (cm1 - c1)/(2*(c1 - 2*c0 + cm1));

			refined.at<Nc>(i,j) = rawDisparity.value<Nc>(i,j) + delta;

		}

	}

	return refined;

}

} //namespace Correlation
} //namespace StereoVision
