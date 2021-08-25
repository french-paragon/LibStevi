#ifndef STEREOVISION_COST_BASED_REFINEMENT_H
#define STEREOVISION_COST_BASED_REFINEMENT_H

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

#include <Eigen/Core>
#include <Eigen/LU>

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
inline Eigen::Vector2f refineCostPatch(float c_m1_m1, float c_m1_0, float c_m1_1,
									   float c_0_m1, float c_0_0, float c_0_1,
									   float c_1_m1, float c_1_0, float c_1_1) {

	constexpr int patchSize = 9;
	constexpr int paramSize = 6;

	typedef Eigen::Matrix<float,patchSize,paramSize> MatrixAType;
	typedef Eigen::Matrix<float,paramSize,1> VectorXType;
	typedef Eigen::Matrix<float,patchSize,1> VectorLType;

	static_assert (kernel == InterpolationKernel::Parabola or kernel == InterpolationKernel::Gaussian, "Unsupported kernel used for patch cost refinement");

	if (kernel == InterpolationKernel::Parabola) {

		VectorLType L;
		L << c_m1_m1, c_m1_0, c_m1_1, c_0_m1, c_0_0, c_0_1, c_1_m1, c_1_0, c_1_1;

		//Many of this could be precomputed -> trust the compiler for that for the moment .
		VectorLType vertDeltas;
		vertDeltas << -1, -1, -1, 0, 0, 0, 1, 1, 1;
		VectorLType horzDeltas;
		horzDeltas << -1, 0, 1, -1, 0, 1, -1, 0, 1;

		MatrixAType A;

		for (int i = 0; i < patchSize; i++) {
			A(i,0) = vertDeltas(i)*vertDeltas(i);
			A(i,1) = vertDeltas(i)*horzDeltas(i);
			A(i,2) = horzDeltas(i)*horzDeltas(i);
			A(i,3) = vertDeltas(i);
			A(i,4) = horzDeltas(i);
			A(i,5) = 1;
		}

		VectorXType fitted = ((A.transpose()*A).inverse() * A.transpose())*L;

		Eigen::Matrix2f M;
		M << 2*fitted(0), fitted(1),
				fitted(1), 2*fitted(2);

		Eigen::Vector2f v;
		v << -fitted(3), -fitted(4);

		return M.inverse()*v;

	} else if (kernel == InterpolationKernel::Gaussian) {
		return refineCostPatch<InterpolationKernel::Parabola>(std::log(c_m1_m1), std::log(c_m1_0), std::log(c_m1_1),
															  std::log(c_0_m1), std::log(c_0_0), std::log(c_0_1),
															  std::log(c_1_m1), std::log(c_1_0), std::log(c_1_1));
	}

	return Eigen::Vector2f::Zero();

}

template<InterpolationKernel kernel>
Multidim::Array<float, 2> refineDispCostInterpolation(Multidim::Array<float, 3> const& truncatedCostVolume,
													  Multidim::Array<disp_t, 2> const& rawDisparity) {
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = rawDisparity.shape();

	Multidim::Array<float, 2> refined(shape);

	auto cv_shape = truncatedCostVolume.shape();

	int cv_radius = (cv_shape[2]-1)/2;

	if (cv_radius < 1 or 2*cv_radius+1 != cv_shape[2]) {
		return Multidim::Array<float, 2>();
	}

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float cm1 = truncatedCostVolume.value<Nc>(i,j,cv_radius-1);
			float c0 = truncatedCostVolume.value<Nc>(i,j,cv_radius);
			float c1 = truncatedCostVolume.value<Nc>(i,j,cv_radius+1);

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

	auto cv_shape = truncatedCostVolume.shape();

	int cv_radius0 = (cv_shape[2]-1)/2;
	int cv_radius1 = (cv_shape[3]-1)/2;

	if (cv_radius0 < 1 or cv_radius1 < 1 or 2*cv_radius0+1 != cv_shape[2] or 2*cv_radius1+1 != cv_shape[3]) {
		return Multidim::Array<float, 3>();
	}

	bool isScoreVolume = false;

	float v0 = truncatedCostVolume.value<Nc>(cv_shape[0]/2, cv_shape[1]/2, cv_radius0, cv_radius1);
	float v1 = truncatedCostVolume.value<Nc>(cv_shape[0]/2, cv_shape[1]/2, cv_radius0+1, cv_radius1);
	float v2 = truncatedCostVolume.value<Nc>(cv_shape[0]/2, cv_shape[1]/2, cv_radius0-1, cv_radius1);
	float v3 = truncatedCostVolume.value<Nc>(cv_shape[0]/2, cv_shape[1]/2, cv_radius0, cv_radius1+1);
	float v4 = truncatedCostVolume.value<Nc>(cv_shape[0]/2, cv_shape[1]/2, cv_radius0, cv_radius1-1);
	// compare with multiple values to account for nans
	if (v0 > v1) {
		isScoreVolume = true;
	}
	if (v0 > v2) {
		isScoreVolume = true;
	}
	if (v0 > v3) {
		isScoreVolume = true;
	}
	if (v0 > v4) {
		isScoreVolume = true;
	}

	auto argminForRow = [&truncatedCostVolume, &cv_shape, isScoreVolume] (int i, int j, int row) -> disp_t {

		float cost_hat = (isScoreVolume) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
		disp_t argmin = 0;
		for (int a = 0; a < cv_shape[2]; a++) {
			if (isScoreVolume) {
				if (truncatedCostVolume.value<Nc>(i,j,a,row) >= cost_hat) {
					cost_hat = truncatedCostVolume.value<Nc>(i,j,a,row);
					argmin = a;
				}
			} else {
				if (truncatedCostVolume.value<Nc>(i,j,a,row) <= cost_hat) {
					cost_hat = truncatedCostVolume.value<Nc>(i,j,a,row);
					argmin = a;
				}
			}
		}

		return argmin;

	};

	auto argminForCol = [&truncatedCostVolume, &cv_shape, isScoreVolume] (int i, int j, int col) -> disp_t {

		float cost_hat = (isScoreVolume) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
		disp_t argmin = 0;
		for (int a = 0; a < cv_shape[3]; a++) {
			if (isScoreVolume) {
				if (truncatedCostVolume.value<Nc>(i,j,col,a) >= cost_hat) {
					cost_hat = truncatedCostVolume.value<Nc>(i,j,col,a);
					argmin = a;
				}
			} else {
				if (truncatedCostVolume.value<Nc>(i,j,col,a) <= cost_hat) {
					cost_hat = truncatedCostVolume.value<Nc>(i,j,col,a);
					argmin = a;
				}
			}
		}

		return argmin;

	};

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float delta0 = 0;
			float delta1 = 0;

			if (isotropHypothesis == IsotropyHypothesis::Isotropic) {

				float c0_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0-1,cv_radius1);
				float c0_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1);
				float c0_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0+1,cv_radius1);

				float c1_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1-1);
				float c1_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1);
				float c1_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1+1);

				delta0 = refineCostTriplet<kernel>(c0_m1, c0_0, c0_1);
				delta1 = refineCostTriplet<kernel>(c1_m1, c1_0, c1_1);

			} else {

				//get the vertical refinement line

				float c0_1_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0-1,cv_radius1);
				float c0_1_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1);
				float c0_1_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0+1,cv_radius1);

				float delta0_1 = refineCostTriplet<kernel>(c0_1_m1, c0_1_0, c0_1_1);


				disp_t argmin0_0 = argminForRow(i,j,cv_radius1-1);

				float delta0_0 = delta0_1;

				if (argmin0_0 > 0 and argmin0_0 < cv_shape[2]-1) {

					float c0_0_m1 = truncatedCostVolume.value<Nc>(i,j,argmin0_0-1,cv_radius1-1);
					float c0_0_0 = truncatedCostVolume.value<Nc>(i,j,argmin0_0,cv_radius1-1);
					float c0_0_1 = truncatedCostVolume.value<Nc>(i,j,argmin0_0+1,cv_radius1-1);

					delta0_0 = argmin0_0 - cv_radius0 + refineCostTriplet<kernel>(c0_0_m1, c0_0_0, c0_0_1);

				}


				disp_t argmin0_2 = argminForRow(i,j,cv_radius1+1);

				float delta0_2 = delta0_1;

				if (argmin0_2 > 0 and argmin0_2 < cv_shape[2]-1) {

					float c0_2_m1 = truncatedCostVolume.value<Nc>(i,j,argmin0_2-1,cv_radius1+1);
					float c0_2_0 = truncatedCostVolume.value<Nc>(i,j,argmin0_2,cv_radius1+1);
					float c0_2_1 = truncatedCostVolume.value<Nc>(i,j,argmin0_2+1,cv_radius1+1);

					delta0_2 = argmin0_2 - cv_radius0 + refineCostTriplet<kernel>(c0_2_m1, c0_2_0, c0_2_1);

				}

				// fit a line delta0 = a0*delta1 + b0
				float a0 = (delta0_2 - delta0_0)/2;
				float b0 = (delta0_0 + delta0_1 + delta0_2)/3;

				//get the horizontal refinement line

				float c1_1_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1-1);
				float c1_1_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1);
				float c1_1_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1+1);

				float delta1_1 = refineCostTriplet<kernel>(c1_1_m1, c1_1_0, c1_1_1);


				disp_t argmin1_0 = argminForCol(i,j,cv_radius0-1);

				float delta1_0 = delta1_1 ;

				if (argmin1_0 > 0 and argmin1_0 < cv_shape[3]-1) {

					float c1_0_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0-1,argmin1_0-1);
					float c1_0_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0-1,argmin1_0);
					float c1_0_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0-1,argmin1_0+1);

					delta1_0 = argmin1_0 - cv_radius1 + refineCostTriplet<kernel>(c1_0_m1, c1_0_0, c1_0_1);

				}

				disp_t argmin1_2 = argminForCol(i,j,cv_radius0+1);

				float delta1_2 = delta1_1;

				if (argmin1_2 > 0 and argmin1_2 < cv_shape[3]-1) {

					float c1_2_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0+1,argmin1_2-1);
					float c1_2_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0+1,argmin1_2);
					float c1_2_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0+1,argmin1_2+1);

					delta1_2 = argmin1_2 - cv_radius1 + refineCostTriplet<kernel>(c1_2_m1, c1_2_0, c1_2_1);

				}

				// fit a line delta1 = a1*delta0 + b1
				float a1 = (delta1_2 - delta1_0)/2;
				float b1 = (delta1_0 + delta1_1 + delta1_2)/3;


				// solve for delta0, delta1, such that both delta0 = a0*delta1 + b0 and delta1 = a1*delta0 + b1 are true
				delta0 = (a0*b1 + b0)/(1-a0*a1);
				delta1 = (a1*b0 + b1)/(1-a0*a1);

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

template<InterpolationKernel kernel>
Multidim::Array<float, 3> refineDisp2dCostPatchInterpolation(Multidim::Array<float, 4> const& truncatedCostVolume,
															 Multidim::Array<disp_t, 3> const& rawDisparity) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = rawDisparity.shape();

	Multidim::Array<float, 3> refined(shape);

	auto cv_shape = truncatedCostVolume.shape();

	int cv_radius0 = (cv_shape[2]-1)/2;
	int cv_radius1 = (cv_shape[3]-1)/2;

	if (cv_radius0 < 1 or cv_radius1 < 1 or 2*cv_radius0+1 != cv_shape[2] or 2*cv_radius1+1 != cv_shape[3]) {
		return Multidim::Array<float, 3>();
	}

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float delta0 = 0;
			float delta1 = 0;

			float cm1_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0-1,cv_radius1-1);
			float cm1_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0-1,cv_radius1);
			float cm1_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0-1,cv_radius1+1);

			float c0_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1-1);
			float c0_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1);
			float c0_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0,cv_radius1+1);

			float c1_m1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0+1,cv_radius1-1);
			float c1_0 = truncatedCostVolume.value<Nc>(i,j,cv_radius0+1,cv_radius1);
			float c1_1 = truncatedCostVolume.value<Nc>(i,j,cv_radius0+1,cv_radius1+1);

			Eigen::Vector2f deltas = refineCostPatch<kernel>(cm1_m1, cm1_0, cm1_1,
															 c0_m1, c0_0, c0_1,
															 c1_m1, c1_0, c1_1);

			delta0 = deltas(0);
			delta1 = deltas(1);

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
