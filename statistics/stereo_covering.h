#ifndef STEREOVISION_STATISTICS_STEREO_COVERING_H
#define STEREOVISION_STATISTICS_STEREO_COVERING_H

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

#include <MultidimArrays/MultidimArrays.h>

#include "../correlation/correlation_base.h"


namespace StereoVision {
namespace Statistics {

template<Correlation::dispDirection dDir = Correlation::dispDirection::RightToLeft, class T_L, class T_R>
Multidim::Array<float, 2> computeCovering(Multidim::Array<T_L, 2> const& disp_l,
										  Multidim::Array<T_R, 2> const& disp_r,
										  float dispScaling = 1,
										  float dispOffset = 0) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;
	constexpr int deltaSign = (dDir == Correlation::dispDirection::RightToLeft) ? 1 : -1;

	Correlation::condImgRef<T_L, T_R, dDir, 2> condRefs(disp_l, disp_r);

	auto const& matchingSource = condRefs.source();
	auto const& matchingTarget = condRefs.target();

	auto shape = matchingSource.shape();
	auto tShape = matchingTarget.shape();

	if (shape[0] != tShape[0]) {
		return Multidim::Array<float, 2>(0,0);
	}

	Multidim::Array<float, 2> mask(shape);

	#pragma omp parallel for
	for(int i = 0; i < mask.shape()[0]; i++) {
		#pragma omp simd
		for(int j = 0; j < mask.shape()[1]; j++) {
			mask.at<Nc>(i,j) = 0;
		}
	}

	#pragma omp parallel for
	for (int i = 0; i < tShape[0]; i++) { //multithreads split across the rows to aviod race condition
		for (int j = 0; j < tShape[1]; j++) {

			float sourcePix = j + deltaSign*dispScaling*(matchingTarget.template value<Nc>(i,j) - dispOffset);

			int lower = static_cast<int>(std::floor(sourcePix));
			int higher = static_cast<int>(std::ceil(sourcePix));

			float propLow = sourcePix-higher;
			float propHigh = 1 - propLow;

			if (lower >= 0 and lower < shape[1]) {
				mask.at<Nc>(i, lower) += propLow;
			}

			if (higher >= 0 and higher < shape[1]) {
				mask.at<Nc>(i, higher) += propHigh;
			}
		}
	}

	return mask;

}

template<Correlation::dispDirection dDir = Correlation::dispDirection::RightToLeft, class T_L, class T_R>
float computeCoveringProportion(Multidim::Array<T_L, 2> const& disp_l,
								Multidim::Array<T_R, 2> const& disp_r,
								float coveringThreshold = 0.5,
								float dispScaling = 1,
								float dispOffset = 0) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 2> covering = computeCovering<dDir>(disp_l, disp_r, dispScaling, dispOffset);

	if (covering.empty()) {
		return -1.;
	}

	float total = covering.flatLenght();
	float covered = 0;

	#pragma omp parallel for
	for(int i = 0; i < covering.shape()[0]; i++) {
		for(int j = 0; j < covering.shape()[1]; j++) {
			if (covering.at<Nc>(i,j) >= coveringThreshold) {
				covered += 1.;
			}
		}
	}

	return covered/total;

}

} //namespace Statistics
} //namespace StereoVision

#endif // STEREOVISION_STATISTICS_STEREO_COVERING_H
