#ifndef STEREOVISION_GUIDED_COST_FILTERING_H
#define STEREOVISION_GUIDED_COST_FILTERING_H
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2023  Paragon<french.paragon@gmail.com>

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

namespace StereoVision {
namespace Correlation {

template <typename ComputeT, typename GuideT, int guideDims, typename SpatialWeightFuncT, typename MatchWeightFuncT>
Multidim::Array<ComputeT, 4> computeAggregationWindows(Multidim::Array<GuideT, guideDims> const& guide,
													   uint8_t h_radius,
													   uint8_t v_radius,
													   SpatialWeightFuncT const& spatialWeightFunc,
													   MatchWeightFuncT const& matchWeightFunc) {

	static_assert (guideDims == 2 or guideDims == 3, "Can only use a guide with two image dimensions, or two image dimensions plus one channel dimension");

	int w_shape_vertical = 2*v_radius+1;
	int w_shape_horizontal = 2*h_radius+1;

	Multidim::Array<ComputeT, 4> ret(guide.shape()[0], guide.shape()[1], w_shape_vertical, w_shape_horizontal);

	int nChannels = 1;

	if (guideDims == 3) {
		nChannels = guide.shape()[2];
	}

	#pragma omp parallel for
	for (int i = 0; i < guide.shape()[0]; i++) {
		for (int j = 0; j < guide.shape()[1]; j++) {

			for (int di = -int(v_radius); di <= v_radius; di++) {

				int window_i_idx = di+v_radius;

				for (int dj = -int(h_radius); dj <= h_radius; dj++) {

					int window_j_idx = dj+h_radius;

					if (i+di < 0 or i+di >= guide.shape()[0] or j+dj < 0 or j+dj >= guide.shape()[1]) {

						ret.atUnchecked(i,j, window_i_idx, window_j_idx) = 0;

					} else {
						ComputeT wSpatial = spatialWeightFunc(di, dj);

						Multidim::Array<GuideT, 1> refVec(nChannels);
						Multidim::Array<GuideT, 1> targetVec(nChannels);

						std::array<int, guideDims> refIdx;
						refIdx[0] = i;
						refIdx[1] = j;

						std::array<int, guideDims> targetIdx;
						targetIdx[0] = i+di;
						targetIdx[1] = j+dj;

						for (int c = 0; c < nChannels; c++) {

							if (guideDims == 3) {
								refIdx[2] = c;
								targetIdx[2] = c;
							}

							refVec.atUnchecked(c) = guide.valueUnchecked(refIdx);
							targetVec.atUnchecked(c) = guide.valueUnchecked(targetIdx);
						}

						ComputeT wMatch = matchWeightFunc(refVec, targetVec);

						ret.atUnchecked(i,j, window_i_idx, window_j_idx) = wSpatial*wMatch;
					}
				}
			}

		}
	}

	return ret;

}

template<typename T_CV, typename ComputeT>
Multidim::Array<T_CV, 3> variableCostVolumeAggregation(Multidim::Array<T_CV, 3> const& costVolume,
													   Multidim::Array<ComputeT, 4> const& aggregationWindows) {

	std::array<int,3> cv_shape = costVolume.shape();

	if (aggregationWindows.shape()[0] != cv_shape[0]) {
		return Multidim::Array<T_CV, 3>();
	}

	if (aggregationWindows.shape()[1] != cv_shape[1]) {
		return Multidim::Array<T_CV, 3>();
	}

	if (aggregationWindows.shape()[2] <= 0) {
		return Multidim::Array<T_CV, 3>();
	}

	if (aggregationWindows.shape()[3] <= 0) {
		return Multidim::Array<T_CV, 3>();
	}

	if (aggregationWindows.shape()[2]%2 != 1) {
		return Multidim::Array<T_CV, 3>();
	}

	if (aggregationWindows.shape()[3]%2 != 1) {
		return Multidim::Array<T_CV, 3>();
	}


	int v_radius = aggregationWindows.shape()[2]/2;
	int h_radius = aggregationWindows.shape()[3]/2;

	Multidim::Array<T_CV, 3> ret(cv_shape);

	#pragma omp parallel for
	for (int i = 0; i < cv_shape[0]; i++) {

		int min_di = std::max(-v_radius, -i);
		int max_di = std::min(v_radius, cv_shape[0]-1-i);

		for (int j = 0; j < cv_shape[1]; j++) {

			int min_dj = std::max(-h_radius, -j);
			int max_dj = std::min(h_radius, cv_shape[1]-1-j);

			for (int c = 0; c < cv_shape[2]; c++) {

				ComputeT acc = 0;
				ComputeT weight = 0;

				for (int di = min_di; di <= max_di; di++) {

					int window_i_idx = di+v_radius;

					for (int dj = min_dj; dj <= max_dj; dj++) {

						int window_j_idx = dj+h_radius;

						ComputeT w = aggregationWindows.valueUnchecked(i,j, window_i_idx, window_j_idx);

						acc += w*costVolume.valueUnchecked(i+di,j+dj,c);
						weight += w;

					}
				}

				ret.atUnchecked(i,j,c) = static_cast<T_CV>(acc/weight);
			}

		}
	}

	return ret;

}

} // namespace Correlation
} // namespace StereoVision

#endif // STEREOVISION_GUIDED_COST_FILTERING_H
