#ifndef STEREOVISION_CENSUS_H
#define STEREOVISION_CENSUS_H

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
#include "./matching_costs.h"
#include "./unfold.h"

namespace StereoVision {
namespace Correlation {

template<typename T_I, Multidim::ArrayDataAccessConstness C>
Multidim::Array<census_data_t, 3> censusFeatures(Multidim::Array<T_I, 3, C> const& baseFeatures) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = baseFeatures.shape();

	if (shape[2] <= 1) {
		return Multidim::Array<census_data_t, 3>(); //impossible to compute the census when only a single feature channel is present
	}

	int size_census_features = (shape[2] - 1)/(sizeof (census_data_t) * 8) + 1;

	Multidim::Array<census_data_t, 3> census(shape[0], shape[1], size_census_features);

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			T_I ref = baseFeatures.template value<Nc>(i,j,0);

			census_data_t d = 0;
			size_t b = 0;
			int census_channel = 0;

			for (int c = 1; c < shape[2]; c++) {
				T_I val = baseFeatures.template value<Nc>(i,j,c);

				census_data_t g = (ref > val) ? 1 : 0;

				d |= g << b;
				b++;

				if (b >= sizeof (census_data_t)*8) {
					census.at<Nc>(i,j, census_channel) = d;
					census_channel++;
					d = 0;
					b = 0;
				}
			}

		}
	}

	return census;
}

template<typename T_I, int nDim>
Multidim::Array<census_data_t, 3> censusTransform2D(Multidim::Array<T_I, nDim> const& input,
													int8_t h_radius,
													int8_t v_radius,
													PaddingMargins const& padding = PaddingMargins()) {

	static_assert (nDim == 2 or nDim == 3, "Can process only 2D (grascale images) or 3D (colored images) arrays.");

	Multidim::Array<T_I, 3> basefeatures = unfold<T_I, T_I>(h_radius, v_radius, input, padding);

	Multidim::Array<census_data_t, 3> census = censusFeatures(basefeatures);

	return census;

}

} //namespace Correlation
} //namespace StereoVision

#endif // CENSUS_H
