#ifndef CENSUS_H
#define CENSUS_H

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

typedef uint64_t census_data_t;
template<typename T_I>
Multidim::Array<census_data_t, 2> censusTransform2D(Multidim::Array<T_I, 2> const& input,
													int8_t h_radius,
													int8_t v_radius){

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = input.shape();

	Multidim::Array<census_data_t, 2> census(shape);

	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			census_data_t d = 0;
			int b = 0;

			for (int k = -v_radius; k <= v_radius; k++) {

				for (int l = -h_radius; l <= h_radius; l++) {

					if (i+k >= 0 and i+k < shape[0] and j+l >= 0 and j+l < shape[1]) {
						census_data_t g = (input.template value<Nc>(i,j) > input.template value<Nc>(i+k,j+l)) ? 1 : 0;
						d |= g << b;
					}

					b++;
				}

			}

			census.at<Nc>(i,j) = d;

		}

	}

	return census;

}

typedef uint8_t census_cv_t;

inline census_cv_t hammingDistance(census_data_t n1, census_data_t n2) {

	census_data_t m = n1 ^ n2;

#ifdef __GNUC__
	return static_cast<census_cv_t>(__builtin_popcountl(m));
#else
	std::bitset<std::numeric_limits<uint64_t>::digits> bs( m );
	return bs.count();
#endif

}

template<typename T_L, typename T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<census_cv_t, 3> censusCostVolume(Multidim::Array<T_L, 2> const& img_l,
												 Multidim::Array<T_R, 2> const& img_r,
												 uint8_t h_radius,
												 uint8_t v_radius,
												 disp_t disp_width,
												 disp_t disp_offset = 0) {

	Multidim::Array<census_data_t, 2> census_l = censusTransform2D(img_l, h_radius, v_radius);
	Multidim::Array<census_data_t, 2> census_r = censusTransform2D(img_r, h_radius, v_radius);

	return buildCostVolume<census_data_t,
			census_data_t,
			census_cv_t,
			hammingDistance,
			dispExtractionStartegy::Cost,
			dDir,
			rmIncompleteRanges> (census_l,
								 census_r,
								 1,
								 1,
								 disp_width,
								 disp_offset);

}

} //namespace Correlation
} //namespace StereoVision

#endif // CENSUS_H
