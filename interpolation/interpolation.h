#ifndef STEREOVISION_INTERPOLATION_H
#define STEREOVISION_INTERPOLATION_H

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

#include "../stevi_global.h"

#include <cmath>

namespace StereoVision {
namespace Interpolation {

template<typename T>
T unidimensionalPyramidFunction(T const& v) {
	T out = -std::abs(v) + 1;
	if (out < 0) {
		out = 0;
	}
	return out;
}

template<typename T, int inDIM>
T pyramidFunction(std::array<T,inDIM> const& pos) {
	T out = 1;
	for (int i = 0; i < inDIM; i++) {
		out *= unidimensionalPyramidFunction(pos[i]);
	}
	return out;
}

template <int inDIM, int outDIM, typename T, T(kernel)(std::array<T,inDIM> const&), int kernelRadius>
Multidim::Array<T, outDIM> interpolate(Multidim::Array<T, inDIM> const& input,
									   Multidim::Array<T, outDIM + 1> const& coordinates) {

	using ShapeIn = typename Multidim::Array<T, inDIM>::ShapeBlock;
	using ShapeOut = typename Multidim::Array<T, outDIM>::ShapeBlock;
	using ShapeCoord = typename Multidim::Array<T, outDIM + 1>::ShapeBlock;

	using IndexIn = typename Multidim::Array<T, inDIM>::IndexBlock;
	using IndexOut = typename Multidim::Array<T, outDIM>::IndexBlock;
	using IndexCoord = typename Multidim::Array<T, outDIM + 1>::IndexBlock;

	using CoordIn = std::array<T, inDIM>;

	assert(coordinates.shape().back() == inDIM);

	IndexCoord b;
	b.fill(0);
	ShapeCoord s = coordinates.shape();

	ShapeOut s_o;
	for (int i = 0; i < outDIM; i++) {
		s_o[i] = s[i];
	}

	Multidim::Array<T, outDIM> out(s_o);

	IndexOut b_o;
	b_o.fill(0);

	ShapeIn s_i = input.shape();

	int n = out.flatLenght();

	for (int i = 0; i < n; i++) {
		CoordIn c;
		IndexIn w_min;
		IndexIn w_max;

		IndexIn i_o;
		i_o.fill(0);

		int nPts = 1;

		for (int i = 0; i < inDIM; i++) {
			b.back() = i;
			ShapeCoord& pos = b;
			c[i] = coordinates.valueUnchecked(pos);

			if (fabs(c[i] - std::round(c[i])) < 1e-3) {
				w_min[i] = static_cast<int>(std::round(c[i])) - kernelRadius;
				w_max[i] = w_min[i] + 1 + 2*kernelRadius;

			} else {
				w_min[i] = static_cast<int>(std::floor(c[i])) - kernelRadius;
				w_max[i] = static_cast<int>(std::ceil(c[i])) + kernelRadius;
			}

			if (kernelRadius == 0) {
				if (w_min[i] == w_max[i]) {
					w_max[i] += 1;
				}
			}

			nPts *= 2*(1 + kernelRadius);
		}

		IndexIn i_w_l = w_max - w_min + 1;
		IndexIn i_w_c;
		i_w_c.fill(0);

		T v = 0;

		for (int j = 0; j < nPts; j++) {

			IndexIn w_c = w_min + i_w_c;

			if (!w_c.isInLimit(s_i)) {
				i_w_c.moveToNextIndex(i_w_l);
				continue;
			}

			CoordIn s_c;
			for (int i = 0; i < inDIM; i++) {
				s_c[i] = kernelRadius + c[i] - w_c[i];
			}

			v += kernel(s_c) * input.valueUnchecked(w_c);

			i_w_c.moveToNextIndex(i_w_l);
		}

		out.atUnchecked(b_o) = v;

		b.moveToNextIndex(s);
		b_o.moveToNextIndex(s_o);
	}

	return out;

}

ImageArray interpolateImage(ImageArray const& imInput,
							ImageArray const& coordinates);

} // namespace Interpolation
} // namespace StereoVision

#endif // STEREOVISION_INTERPOLATION_H
