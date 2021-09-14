#ifndef STEREOVISION_INTERPOLATION_DOWNSAMPLING_H
#define STEREOVISION_INTERPOLATION_DOWNSAMPLING_H

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


namespace StereoVision {
namespace Interpolation {

class DownSampleWindows {
public:

	DownSampleWindows(int size) :
		_horizontal(size),
		_vertical(size)
	{

	}

	DownSampleWindows(int h_size, int v_size) :
		_horizontal(h_size),
		_vertical(v_size)
	{

	}

	DownSampleWindows(DownSampleWindows const& other) :
		_horizontal(other._horizontal),
		_vertical(other._vertical)
	{

	}

	inline int horizontal() const {
		return _horizontal;
	}

	inline int vertical() const {
		return _vertical;
	}

private:
	int _horizontal;
	int _vertical;
};

template<class T_I>
Multidim::Array<float, 2> averagePoolingDownsample(Multidim::Array<T_I, 2> const& input,
												   DownSampleWindows const& windows) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = input.shape();

	int newHeight = (shape[0]+(windows.vertical()-1))/windows.vertical();
	int newWidth = (shape[1]+(windows.horizontal()-1))/windows.horizontal();

	int hRem = newHeight*windows.vertical() - shape[0];
	int vRem = newWidth*windows.horizontal() - shape[1];

	int initialHOffset = hRem/2;
	int initialVOffset = vRem/2;

	Multidim::Array<float, 2> output(newHeight, newWidth);

	#pragma omp parallel for
	for (int i = 0; i < output.shape()[0]; i++) {
		for (int j = 0; j < output.shape()[1]; j++) {

			float val = 0;
			float count = 0;

			for (int dv = 0; dv < windows.horizontal(); dv++) {

				int p_i = i*windows.vertical() - initialVOffset + dv;

				for (int dh = 0; dh < windows.horizontal(); dh++) {

					int p_j = j*windows.horizontal() - initialHOffset + dh;

					if (p_i >= 0 and p_i < shape[0] and p_j >= 0 and p_j < shape[1]) {
						val += input.template value<Nc>(p_i, p_j);
						count += 1;
					}
				}
			}

			val /= count;
			output.at<Nc>(i,j) = val;
		}
	}

	return output;
}

template<class T_I>
ImageArray averagePoolingDownsample(Multidim::Array<T_I, 3> const& input,
									DownSampleWindows const& windows) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = input.shape();

	int newHeight = (shape[0]+(windows.vertical()-1))/windows.vertical();
	int newWidth = (shape[1]+(windows.horizontal()-1))/windows.horizontal();

	int hRem = newHeight*windows.vertical() - shape[0];
	int vRem = newWidth*windows.horizontal() - shape[1];

	int initialHOffset = hRem/2;
	int initialVOffset = vRem/2;

	ImageArray output(newHeight, newWidth, shape[2]);

	#pragma omp parallel for
	for (int i = 0; i < output.shape()[0]; i++) {
		for (int j = 0; j < output.shape()[1]; j++) {

			for (int f = 0; f < shape[2]; f++) {

				float val = 0;
				float count = 0;

				for (int dv = 0; dv < windows.horizontal(); dv++) {

					int p_i = i*windows.vertical() - initialVOffset + dv;

					for (int dh = 0; dh < windows.horizontal(); dh++) {

						int p_j = j*windows.horizontal() - initialHOffset + dh;

						if (p_i >= 0 and p_i < shape[0] and p_j >= 0 and p_j < shape[1]) {
							val += input.template value<Nc>(p_i, p_j, f);
							count += 1;
						}

					}
				}

				val /= count;
				output.at<Nc>(i,j,f) = val;

			}

		}
	}

	return output;
}

} // namespace Interpolation
} // namespace StereoVision

#endif // DOWNSAMPLING_H
