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

#include "interpolation.h"

namespace StereoVision {
namespace Interpolation {

Multidim::Array<float, 3> interpolateImage(Multidim::Array<float, 3> const& imInput,
										   Multidim::Array<float, 3> const& coordinates) {

	auto s = imInput.shape();

	if (s[2] == 3) {

		Multidim::Array<float, 3> imgOut(coordinates.shape()[0], coordinates.shape()[1], 3);

		for (int i = 0; i < 3; i++) {
			Multidim::Array<float, 3>* nonConst = const_cast<Multidim::Array<float, 3>*>(&imInput);

			Multidim::Array<float, 2> inChannel = nonConst->subView(Multidim::DimSlice(),
																	Multidim::DimSlice(),
																	Multidim::DimIndex(i));

			Multidim::Array<float, 2> outChannel = imgOut.subView(Multidim::DimSlice(),
																  Multidim::DimSlice(),
																  Multidim::DimIndex(i));

			outChannel.copyData(interpolate<2, 2, float, pyramidFunction<float, 2>, 0>(inChannel, coordinates));

		}

		return imgOut;

	} else if (s[2] == 1) {

		Multidim::Array<float, 3> imgOut(coordinates.shape()[0], coordinates.shape()[1], 1);

		Multidim::Array<float, 3>* nonConst = const_cast<Multidim::Array<float, 3>*>(&imInput);

		Multidim::Array<float, 2> inChannel = nonConst->subView(Multidim::DimSlice(),
																Multidim::DimSlice(),
																Multidim::DimIndex(0));

		Multidim::Array<float, 2> outChannel = imgOut.subView(Multidim::DimSlice(),
															  Multidim::DimSlice(),
															  Multidim::DimIndex(0));

		outChannel.copyData(interpolate<2, 2, float, pyramidFunction<float, 2>, 0>(inChannel, coordinates));


		return imgOut;
	}

	return Multidim::Array<float, 3>();

}

} // namespace Interpolation
} // namespace StereoVision
