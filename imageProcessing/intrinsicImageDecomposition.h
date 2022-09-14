#ifndef INTRINSICIMAGEDECOMPOSITION_H
#define INTRINSICIMAGEDECOMPOSITION_H
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022  Paragon<french.paragon@gmail.com>

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

#include <optional>

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

namespace StereoVision {
namespace ImageProcessing {

template<typename T, int nDim>
struct IntrinsicImageDecomposition {
	Multidim::Array<T, nDim> reflectance;
	Multidim::Array<T, nDim> shading;
};

template<typename T, int nDim, typename ComputeType=float>
IntrinsicImageDecomposition<ComputeType, nDim> performIntrinsicImageDecomposition(Multidim::Array<T, nDim> const& original,
																				  ComputeType lambda,
																				  std::optional<int> channelDim= (nDim > 2) ? std::optional<int>(-1) : std::nullopt,
																				  int maxIterations = 100,
																				  Multidim::Array<ComputeType, nDim> initialRefl = Multidim::Array<ComputeType, nDim>())
{

	typedef Multidim::Array<T, nDim> MDArray;
	typedef Multidim::Array<ComputeType, nDim> RMDArray;
	typedef IntrinsicImageDecomposition<ComputeType, nDim> RStruct;

	int excludedDim = -1;

	if (channelDim.has_value()) {
		if (channelDim.value() >= 0) {
			excludedDim = channelDim.value();
		} else {
			excludedDim = nDim + channelDim.value();
		}
	}

	int nPixs = 1;

	for (int i = 0; i < nDim; i++) {
		if (i != excludedDim) {
			nPixs *= original.shape()[i];
		}
	}


}

} // namespace StereoVision
} //namespace ImageProcessing

#endif // INTRINSICIMAGEDECOMPOSITION_H
