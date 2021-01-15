#include "interpolation.h"

namespace StereoVision {
namespace Interpolation {

Multidim::Array<float, 3> interpolateImage(Multidim::Array<float, 3> const& imInput,
										   Multidim::Array<float, 3> const& coordinates) {

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

}

} // namespace Interpolation
} // namespace StereoVision
