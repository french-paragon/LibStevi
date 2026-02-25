#ifndef STEREOVISION_INTERPOLATION_H
#define STEREOVISION_INTERPOLATION_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021-2023  Paragon<french.paragon@gmail.com>

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

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <cmath>

namespace StereoVision {
namespace Interpolation {

enum BorderCondition {
    Zero,
    Constant
};

template<typename T>
T unidimensionalPyramidFunction(T const& v) {
    T out = -std::abs(v) + 1;
    if (out < 0) {
        out = 0;
    }
    return out;
}

/*!
 * \brief unidimensionalBicubicKernel represent the unidimensional version of the bi-cubic interpolation kernel with unit nodes
 * \param v the parameter of the function
 * \return the kernel value
 */
template<typename T, int ratioNumerator = 1, int ratioDenominator = 2>
T unidimensionalBicubicKernel(T const& v) {
    T a = -T(ratioNumerator)/T(ratioDenominator);
    T x = std::abs(v);
    if (x < 1) {
        return (a+2)*x*x*x - (a+3)*x*x + 1;
    } else if (x < 2) {
        return a*x*x*x - 5*a*x*x + 8*a*x - 4*a;
    }
    return 0;
}

template<typename T, int inDIM>
T pyramidFunction(std::array<T,inDIM> const& pos) {
    T out = 1;
    for (int i = 0; i < inDIM; i++) {
        out *= unidimensionalPyramidFunction(pos[i]);
    }
    return out;
}

template<typename T, int inDIM, int ratioNumerator = 1, int ratioDenominator = 2>
T bicubicKernel(std::array<T,inDIM> const& pos) {
    T out = 1;
    for (int i = 0; i < inDIM; i++) {
        out *= unidimensionalBicubicKernel<T,ratioNumerator,ratioDenominator>(pos[i]);
    }
    return out;
}

template <int inDIM, typename T, T(kernel)(std::array<T,inDIM> const&), int kernelRadius, BorderCondition bCond = Constant>
inline T interpolateValue(Multidim::Array<T, inDIM> const& input,
                          std::array<T,inDIM> const& fractionalCoord) {

    using IndexIn = typename Multidim::Array<T, inDIM>::IndexBlock;
    using ShapeIn = typename Multidim::Array<T, inDIM>::ShapeBlock;
    using CoordIn = std::array<T, inDIM>;

    ShapeIn imgShape = input.shape();

    CoordIn const& c = fractionalCoord;
    IndexIn w_min;
    IndexIn w_max;

    //round up and extend coordinates
    for (int i = 0; i < inDIM; i++) {

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
    }

    IndexIn grid_shape = w_max - w_min + 1;

    T v = 0;

    Multidim::IndexConverter<inDIM> gridIdxConverter(grid_shape);

    for (int j = 0; j < gridIdxConverter.numberOfPossibleIndices() ; j++) {

        IndexIn filterCoordinate = gridIdxConverter.getIndexFromPseudoFlatId(j);
        IndexIn imgCoordinate = w_min + filterCoordinate;

        if (bCond == Zero) {
            if (!imgCoordinate.isInLimit(imgShape)) {
                continue;
            }
        }

        CoordIn kernelCoordinate;
        for (int i = 0; i < inDIM; i++) {
            kernelCoordinate[i] = imgCoordinate[i] - c[i];
        }

        if (bCond != Zero) {
            if (!imgCoordinate.isInLimit(imgShape)) {
                imgCoordinate.clip(imgShape);
            }
        }

        T kerVal = kernel(kernelCoordinate);
        T arrayVal = input.valueUnchecked(imgCoordinate);
        v += kerVal * arrayVal;

    }

    return v;

}

template <int inDIM, int outDIM, typename T, T(kernel)(std::array<T,inDIM> const&), int kernelRadius, BorderCondition bCond = Constant>
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

    ShapeCoord s = coordinates.shape();

    ShapeOut s_o;
    for (int i = 0; i < outDIM; i++) {
        s_o[i] = s[i];
    }

    Multidim::Array<T, outDIM> out(s_o);

    ShapeIn s_i = input.shape();

    int n = out.flatLenght();

    Multidim::DimsExclusionSet<outDIM+1> dimExclSet(outDIM);
    Multidim::IndexConverter<outDIM+1> gridIdxConverter(coordinates.shape(), dimExclSet);

    #pragma omp parallel for
    for (int i = 0; i < gridIdxConverter.numberOfPossibleIndices(); i++) {

        IndexCoord b = gridIdxConverter.getIndexFromPseudoFlatId(i);
        IndexOut b_o;

        for (int d = 0; d < outDIM; d++) {
            b_o[d] = b[d];
        }

        CoordIn c;
        IndexIn w_min;
        IndexIn w_max;

        IndexIn i_o;
        i_o.fill(0);

        for (int i = 0; i < inDIM; i++) {
            b.back() = i;
            ShapeCoord& pos = b;
            c[i] = coordinates.valueUnchecked(pos);
        }

        T v = interpolateValue<2,T,kernel,kernelRadius,bCond>(input, c);

        out.atUnchecked(b_o) = v;
    }

    return out;

}

inline ImageArray interpolateImage(ImageArray const& imInput,
							ImageArray const& coordinates) {

	constexpr BorderCondition bCond = Constant;
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

			outChannel.copyData(interpolate<2, 2, float, pyramidFunction<float, 2>, 0, bCond>(inChannel, coordinates));

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

		outChannel.copyData(interpolate<2, 2, float, pyramidFunction<float, 2>, 0, bCond>(inChannel, coordinates));


		return imgOut;
	}

	return Multidim::Array<float, 3>();

}

} // namespace Interpolation
} // namespace StereoVision

#endif // STEREOVISION_INTERPOLATION_H
