#ifndef COLORCONVERSIONS_H
#define COLORCONVERSIONS_H

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

#include <cmath>
#include <vector>

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include "../utils/colors.h"
#include "../utils/types_manipulations.h"

namespace StereoVision {
namespace ImageProcessing {

template<typename T, typename O = T>
Multidim::Array<O, 3> normalizedIntensityRGBImage(Multidim::Array<T, 3> const& rgbImg, O scale = 1) {

	using ctypeI = TypesManipulations::accumulation_extended_t<T>;
	using ctypeO = TypesManipulations::accumulation_extended_t<O>;
	constexpr T whiteTin = (std::is_integral_v<T>) ? std::numeric_limits<T>::max() : 1.0;
	constexpr T whiteTout = (std::is_integral_v<T>) ? std::numeric_limits<T>::max() : 1.0;

	if (rgbImg.shape()[2] != 3 and rgbImg.shape()[2] != 4) {
		return Multidim::Array<O, 3>();
	}

	Multidim::Array<O, 3> normalizedImg(rgbImg.shape(), rgbImg.strides());

	Multidim::DimsExclusionSet<3> exlusionSet(2);
	Multidim::IndexConverter<3> idxConverter(rgbImg.shape(), exlusionSet);

	#pragma omp parallel for
	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		idx[2] = 0;
		T red = rgbImg.valueUnchecked(idx);
		idx[2] = 1;
		T green = rgbImg.valueUnchecked(idx);
		idx[2] = 2;
		T blue = rgbImg.valueUnchecked(idx);

		ctypeI It = ((red + green + blue)/3);

		ctypeO nR = (whiteTout*red)/It;
		ctypeO nG = (whiteTout*green)/It;
		ctypeO nB = (whiteTout*blue)/It;

		idx[2] = 0;
		normalizedImg.atUnchecked(idx) = scale*nR;
		idx[2] = 1;
		normalizedImg.atUnchecked(idx) = scale*nG;
		idx[2] = 2;
		normalizedImg.atUnchecked(idx) = scale*nB;

		if (rgbImg.shape()[2] == 4) {
			idx[2] = 3;
			ctypeO alpha = whiteTout*rgbImg.valueUnchecked(idx)/whiteTin;
			normalizedImg.atUnchecked(idx) = alpha;
		}
	}

	return normalizedImg;

}

template<typename T, typename O = float>
Multidim::Array<O, 3> rgb2hsi(Multidim::Array<T, 3> const& rgbImg) {

	using ctype = TypesManipulations::accumulation_extended_t<T>;
	constexpr T whiteTin = (std::is_integral_v<T>) ? std::numeric_limits<T>::max() : 1.0;

	static_assert (std::is_floating_point_v<O>, "needs floating point output when computing hsi img, integral type not implemented yet");

	if (rgbImg.shape()[2] != 3 and rgbImg.shape()[2] != 4) {
		return Multidim::Array<O, 3>();
	}

	Multidim::Array<O, 3> hsiImg(rgbImg.shape(), rgbImg.strides());

	Multidim::DimsExclusionSet<3> exlusionSet(2);
	Multidim::IndexConverter<3> idxConverter(rgbImg.shape(), exlusionSet);

	#pragma omp parallel for
	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		idx[2] = 0;
		T red = rgbImg.valueUnchecked(idx);
		idx[2] = 1;
		T green = rgbImg.valueUnchecked(idx);
		idx[2] = 2;
		T blue = rgbImg.valueUnchecked(idx);

		ctype It = (red + green + blue)/3;

		ctype min = std::min({red, green, blue});
		O S = (It == 0) ? 0. : 1.0 - O(min)/It;

		O I = O(It)/whiteTin;
		O H;

		float R = red;
		float G = green;
		float B = blue;
		double proj = (R - 0.5*G - 0.5*B)/std::sqrt(R*R + G*G + B*B  - R*G - R*B - G*B);
		H = std::acos(proj);

		if (blue > green) {
			H = 2*M_PI - H;
		}

		H = H / M_PI * 180;

		idx[2] = 0;
		hsiImg.atUnchecked(idx) = H;
		idx[2] = 1;
		hsiImg.atUnchecked(idx) = S;
		idx[2] = 2;
		hsiImg.atUnchecked(idx) = I;

		if (rgbImg.shape()[2] == 4) {
			idx[2] = 3;
			hsiImg.atUnchecked(idx) = O(rgbImg.valueUnchecked(idx))/whiteTin;
		}

	}

	return hsiImg;

}


template<typename T>
Multidim::Array<T, 3> bgr2rgb(Multidim::Array<T, 3> const& bgrImg) {


	Multidim::Array<T, 3> rgbImg(bgrImg.shape());

	for (int i = 0; i < bgrImg.shape()[0]; i++) {
		for (int j = 0; j < bgrImg.shape()[1]; j++) {

			rgbImg.atUnchecked(i,j,0) = bgrImg.valueUnchecked(i,j,2);
			rgbImg.atUnchecked(i,j,1) = bgrImg.valueUnchecked(i,j,1);
			rgbImg.atUnchecked(i,j,2) = bgrImg.valueUnchecked(i,j,0);

		}
	}

	return rgbImg;
}

template<typename T, typename O = T>
Multidim::Array<O, 3> yuv2rgb(Multidim::Array<T, 3> const& yuvImg) {

	using ctype = std::conditional_t<std::is_integral_v<T>, long, float>;
	constexpr ctype whiteTin = (std::is_integral_v<T>) ? std::numeric_limits<T>::max() : 1000.0;
	constexpr ctype blackTin = (std::is_integral_v<T>) ? 0 : -1000.0;

	auto shp = yuvImg.shape();

	if (shp[2] != 3) {
		return Multidim::Array<O, 3>();
	}

	if (shp[1] < 2) {
		return Multidim::Array<O, 3>();
	}

	Multidim::Array<O, 3> rgbImg(shp);

	for (int i = 0; i < shp[0]; i++) {

		for (int j = 0; j < shp[1]; j++) {

			ctype Y = yuvImg.valueUnchecked(i,j,0);
			ctype U = yuvImg.valueUnchecked(i,j,1);
			ctype V = yuvImg.valueUnchecked(i,j,2);

			ctype rTmp;
			ctype gTmp;
			ctype bTmp;

			if (std::is_integral_v<T>) {
				rTmp = Y + ((351*(V-128))>>8);
				gTmp = Y - ((179*(V-128) + 86*(U-128))>>8);
				bTmp = Y + ((443*(U-128))>>8);
			} else {
				rTmp = Y + (1.370705 * (V-128.));
				gTmp = Y - (0.698001 * (V-128.)) - (0.337633 * (U-128.));
				bTmp = Y + (1.732446 * (U-128.));
			}

			rgbImg.atUnchecked(i,j,0) = std::min(whiteTin, std::max(blackTin, rTmp));
			rgbImg.atUnchecked(i,j,1) = std::min(whiteTin, std::max(blackTin, gTmp));
			rgbImg.atUnchecked(i,j,2) = std::min(whiteTin, std::max(blackTin, bTmp));

		}
	}

	return rgbImg;

}

template<typename T, typename O = T>
Multidim::Array<O, 3> yuyv2rgb(Multidim::Array<T, 3> const& yuyvImg) {

	using ctype = std::conditional_t<std::is_integral_v<T>, long, float>;
	constexpr ctype whiteTin = (std::is_integral_v<T>) ? std::numeric_limits<T>::max() : 1000.0;
	constexpr ctype blackTin = (std::is_integral_v<T>) ? 0 : -1000.0;

	auto shp = yuyvImg.shape();

	if (shp[2] != 2) {
		return Multidim::Array<O, 3>();
	}

	if (shp[1] < 2) {
		return Multidim::Array<O, 3>();
	}

	shp[2] = 3;

	Multidim::Array<O, 3> rgbImg(shp);


	for (int i = 0; i < shp[0]; i++) {

		ctype U = yuyvImg.valueUnchecked(i,0,1);
		ctype V = yuyvImg.valueUnchecked(i,1,1);

		for (int j = 0; j < shp[1]; j++) {

			ctype Y = yuyvImg.valueUnchecked(i,j,0);

			if (j%2 == 0) {
				U = yuyvImg.valueUnchecked(i,j,1);
			} else {
				V = yuyvImg.valueUnchecked(i,j,1);
			}

			ctype rTmp;
			ctype gTmp;
			ctype bTmp;

			if (std::is_integral_v<T>) {
				rTmp = Y + ((351*(V-128))>>8);
				gTmp = Y - ((179*(V-128) + 86*(U-128))>>8);
				bTmp = Y + ((443*(U-128))>>8);
			} else {
				rTmp = Y + (1.370705 * (V-128.));
				gTmp = Y - (0.698001 * (V-128.)) - (0.337633 * (U-128.));
				bTmp = Y + (1.732446 * (U-128.));
			}

			rgbImg.atUnchecked(i,j,0) = std::min(whiteTin, std::max(blackTin, rTmp));
			rgbImg.atUnchecked(i,j,1) = std::min(whiteTin, std::max(blackTin, gTmp));
			rgbImg.atUnchecked(i,j,2) = std::min(whiteTin, std::max(blackTin, bTmp));

		}
	}

	return rgbImg;

}

template<typename T, typename O = T>
Multidim::Array<O, 3> yvyu2rgb(Multidim::Array<T, 3> const& yvyuImg) {

	using ctype = std::conditional_t<std::is_integral_v<T>, long, float>;
	constexpr ctype whiteTin = (std::is_integral_v<T>) ? std::numeric_limits<T>::max() : 1000.0;
	constexpr ctype blackTin = (std::is_integral_v<T>) ? 0 : -1000.0;

	auto shp = yvyuImg.shape();

	if (shp[2] != 2) {
		return Multidim::Array<O, 3>();
	}

	if (shp[1] < 2) {
		return Multidim::Array<O, 3>();
	}

	shp[2] = 3;

	Multidim::Array<O, 3> rgbImg(shp);


	for (int i = 0; i < shp[0]; i++) {

		ctype V = yvyuImg.valueUnchecked(i,0,1);
		ctype U = yvyuImg.valueUnchecked(i,1,1);

		for (int j = 0; j < shp[1]; j++) {

			ctype Y = yvyuImg.valueUnchecked(i,j,0);

			if (j%2 == 0) {
				V = yvyuImg.valueUnchecked(i,j,1);
			} else {
				U = yvyuImg.valueUnchecked(i,j,1);
			}

			ctype rTmp;
			ctype gTmp;
			ctype bTmp;

			if (std::is_integral_v<T>) {
				rTmp = Y + ((351*(V-128))>>8);
				gTmp = Y - ((179*(V-128) + 86*(U-128))>>8);
				bTmp = Y + ((443*(U-128))>>8);
			} else {
				rTmp = Y + (1.370705 * (V-128.));
				gTmp = Y - (0.698001 * (V-128.)) - (0.337633 * (U-128.));
				bTmp = Y + (1.732446 * (U-128.));
			}

			rgbImg.atUnchecked(i,j,0) = std::min(whiteTin, std::max(blackTin, rTmp));
			rgbImg.atUnchecked(i,j,1) = std::min(whiteTin, std::max(blackTin, gTmp));
			rgbImg.atUnchecked(i,j,2) = std::min(whiteTin, std::max(blackTin, bTmp));

		}
	}

	return rgbImg;

}


template<typename T, typename O = T>
Multidim::Array<O, 2> img2gray(Multidim::Array<T, 3> const& img, std::vector<float> const& weights = {0.2989, 0.5870, 0.1140}) {

	auto shp = img.shape();

	if (weights.size() != shp[2]) {
		return Multidim::Array<O, 2>();
	}


	Multidim::Array<O, 2> grayImg(shp[0], shp[1]);

	for (int i = 0; i < shp[0]; i++) {

		for (int j = 0; j < shp[1]; j++) {

			float gray = 0;

			for (int c = 0; c < shp[2]; c++) {
				gray += weights[c]*img.valueUnchecked(i,j,c);
			}

			grayImg.atUnchecked(i,j) = static_cast<O>(gray);

		}
	}

	return grayImg;
}


} // namespace StereoVision
} //namespace ImageProcessing

#endif // COLORCONVERSIONS_H
