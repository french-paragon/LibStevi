#ifndef UNFOLD_H
#define UNFOLD_H

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

#include <array>
#include <vector>
#include <MultidimArrays/MultidimArrays.h>

namespace StereoVision {
namespace Correlation {

class PaddingMargins {

public:

	PaddingMargins() :
		_left(0),
		_right(0),
		_top(0),
		_bottom(0),
		_auto(true)
	{

	}
	PaddingMargins(int padding) :
		_left(padding),
		_right(padding),
		_top(padding),
		_bottom(padding),
		_auto(false)
	{

	}
	PaddingMargins(int leftright, int topbottom) :
		_left(leftright),
		_right(leftright),
		_top(topbottom),
		_bottom(topbottom),
		_auto(false)
	{

	}
	PaddingMargins(int left, int top, int right, int bottom) :
		_left(left),
		_right(right),
		_top(top),
		_bottom(bottom),
		_auto(false)
	{

	}

	PaddingMargins(PaddingMargins const& other) :
		_left(other._left),
		_right(other._right),
		_top(other._top),
		_bottom(other._bottom),
		_auto(other._auto)
	{

	}

	PaddingMargins& operator=(PaddingMargins const& other)
	{
		_left = other._left;
		_right = other._right;
		_top = other._top;
		_bottom = other._bottom;
		_auto = other._auto;

		return *this;
	}

	inline bool isAuto() const { return _auto; }
	inline int left() const { return _left; }
	inline int right() const { return _right; }
	inline int top() const  { return _top; }
	inline int bottom() const  { return _bottom; }

protected:

	int _left;
	int _right;
	int _top;
	int _bottom;
	bool _auto;

};


class UnFoldCompressor {

public:

	struct pixelIndex {
		int verticalShift;
		int horizontalShift;
		int featureIndex;
		float weight;
	};

	explicit UnFoldCompressor(Multidim::Array<int,2> const& mask);

	inline int nFeatures() const { return _nFeatures; }
	inline int width() const { return _width; }
	inline int height() const { return _height; }
	inline PaddingMargins margins() const { return _margins; }
	inline std::vector<pixelIndex> indices() const { return _indices; }

protected:

	int _nFeatures;
	int _width;
	int _height;
	PaddingMargins _margins;
	std::vector<pixelIndex> _indices;
};

template<class T_I>
Multidim::Array<float, 3> unfold(uint8_t h_radius,
								 uint8_t v_radius,
								 Multidim::Array<T_I, 2> const& in_data,
								 PaddingMargins const& padding = PaddingMargins()) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int padding_left = (padding.isAuto()) ? h_radius : padding.left();
	int padding_right = (padding.isAuto()) ? h_radius : padding.right();
	int padding_top = (padding.isAuto()) ? v_radius : padding.top();
	int padding_bottom = (padding.isAuto()) ? v_radius : padding.bottom();

	int h = 2*h_radius+1;
	int v = 2*v_radius+1;

	int featureSpaceSize = h*v;

	int inHeight = in_data.shape()[0];
	int inWidth = in_data.shape()[1];

	int outHeight = inHeight - v + padding_top + padding_bottom + 1;
	int outWidth = inWidth - h + padding_left + padding_right + 1;

	Multidim::Array<float, 3> out({outHeight, outWidth, featureSpaceSize}, {outWidth*featureSpaceSize, featureSpaceSize, 1});

	#pragma omp parallel for
	for (int i = 0; i < outHeight; i++) {
		int in_i = i - padding_top;

		for (int j = 0; j < outWidth; j++) {
			int in_j = j - padding_left;

			for (int k = 0; k < v; k++) {
				for (int l = 0; l < h; l++) {
					int c = k*h + l;
					out.at<Nc>(i,j,c) = static_cast<float>( in_data.valueOrAlt({in_i+k, in_j+l}, 0) );
				}
			}
		}
	}

	return out;
}
template<class T_I>
Multidim::Array<float, 3> unfold(uint8_t h_radius,
								 uint8_t v_radius,
								 Multidim::Array<T_I, 3> const& in_data,
								 PaddingMargins const& padding = PaddingMargins()) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int padding_left = (padding.isAuto()) ? h_radius : padding.left();
	int padding_right = (padding.isAuto()) ? h_radius : padding.right();
	int padding_top = (padding.isAuto()) ? v_radius : padding.top();
	int padding_bottom = (padding.isAuto()) ? v_radius : padding.bottom();

	int h = 2*h_radius+1;
	int v = 2*v_radius+1;
	int f = in_data.shape()[2];

	int featureSpaceSize = h*v*f;

	int inHeight = in_data.shape()[0];
	int inWidth = in_data.shape()[1];

	int outHeight = inHeight - v + padding_top + padding_bottom + 1;
	int outWidth = inWidth - h + padding_left + padding_right + 1;

	Multidim::Array<float, 3> out(outHeight, outWidth, featureSpaceSize);

	for (int k = 0; k < v; k++) {
		for (int l = 0; l < h; l++) {
			for (int in_c = 0; in_c < f; in_c++) {

				int c = in_c*h*v + k*h + l;

				#pragma omp parallel for
				for (int i = 0; i < outHeight; i++) {
					int in_i = i + k - padding_top;

					#pragma omp simd
					for (int j = 0; j < outWidth; j++) {

						int in_j = j + l - padding_left;

						out.at<Nc>(i,j,c) = static_cast<float>( in_data.valueOrAlt({in_i, in_j, in_c}, 0) );
					}

				}
			}
		}
	}

	return out;
}

template<class T_I>
Multidim::Array<float, 3> unfold(UnFoldCompressor const& compressor,
								 Multidim::Array<T_I, 2> const& in_data,
								 PaddingMargins const& padding = PaddingMargins()) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int padding_left = (padding.isAuto()) ? compressor.margins().left() : padding.left();
	int padding_right = (padding.isAuto()) ? compressor.margins().right() : padding.right();
	int padding_top = (padding.isAuto()) ? compressor.margins().top() : padding.top();
	int padding_bottom = (padding.isAuto()) ? compressor.margins().bottom() : padding.bottom();

	int h = compressor.width();
	int v = compressor.height();

	int featureSpaceSize = compressor.nFeatures();

	int inHeight = in_data.shape()[0];
	int inWidth = in_data.shape()[1];

	int outHeight = inHeight - v + padding_top + padding_bottom + 1;
	int outWidth = inWidth - h + padding_left + padding_right + 1;

	Multidim::Array<float, 3> out(outHeight, outWidth, featureSpaceSize);

	#pragma omp parallel for
	for (int i = 0; i < outHeight; i++) {
		#pragma omp simd
		for (int j = 0; j < outWidth; j++) {
			for (int f = 0; f < featureSpaceSize; f++) {
				out.at<Nc>(i,j,f) = 0.0;
			}
		}
	}

	int top = compressor.margins().top();
	int left = compressor.margins().left();

	for (UnFoldCompressor::pixelIndex const& ind : compressor.indices()) {

		int k = ind.verticalShift;
		int l = ind.horizontalShift;
		int f = ind.featureIndex;

		#pragma omp parallel for
		for (int i = 0; i < outHeight; i++) {
			int in_i = i + k + top - padding_top;

			#pragma omp simd
			for (int j = 0; j < outWidth; j++) {

				int in_j = j + l + left - padding_left;

				out.at<Nc>(i,j,f) += ind.weight*static_cast<float>( in_data.valueOrAlt({in_i, in_j}, 0) );
			}

		}
	}

	return out;
}

template<class T_I>
Multidim::Array<float, 3> unfold(UnFoldCompressor const& compressor,
								 Multidim::Array<T_I, 3> const& in_data,
								 PaddingMargins const& padding = PaddingMargins()) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int padding_left = (padding.isAuto()) ? compressor.margins().left() : padding.left();
	int padding_right = (padding.isAuto()) ? compressor.margins().right() : padding.right();
	int padding_top = (padding.isAuto()) ? compressor.margins().top() : padding.top();
	int padding_bottom = (padding.isAuto()) ? compressor.margins().bottom() : padding.bottom();

	int h = compressor.width();
	int v = compressor.height();

	int featureSpaceSize = in_data.shape()[2]*compressor.nFeatures();

	int inHeight = in_data.shape()[0];
	int inWidth = in_data.shape()[1];

	int outHeight = inHeight - v + padding_top + padding_bottom + 1;
	int outWidth = inWidth - h + padding_left + padding_right + 1;

	Multidim::Array<float, 3> out(outHeight, outWidth, featureSpaceSize);

	#pragma omp parallel for
	for (int i = 0; i < outHeight; i++) {
		#pragma omp simd
		for (int j = 0; j < outWidth; j++) {
			for (int f = 0; f < featureSpaceSize; f++) {
				out.at<Nc>(i,j,f) = 0.0;
			}
		}
	}

	int top = compressor.margins().top();
	int left = compressor.margins().left();

	for (UnFoldCompressor::pixelIndex const& ind : compressor.indices()) {

		int k = ind.verticalShift;
		int l = ind.horizontalShift;
		int f = ind.featureIndex;
		int in_c = 0;

		for (int in_c = 0; in_c < in_data.shape()[2]; in_c++) {
			#pragma omp parallel for
			for (int i = 0; i < outHeight; i++) {
				int in_i = i + k + top - padding_top;

				#pragma omp simd
				for (int j = 0; j < outWidth; j++) {

					int in_j = j + l + left - padding_left;

					out.at<Nc>(i,j,in_c*compressor.nFeatures() + f) += ind.weight*static_cast<float>( in_data.valueOrAlt({in_i, in_j, in_c}, 0) );

				}

			}
		}
	}

	return out;
}

namespace CompressorGenerators {

Multidim::Array<int,2> GrPix17R3Filter();
Multidim::Array<int,2> GrPix17R4Filter();

} //namespace CompressorGenerators

} //namespace Correlation
} //namespace StereoVision

#endif // UNFOLD_H
