#ifndef STEREOVISION_UNFOLD_H
#define STEREOVISION_UNFOLD_H

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

#include "../utils/margins.h"

#include <array>
#include <vector>
#include <map>
#include <MultidimArrays/MultidimArrays.h>

#include <Eigen/Core>

namespace StereoVision {
namespace Correlation {



class UnFoldCompressor {

public:

	struct pixelIndex {
		int verticalShift;
		int horizontalShift;
		int featureIndex;
		float weight;
	};

	template<Multidim::ArrayDataAccessConstness viewConstness>
	explicit UnFoldCompressor(Multidim::Array<int,2, viewConstness> const& mask) {
		int height = mask.shape()[0];
		int width = mask.shape()[1];

		int s = height*width;

		int v_offset = height/2;
		int h_offset = width/2;

		int minH = 0;
		int maxH = 0;
		int minW = 0;
		int maxW = 0;

		std::map<int, int> pixPerSuperpix;
		std::vector<int> feats;
		feats.reserve(s);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int feat = mask.value(i,j);

				if (feat > 0) {

					if (i - v_offset < minH) {
						minH = i - v_offset;
					}
					if (i - v_offset > maxH) {
						maxH = i - v_offset;
					}
					if (j - h_offset < minW) {
						minW = j - h_offset;
					}
					if (j - h_offset > maxW) {
						maxW = j - h_offset;
					}

					if (pixPerSuperpix.count(feat)) {
						pixPerSuperpix[feat]++;
					} else {
						pixPerSuperpix[feat] = 1;
						feats.push_back(feat);
					}
				}
			}
		}

		_height = maxH - minH + 1;
		_width = maxW - minW + 1;

		_margins = PaddingMargins(-minW,-minH,maxW,maxH);

		_nFeatures = pixPerSuperpix.size();
		std::sort(feats.begin(), feats.end()); //features are in order of number used in the mask;

		_indices.reserve(s);

		for(int f = 0; f < _nFeatures; f++) {

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					int feat = mask.value(i,j);

					if (feat == feats[f]) {

						_indices.push_back({i - v_offset,
											j - h_offset,
											f,
											static_cast<float>(1./pixPerSuperpix[feat])});
					}
				}
			}

		}
	}

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

enum UnfoldPatchOrientation{
	Rotate0,
	Rotate90,
	Rotate180,
	Rotate270
};

/*!
 * \brief rotatedOffsetsFromOrientation return rotated offsets for a given orientation assuming offset 0 is the center of the patch
 * \param vOffset the vertical offset
 * \param hOffset the horizontal offset
 * \param orientation the orientation
 * \return rotated offsets for a given orientation assuming offset 0 is the center of the patch
 */
inline std::tuple<int, int> rotatedOffsetsFromOrientation(int vOffset,
														  int hOffset,
														  UnfoldPatchOrientation orientation) {

	switch (orientation) {
	case Rotate0:
		return {vOffset, hOffset};
	case Rotate90:
		return {-hOffset, vOffset};
	case Rotate180:
		return {-vOffset, -hOffset};
	case Rotate270:
		return {hOffset, -vOffset};
	}

        return {vOffset, hOffset};
}

inline int channelFromCord(int vertical,
						   int horizontal,
						   int channel,
						   int hSize,
						   int vSize,
						   int channels,
						   UnfoldPatchOrientation orientation = Rotate0) {

	if (orientation == Rotate0) {
		return channels*hSize*vertical + channels*horizontal + channel;
	} else if (orientation == Rotate90) {
		return channels*vSize*(hSize - horizontal - 1) + channels*vertical + channel;
	} else if (orientation == Rotate180) {
		return channels*hSize*(vSize - vertical - 1) + channels*(hSize - horizontal - 1) + channel;
	} else if (orientation == Rotate270) {
		return channels*vSize*horizontal + channels*(vSize - vertical - 1) + channel;
	}

	return -1;

}

/*!
 * \brief getFeatureSlidingSubwindowIdxs return a matrix of indices allowing to identify the features corresponding to a sliding subwindows inside a feature vector obtained with a larger sliding window
 * \param h_radius_base The horizontal radius used for the unfold operator leading to the cost volume.
 * \param v_radius_base The vertical radius used for the unfold operator leading to the cost volume.
 * \param sub_h_radius The horizontal size of the inner sliding window.
 * \param sub_v_radius The vertical size of the inner sliding window.
 * \param nChannels The number of channels in the image which was used to build the feature volume.
 * \return a matrix which rows represent the features and columns the sliding subwindows.
 */
inline Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>
getUnfoldFeatureSlidingSubwindowIdxs(uint8_t h_radius_base,
									 uint8_t v_radius_base,
									 uint8_t sub_h_size,
									 uint8_t sub_v_size,
									 int nChannels) {

	int h_orig = 2*h_radius_base+1;
	int v_orig = 2*v_radius_base+1;

	long nSubFeatures = sub_h_size*sub_v_size*nChannels;
	long nCols = (h_orig-sub_h_size+1) * (v_orig-sub_v_size+1);

	if (nCols < 1) {
		return Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>(0,0);
	}

	if (nSubFeatures < 1) {
		return Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>(0,0);
	}

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> out(nSubFeatures,nCols);

	for(int i = 0; i < v_orig-sub_v_size+1; i++) {
		for (int j = 0; j < h_orig-sub_h_size+1; j++) {

			for (int k = 0; k < sub_v_size; k++) {
				for (int l = 0; l < sub_h_size; l++) {
					for (int c = 0; c < nChannels; c++) {

						int channel = channelFromCord(i+k, j+l, c, h_orig, v_orig, nChannels, Rotate0);
						int outRow = channelFromCord(k, l, c, sub_h_size, sub_v_size, nChannels, Rotate0);
						int outCol = channelFromCord(i, j, 0, h_orig-sub_h_size+1, v_orig-sub_v_size+1, 1, Rotate0);

						out(outRow,outCol) = channel;
					}
				}
			}

		}
	}

	return out;
}

template<class T_I, class T_O = float>
Multidim::Array<T_O, 3> unfold(uint8_t h_radius,
							   uint8_t v_radius,
							   Multidim::Array<T_I, 2> const& in_data,
							   PaddingMargins const& padding = PaddingMargins(),
							   UnfoldPatchOrientation orientation = Rotate0) {

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

	Multidim::Array<T_O, 3> out({outHeight, outWidth, featureSpaceSize}, {outWidth*featureSpaceSize, featureSpaceSize, 1});

	#pragma omp parallel for
	for (int i = 0; i < outHeight; i++) {
		int in_i = i - padding_top;

		for (int j = 0; j < outWidth; j++) {
			int in_j = j - padding_left;

			for (int k = 0; k < v; k++) {
				for (int l = 0; l < h; l++) {
					int c = channelFromCord(k, l, 0, h, v, 1, orientation);
					out.template at<Nc>(i,j,c) = static_cast<T_O>( in_data.valueOrAlt({in_i+k, in_j+l}, 0) );
				}
			}
		}
	}

	return out;
}
template<class T_I, class T_O = float>
Multidim::Array<T_O, 3> unfold(uint8_t h_radius,
							   uint8_t v_radius,
							   Multidim::Array<T_I, 3> const& in_data,
							   PaddingMargins const& padding = PaddingMargins(),
							   UnfoldPatchOrientation orientation = Rotate0) {

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

	Multidim::Array<T_O, 3> out(outHeight, outWidth, featureSpaceSize);

	for (int k = 0; k < v; k++) {
		for (int l = 0; l < h; l++) {
			for (int in_c = 0; in_c < f; in_c++) {

				int c = channelFromCord(k, l, in_c, h, v, f, orientation);

				#pragma omp parallel for
				for (int i = 0; i < outHeight; i++) {
					int in_i = i + k - padding_top;

					#pragma omp simd
					for (int j = 0; j < outWidth; j++) {

						int in_j = j + l - padding_left;

						out.template at<Nc>(i,j,c) = static_cast<T_O>( in_data.valueOrAlt({in_i, in_j, in_c}, 0) );
					}

				}
			}
		}
	}

	return out;
}

template<class T_I, class T_O = float>
Multidim::Array<T_O, 3> unfold(UnFoldCompressor const& compressor,
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

	Multidim::Array<T_O, 3> out(outHeight, outWidth, featureSpaceSize);

	#pragma omp parallel for
	for (int i = 0; i < outHeight; i++) {
		#pragma omp simd
		for (int j = 0; j < outWidth; j++) {
			for (int f = 0; f < featureSpaceSize; f++) {
				out.template at<Nc>(i,j,f) = 0.0;
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

				out.template at<Nc>(i,j,f) += static_cast<T_O>( ind.weight*in_data.valueOrAlt({in_i, in_j}, 0) );
			}

		}
	}

	return out;
}

template<class T_I, class T_O = float>
Multidim::Array<T_O, 3> unfold(UnFoldCompressor const& compressor,
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

	Multidim::Array<T_O, 3> out(outHeight, outWidth, featureSpaceSize);

	#pragma omp parallel for
	for (int i = 0; i < outHeight; i++) {
		#pragma omp simd
		for (int j = 0; j < outWidth; j++) {
			for (int f = 0; f < featureSpaceSize; f++) {
				out.template at<Nc>(i,j,f) = 0.0;
			}
		}
	}

	int top = compressor.margins().top();
	int left = compressor.margins().left();

	for (UnFoldCompressor::pixelIndex const& ind : compressor.indices()) {

		int k = ind.verticalShift;
		int l = ind.horizontalShift;
		int f = ind.featureIndex;

		for (int in_c = 0; in_c < in_data.shape()[2]; in_c++) {
			#pragma omp parallel for
			for (int i = 0; i < outHeight; i++) {
				int in_i = i + k + top - padding_top;

				#pragma omp simd
				for (int j = 0; j < outWidth; j++) {

					int in_j = j + l + left - padding_left;

					out.template at<Nc>(i,j,in_c*compressor.nFeatures() + f) += static_cast<T_O>( ind.weight*in_data.valueOrAlt({in_i, in_j, in_c}, 0) );

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
