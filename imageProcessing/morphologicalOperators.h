#ifndef LIBSTEVI_MORPHOLOGICALOPERATORS_H
#define LIBSTEVI_MORPHOLOGICALOPERATORS_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2023  Paragon<french.paragon@gmail.com>

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

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <vector>

#include "../utils/margins.h"

#include "../correlation/unfold.h"

namespace StereoVision {
namespace ImageProcessing {

template<class T_I, class T_O = T_I>
Multidim::Array<T_O, 2> minFeature(Multidim::Array<T_I, 3> const& featureVolume) {

	constexpr int channelDim = 2;

	Multidim::DimsExclusionSet<3> exlusionSet(channelDim);
	Multidim::IndexConverter<3> idxConverter(featureVolume.shape(), exlusionSet);

	int nFeatures = featureVolume.shape()[channelDim];

	Multidim::Array<T_O, 2> out(featureVolume.shape()[0], featureVolume.shape()[1]);

	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		T_O min = static_cast<T_O>(featureVolume.valueUnchecked(idx));

		for (int f = 0; f < nFeatures; f++) {

			idx[channelDim] = f;

			T_O v = static_cast<T_O>(featureVolume.valueUnchecked(idx));

			if (v < min) {
				min = v;
			}
		}

		out.atUnchecked(idx[0], idx[1]) = min;
	}

	return out;
}

template<class T_I, class T_O = T_I>
Multidim::Array<T_O, 2> maxFeature(Multidim::Array<T_I, 3> const& featureVolume) {

	constexpr int channelDim = 2;

	Multidim::DimsExclusionSet<3> exlusionSet(channelDim);
	Multidim::IndexConverter<3> idxConverter(featureVolume.shape(), exlusionSet);

	int nFeatures = featureVolume.shape()[channelDim];

	Multidim::Array<T_O, 2> out(featureVolume.shape()[0], featureVolume.shape()[1]);

	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		T_O max = static_cast<T_O>(featureVolume.valueUnchecked(idx));

		for (int f = 0; f < nFeatures; f++) {

			idx[channelDim] = f;

			T_O v = static_cast<T_O>(featureVolume.valueUnchecked(idx));

			if (v > max) {
				max = v;
			}
		}

		out.atUnchecked(idx[0], idx[1]) = max;
	}

	return out;
}

template<class T_I, class T_O = T_I>
Multidim::Array<T_O, 2> medianFeature(Multidim::Array<T_I, 3> const& featureVolume) {

	constexpr int channelDim = 2;

	Multidim::DimsExclusionSet<3> exlusionSet(channelDim);
	Multidim::IndexConverter<3> idxConverter(featureVolume.shape(), exlusionSet);

	int nFeatures = featureVolume.shape()[channelDim];
	int medianPos = nFeatures/2;

	Multidim::Array<T_O, 2> out(featureVolume.shape()[0], featureVolume.shape()[1]);

	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		std::vector<T_O> sequence(nFeatures);

		for (int f = 0; f < nFeatures; f++) {

			idx[channelDim] = f;

			sequence[f] = static_cast<T_O>(featureVolume.valueUnchecked(idx));
		}

		std::nth_element(sequence.begin(), sequence.begin()+medianPos, sequence.end());

		out.atUnchecked(idx[0], idx[1]) = sequence[medianPos];
	}

	return out;
}

template<class T_I, class T_O = T_I>
Multidim::Array<T_O, 2> erosion(int h_radius, int v_radius, Multidim::Array<T_I, 2> const& image, PaddingMargins const& padding = PaddingMargins()) {

	Multidim::Array<T_O, 3> featureVolume = Correlation::unfold<T_I, T_O>(h_radius, v_radius, image, padding);
	return minFeature(featureVolume);

}

template<class T_I, class T_O = T_I>
Multidim::Array<T_O, 2> dilation(int h_radius, int v_radius, Multidim::Array<T_I, 2> const& image, PaddingMargins const& padding = PaddingMargins()) {

	Multidim::Array<T_O, 3> featureVolume = Correlation::unfold<T_I, T_O>(h_radius, v_radius, image, padding);
	return maxFeature(featureVolume);

}

template<class T_I, class T_O = T_I>
Multidim::Array<T_O, 2> medianFilter(int h_radius, int v_radius, Multidim::Array<T_I, 2> const& image, PaddingMargins const& padding = PaddingMargins()) {

	Multidim::Array<T_O, 3> featureVolume = Correlation::unfold<T_I, T_O>(h_radius, v_radius, image, padding);
	return medianFeature(featureVolume);

}

}
}

#endif // LIBSTEVI_MORPHOLOGICALOPERATORS_H
