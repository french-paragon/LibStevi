#ifndef LIBSTEVI_MEANSHIFTCLUSTERING_H
#define LIBSTEVI_MEANSHIFTCLUSTERING_H

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

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <optional>
#include <functional>
#include <cmath>

namespace StereoVision {
namespace ImageProcessing {

template<typename ComputeType=float>
class RadiusKernel {

public:

	explicit RadiusKernel(ComputeType limit) :
		_limit(limit)
	{

	}

	ComputeType operator()(std::vector<ComputeType> const& v1,std::vector<ComputeType> const& v2) {

		ComputeType d = 0;

		for (int i = 0; i < v1.size(); i++) {
			ComputeType tmp = v1[i] - v2[i];
			d += tmp*tmp;
		}

		if (d < _limit*_limit) {
			return 1;
		}

		return 0;

	}

protected:

	ComputeType _limit;

};

template<typename T, int nDim, typename ComputeType=float>
Multidim::Array<ComputeType, nDim> meanShiftClustering(Multidim::Array<T, nDim> const& original,
													   std::function<ComputeType(std::vector<ComputeType> const& v1,std::vector<ComputeType> const& v2)> const& kernel,
													   std::optional<int> channelDim=-1,
													   std::optional<ComputeType> incrLimit = std::nullopt,
													   int maxIterations = 100) {


	typedef Multidim::Array<T, nDim> MDArray;
	typedef Multidim::Array<ComputeType, nDim> RMDArray;

	Multidim::Array<ComputeType, nDim> clustered(original.shape()); //initialize array from the original

	int excludedDim = -1;

	if (channelDim.has_value()) {
		if (channelDim.value() >= 0) {
			excludedDim = channelDim.value();
		} else {
			excludedDim = nDim + channelDim.value();
		}
	}

	ComputeType incrLim = incrLimit.value_or(0);

	int nPixs = 1;
	int nVecPixs = 1;

	for (int i = 0; i < nDim; i++) {
		if (i != excludedDim) {
			nPixs *= original.shape()[i];
		} else {
			nVecPixs = original.shape()[i];
		}
	}

	Multidim::DimsExclusionSet<nDim> excludedDimFilter;

	if (excludedDim >= 0) {
		excludedDimFilter = Multidim::DimsExclusionSet<nDim>(excludedDim);
	}

	const Multidim::IndexConverter<nDim> indexIterator(original.shape(), excludedDimFilter);

	int nIdxs = indexIterator.numberOfPossibleIndices();

	#pragma omp parallel for
	for (int i = 0; i < nIdxs; i++) {

		std::vector<ComputeType> current(nVecPixs);
		typename MDArray::IndexBlock mdarray_index = indexIterator.getIndexFromPseudoFlatId(i);

		if (excludedDim >= 0) {
			for (int i = 0; i < nVecPixs; i++) {
				auto d_mdarray_index = mdarray_index;
				d_mdarray_index[excludedDim] = i;
				current[i] = static_cast<ComputeType>(original.valueUnchecked(d_mdarray_index));
			}
		} else {
			current[0] = static_cast<ComputeType>(original.valueUnchecked(mdarray_index));
		}

		std::vector<ComputeType> mean = current;

		ComputeType weight;
		std::vector<ComputeType> next(nVecPixs);

		for (int i = 0; i < maxIterations; i++) {

			weight = 0;
			for (int i = 0; i < nVecPixs; i++) {
				next[i] = 0;
			}

			for (int j = 0; j < nIdxs; j++) {

				std::vector<ComputeType> target(nVecPixs);
				typename MDArray::IndexBlock cdarray_index = indexIterator.getIndexFromPseudoFlatId(j);

				if (excludedDim >= 0) {
					for (int i = 0; i < nVecPixs; i++) {
						auto d_mdarray_index = cdarray_index;
						d_mdarray_index[excludedDim] = i;
						target[i] = static_cast<ComputeType>(original.valueUnchecked(d_mdarray_index));
					}
				} else {
					target[0] = static_cast<ComputeType>(original.valueUnchecked(cdarray_index));
				}

				ComputeType v = kernel(mean, target);

				weight += v;
				for (int i = 0; i < nVecPixs; i++) {
					next[i] += v*target[i];
				}
			}

			ComputeType rms = 0;

			for (int i = 0; i < nVecPixs; i++) {
				next[i] /= weight;
				ComputeType tmp = mean[i] - next[i];
				rms += tmp*tmp;
				mean[i] = next[i];
			}

			if (rms < incrLim*incrLim) {
				break;
			}
		}

		if (excludedDim >= 0) {
			for (int i = 0; i < nVecPixs; i++) {
				auto d_mdarray_index = mdarray_index;
				d_mdarray_index[excludedDim] = i;
				clustered.atUnchecked(d_mdarray_index) = mean[i];
			}
		} else {
			clustered.atUnchecked(mdarray_index) = mean[0];
		}
	}

	return clustered;
}

} //namespace StereoVision
} //namespace ImageProcessing

#endif // LIBSTEVI_MEANSHIFTCLUSTERING_H
