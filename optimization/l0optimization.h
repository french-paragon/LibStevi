#ifndef LIBSTEVI_L0_OPTIMIZATION_H
#define LIBSTEVI_L0_OPTIMIZATION_H

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

#include <optional>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

#include "../utils/indexers.h"

namespace StereoVision {
namespace Optimization {


template<typename T, int nDim, typename ComputeType=float>
/*!
 * \brief regionFusionL0Approximation tries to find the best approximation of an images with a penalty on the l0 norm of the gradient.
 * \param original the images on want to approximate
 * \param lambda the penatly term
 * \return The approximation
 *
 * The algorithm implemented is from "Fast and Effective L0 Gradient Minimization by Region Fusion, ICCV, 2015"
 *
 * For practicalyty, we use the linear option for the evolution of the parameter beta.
 */
Multidim::Array<ComputeType, nDim> regionFusionL0Approximation(Multidim::Array<T, nDim> const& original,
															   ComputeType lambda,
															   std::optional<int> channelDim=-1,
															   int maxIterations = 100) {

	typedef Multidim::Array<T, nDim> MDArray;
	typedef Multidim::Array<ComputeType, nDim> RMDArray;

	typedef Indexers::IndexPairMap IndexPairMap;
	typedef Indexers::FixedSizeDisjointSetForest FixedSizeDisjointSetForest;

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

	auto computeFlatIdx = [] (typename MDArray::IndexBlock const& idx,
							  typename MDArray::ShapeBlock const& shape,
							  int excludedDim) {

		int flatidx = 0;

		for (int i = 0; i < idx.size(); i++) {
			if (i != excludedDim) {
				int tmp = idx[i];
				for (int j = 0; j < i; j++) {
					if (j != excludedDim) {
						tmp *= shape[j];
					}
				}

				flatidx += tmp;
			}
		}

		return flatidx;

	};

	//initializations

	constexpr int baseNeighbors = 1 << nDim; //2^nDim
	constexpr int neighborsScalemargin = 2*baseNeighbors; //2^nDim

	FixedSizeDisjointSetForest groups(nPixs);

	std::vector<std::set<int>> neighbors(nPixs);

	std::set<int> groupIndices;
	IndexPairMap nConnections(nPixs*neighborsScalemargin); //connections between indices

	std::vector<typename MDArray::IndexBlock> groupIdxs(nPixs);

	typename MDArray::IndexBlock initial_mdarray_index;

	for (int i = 0; i < nDim; i++) {
		initial_mdarray_index[i] = 0;
	}
	typename MDArray::IndexBlock mdarray_index = initial_mdarray_index;

	bool go_on = true;

	while(go_on) {

		int flatId = computeFlatIdx(mdarray_index, original.shape(), excludedDim);

		groupIndices.insert(flatId);

		groupIdxs[flatId] = mdarray_index;

		for (int i = 0; i < nDim; i++) {
			if (i != excludedDim) {
				if (mdarray_index[i] + 1 >= 0 and mdarray_index[i] + 1 < original.shape()[i] ) {

					auto d_mdarray_index = mdarray_index;
					d_mdarray_index[i] += 1;
					int dFlatId = computeFlatIdx(d_mdarray_index, original.shape(), excludedDim);

					nConnections.setElement(flatId, dFlatId, 1);
					neighbors[flatId].insert(dFlatId);
					neighbors[dFlatId].insert(flatId);
				}
			}
		}

		//go to next index
		go_on = false;
		for (int i = 0; i < nDim; i++) {
			if (i != excludedDim) {
				mdarray_index[i]++;
				if (mdarray_index[i] < original.shape()[i]) { //if the current index could be incremented
					go_on = true;
					break; //continue
				} else { //else se the current index to 0 and move on to increment the next one.
					mdarray_index[i] = 0;
				}
			}
		}
		//if the loop has not been interrupted, go_on will be false and the main loop will be interrupted.
	}

	//actual optimization loop

	RMDArray approx(original.shape()); //initialize the solution from the original.

	mdarray_index = initial_mdarray_index;
	go_on = true;

	while(go_on) {

		approx.atUnchecked(mdarray_index) = ComputeType(original.valueUnchecked(mdarray_index));

		//go to next index
		go_on = false;
		for (int i = 0; i < nDim; i++) {
			mdarray_index[i]++;
			if (mdarray_index[i] < original.shape()[i]) { //if the current index could be incremented
				go_on = true;
				break; //continue
			} else { //else se the current index to 0 and move on to increment the next one.
				mdarray_index[i] = 0;
			}
		}
		//if the loop has not been interrupted, go_on will be false and the main loop will be interrupted.
	}

	for (int it = 0; it < maxIterations; it++) {

		ComputeType beta = ComputeType(1+it)/ComputeType(maxIterations) * lambda;

		std::set<int> schedule = groupIndices;

		for (int i : schedule) { //iterate over all groups

			if (groupIndices.find(i) == groupIndices.end()) { //the group has been removed after being scheduled
				continue;
			}

			typename MDArray::IndexBlock mdIndex = groupIdxs[i];

			std::set<int> n_schedule = neighbors[i];
			for (int n : n_schedule) { //check the group neighbors

				typename MDArray::IndexBlock n_mdIndex = groupIdxs[n];

				ComputeType squareDiff = 0;
				if (excludedDim >= 0 and excludedDim < nDim) {

					for (int l = 0; l < original.shape()[excludedDim]; l++) {
						mdIndex[excludedDim] = l;
						n_mdIndex[excludedDim] = l;

						ComputeType tmp = original.valueUnchecked(mdIndex) - original.valueUnchecked(n_mdIndex);
						squareDiff += tmp*tmp;
					}

				} else {
					ComputeType tmp = original.valueUnchecked(mdIndex) - original.valueUnchecked(n_mdIndex);
					squareDiff = tmp*tmp;
				}

				int sumGroupSize = groups.getGroupSize(i)+groups.getGroupSize(n);
				if(squareDiff*groups.getGroupSize(i)*groups.getGroupSize(n) <= beta*nConnections.getElementOrDefault(i,n,0)*sumGroupSize) {

					//recompute group mean
					if (excludedDim >= 0 and excludedDim < nDim) {

						for (int l = 0; l < original.shape()[excludedDim]; l++) {
							mdIndex[excludedDim] = l;
							n_mdIndex[excludedDim] = l;

							ComputeType wMean = groups.getGroupSize(i)*approx.atUnchecked(mdIndex) + groups.getGroupSize(n)*approx.atUnchecked(n_mdIndex);
							wMean /= sumGroupSize;
							approx.atUnchecked(mdIndex) = wMean;
						}

					} else {

						ComputeType wMean = groups.getGroupSize(i)*approx.atUnchecked(mdIndex) + groups.getGroupSize(n)*approx.atUnchecked(n_mdIndex);
						wMean /= sumGroupSize;
						approx.atUnchecked(mdIndex) = wMean;
					}

					int removedGroup = groups.joinNode(n,i);

					//join groups
					groupIndices.erase(removedGroup);

					//uptate neighbors
					neighbors[i].erase(removedGroup);

					for (int n : neighbors[removedGroup]) {

						if (n == i) {
							continue;
						}

						neighbors[n].erase(removedGroup);
						neighbors[n].insert(i);
						neighbors[i].insert(n);

						int nnConnection = nConnections.getElementOrDefault(i,n,0) + nConnections.getElement(n,removedGroup).value();

						nConnections.setElement(i,n, nnConnection);
					}


				}

			}

		}

	}

	//copy the mean value in each pixel of each group
	for (int i = 0; i < nPixs; i++) {

		int group = groups.getGroup(i);

		typename MDArray::IndexBlock mdIndex = groupIdxs[i];

		typename MDArray::IndexBlock g_mdIndex = groupIdxs[group];

		if (excludedDim >= 0 and excludedDim < nDim) {

			for (int l = 0; l < original.shape()[excludedDim]; l++) {
				mdIndex[excludedDim] = l;
				g_mdIndex[excludedDim] = l;
				approx.atUnchecked(mdIndex) = approx.atUnchecked(g_mdIndex);
			}

		} else {
			approx.atUnchecked(mdIndex) = approx.atUnchecked(g_mdIndex);
		}

	}

	return approx; //return the solution

}

} // namespace Optimization
} // namespace StereoVision

#endif // L0OPTIMIZATION_H
