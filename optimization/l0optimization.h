#ifndef LIBSTEVI_L0_OPTIMIZATION_H
#define LIBSTEVI_L0_OPTIMIZATION_H

#include <MultidimArrays/MultidimArrays.h>

#include <optional>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

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

	class IndexPairMap {

	public:
		typedef int index_t;

	private:

		typedef std::pair<index_t, index_t> indexPair;

		struct hashPair {
			std::size_t operator()(indexPair const& pair) const {
				return std::hash<index_t>{}(pair.first) | std::hash<index_t>{}(pair.second);
			}
		};

		std::unordered_map<indexPair, int, hashPair> _mapping;

		inline indexPair getPair(index_t id1, index_t id2) {
			return {std::min(id1, id2), std::max(id1, id2)};
		}

	public:
		IndexPairMap(int initialSize) : _mapping(initialSize) {

		}

		std::optional<int> getElement(index_t id1, index_t id2) {

			indexPair p = getPair(id1, id2);

			if (_mapping.count(p) > 0) {
				return _mapping[p];
			}

			return std::nullopt;
		}

		int getElementOrDefault(index_t id1, index_t id2, int defaultValue) {

			indexPair p = getPair(id1, id2);

			if (_mapping.count(p) > 0) {
				return _mapping[p];
			}

			return defaultValue;
		}

		bool hasElement(index_t id1, index_t id2) {
			indexPair p = getPair(id1, id2);

			return _mapping.count(p) > 0;
		}

		void setElement(index_t id1, index_t id2, int value) {
			indexPair p = getPair(id1, id2);

			_mapping.insert({p, value});
		}

		void clearElement(index_t id1, index_t id2) {
			indexPair p = getPair(id1, id2);

			_mapping.erase(p);
		}
	};

	class FixedSizeDisjointSetForest {
	public:
		FixedSizeDisjointSetForest(int nElements) :
			_elements(nElements),
			_groupSize(nElements)
		{
			for (int i = 0; i < nElements; i++) {
				_elements[i] = i;
			}
			std::fill(_groupSize.begin(), _groupSize.end(), 1);
		}

		int getGroup(int element) {
			std::vector<int> path;
			int pos = element;

			while(_elements[pos] != pos) {//node is not pointing to itself
				path.push_back(pos);
				pos = _elements[pos];
			}

			for (int node : path) {
				_elements[node] = pos;
			}

			return pos;
		}

		int getGroupSize(int element) {
			int group = getGroup(element);
			return _groupSize[group];
		}

		/*!
		 * \brief joinNode join the group of the source node in the group of the target node
		 * \param source the source node
		 * \param target the target node
		 * \return the index of the removed group (the group of the source node).
		 */
		int joinNode(int source, int target) {
			int sourceGroup = getGroup(source);
			int targetGroup = getGroup(target);
			_elements[sourceGroup] = targetGroup;
			_groupSize[targetGroup] += _groupSize[sourceGroup];
			return sourceGroup;
		}

	private:
		std::vector<int> _elements;
		std::vector<int> _groupSize;
	};

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
				int tmp = i;
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
				if (mdarray_index[i] < original.shape()[i]) {
					go_on = true;
					break;
				} else {
					mdarray_index[i] = 0;
				}
			}
		}
	}

	//actual optimization loop

	RMDArray approx(original.shape()); //initialize the solution from the original.

	for (int i = 0; i < original.flatLenght(); i++) {
		approx.atUnchecked(i) = ComputeType(original.valueUnchecked(i));
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

						ComputeType tmp = squareDiff = original.valueUnchecked(mdIndex) - original.valueUnchecked(n_mdIndex);
						squareDiff += tmp*tmp;
					}

				} else {
					ComputeType tmp = squareDiff = original.valueUnchecked(mdIndex) - original.valueUnchecked(n_mdIndex);
					squareDiff = tmp*tmp;
				}

				int sumGroupSize = groups.getGroupSize(i)+groups.getGroupSize(n);
				if(squareDiff*groups.getGroupSize(i)*groups.getGroupSize(n) <= beta*nConnections.getElementOrDefault(i,n,0)*sumGroupSize) {
					int removedGroup = groups.joinNode(n,i);

					//join groups
					groupIndices.erase(removedGroup);

					//uptate neighbors
					neighbors[i].erase(n);
					for (int n : neighbors[removedGroup]) {
						neighbors[n].erase(removedGroup);
						neighbors[n].insert(i);
						neighbors[i].insert(n);

						int nnConnection = nConnections.getElementOrDefault(i,n,0) + nConnections.getElement(n,removedGroup).value();

						nConnections.setElement(i,n, nnConnection);
					}

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
