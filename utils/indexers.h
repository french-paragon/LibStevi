#ifndef INDEXERS_H
#define INDEXERS_H

#include <optional>
#include <vector>
#include <unordered_map>

namespace StereoVision {

/*!
 * The indexers namespace contain a series of utility classes mean at organizing, tracking and, obviously, indexing data.
 */
namespace Indexers {

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

	inline std::optional<int> getElement(index_t id1, index_t id2) {

		indexPair p = getPair(id1, id2);

		if (_mapping.count(p) > 0) {
			return _mapping[p];
		}

		return std::nullopt;
	}

	inline int getElementOrDefault(index_t id1, index_t id2, int defaultValue) {

		indexPair p = getPair(id1, id2);

		if (_mapping.count(p) > 0) {
			return _mapping[p];
		}

		return defaultValue;
	}

	inline bool hasElement(index_t id1, index_t id2) {
		indexPair p = getPair(id1, id2);

		return _mapping.count(p) > 0;
	}

	inline void setElement(index_t id1, index_t id2, int value) {
		indexPair p = getPair(id1, id2);

		_mapping.insert({p, value});
	}

	inline void clearElement(index_t id1, index_t id2) {
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

	inline int getGroup(int element) {
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

	inline int getGroupSize(int element) {
		int group = getGroup(element);
		return _groupSize[group];
	}

	/*!
	 * \brief joinNode join the group of the source node in the group of the target node
	 * \param source the source node
	 * \param target the target node
	 * \return the index of the removed group (the group of the source node).
	 */
	inline int joinNode(int source, int target) {
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


class GrowingSizeDisjointSetForest {
public:
	GrowingSizeDisjointSetForest(int nInitialElements = 0) :
		_elements(nInitialElements),
		_groupSize(nInitialElements)
	{
		for (int i = 0; i < nInitialElements; i++) {
			_elements[i] = i;
		}
		std::fill(_groupSize.begin(), _groupSize.end(), 1);
	}

	inline int nElements() const {
		return _elements.size();
	}

	inline int getGroup(int element) {
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

	inline int getGroupSize(int element) {
		int group = getGroup(element);
		return _groupSize[group];
	}

	/*!
	 * \brief joinNode join the group of the source node in the group of the target node
	 * \param source the source node
	 * \param target the target node
	 * \return the index of the removed group (the group of the source node).
	 */
	inline int joinNode(int source, int target) {
		int sourceGroup = getGroup(source);
		int targetGroup = getGroup(target);
		_elements[sourceGroup] = targetGroup;
		_groupSize[targetGroup] += _groupSize[sourceGroup];
		return sourceGroup;
	}

	/*!
	 * \brief addNode add a node to the Disjoint set forest
	 * \return the index of the new element (forming a single group).
	 */
	inline int addNode() {
		int elemId = _elements.size();
		_elements.push_back(elemId);
		_groupSize.push_back(1);
		return elemId;
	}

	/*!
	 * \brief addNodes add a number of nodes to the Disjoint set forest
	 * \return the largest index of the new elements (all forming a single group), or -1 if no element was added.
	 */
	inline int addNodes(int n) {
		int ret = -1;
		for (int i = 0; i < n; i++) {
			ret = addNode();
		}

		return ret;
	}

	/*!
	 * \brief addNodesToIndex add a number of nodes to the Disjoint set forest to reach a given index.
	 * \return the largest index of the new elements (all forming a single group), or -1 if no element was added.
	 */
	inline int addNodesToIndex(int target) {
		return addNodes(target - nElements() + 1);
	}

private:
	std::vector<int> _elements;
	std::vector<int> _groupSize;
};

} // namespace Indexers

} // namespace StereoVision

#endif // INDEXERS_H
