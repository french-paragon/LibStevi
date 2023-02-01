#ifndef STEREOVISION_GRAPHS_H
#define STEREOVISION_GRAPHS_H

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

#include <type_traits>
#include <cinttypes>

#include <vector>
#include <optional>
#include <array>
#include <unordered_map>

#include "../utils/hash_utils.h"

namespace StereoVision {

namespace GraphProcessing {

enum EdgeDirectedType {
	UndirectedEdges,
	DirectedEdges
};

enum VertexEdgePosition {
	SourceVertex,
	TargetVertex
};

enum GraphMovingDirection {
	Forward,
	Backward,
	Both
};

inline constexpr VertexEdgePosition invertVertexEdgePosition(VertexEdgePosition vType) {
	return (vType == SourceVertex) ? TargetVertex: SourceVertex;
}

template<typename VD_T>
class GraphVertex {
public:

	GraphVertex() :
		_id(-1),
		_data()
	{

	}

	GraphVertex(int id, VD_T data) :
		_id(id),
		_data(data)
	{

	}

	inline bool isValid() const {
		return _id >= 0;
	}

	inline int id() const {
		return _id;
	}

	inline VD_T const& data() const {
		return _data;
	}

	inline VD_T& data() {
		return _data;
	}

protected:
	int _id;
	VD_T _data;

};

template<>
class GraphVertex<void> {
public:

	GraphVertex() :
		_id(-1)
	{

	}

	GraphVertex(int id) :
		_id(id)
	{

	}

	inline bool isValid() const {
		return _id >= 0;
	}

	inline int id() const {
		return _id;
	}

	inline void data() const {
		return;
	}

protected:
	int _id;

};


template<typename WT>
class GraphEdge {
public:
	static_assert (std::is_arithmetic_v<WT> or std::is_void_v<WT>, "Must have an arithmetic type as a weight, or void for no weight");
	using WeightType = WT;

	GraphEdge() :
		_vertex1_id(-1),
		_vertex2_id(-1),
		_weight(0)
	{

	}

	GraphEdge(int v1_id, int v2_id, WeightType w = 1) :
		_vertex1_id(v1_id),
		_vertex2_id(v2_id),
		_weight(w)
	{

	}

	inline bool isValid() const {
		return _vertex1_id >= 0 and _vertex2_id >= 0 and _vertex1_id != _vertex2_id;
	}

	inline int vertex1() const {
		return _vertex1_id;
	}

	inline int vertex2() const {
		return _vertex2_id;
	}

	inline int vertex(VertexEdgePosition vType) const {
		return (vType == SourceVertex) ? _vertex1_id : _vertex2_id;
	}


	inline WeightType weight() const {
		return _weight;
	}

	inline void setWeight(WeightType const& w) {
		_weight = w;
	}

protected:
	int _vertex1_id;
	int _vertex2_id;
	WeightType _weight;
};

template<>
class GraphEdge<void> {
public:
	using WeightType = uint8_t;

	GraphEdge() :
		_vertex1_id(-1),
		_vertex2_id(-1)
	{

	}

	GraphEdge(int v1_id, int v2_id, WeightType w = 1) :
		_vertex1_id(v1_id),
		_vertex2_id(v2_id)
	{
		(void) w;
	}

	inline bool isValid() const {
		return _vertex1_id >= 0 and _vertex2_id >= 0 and _vertex1_id != _vertex2_id;
	}

	inline int vertex1() const {
		return _vertex1_id;
	}

	inline int vertex2() const {
		return _vertex2_id;
	}

	inline int vertex(VertexEdgePosition vType) const {
		return (vType == SourceVertex) ? _vertex1_id : _vertex2_id;
	}


	inline WeightType weight() const {
		return 1;
	}

	inline void setWeight(WeightType const& w) {
		(void) w;
	}

protected:
	int _vertex1_id;
	int _vertex2_id;
};

template<EdgeDirectedType E_T, typename VD_T = void, typename EW_T = void>
class Graph {
public:

	static const EdgeDirectedType edgeType = E_T;

	using VertexT = GraphVertex<VD_T>;
	using EdgeT = GraphEdge<EW_T>;

	using VertexDataType = VD_T;
	using EdgeWeightType = EW_T;

	Graph() {

	}

	Graph(int nVertex) {
		_vertices.reserve(nVertex);

		for (int i = 0; i < nVertex; i++) {
			_vertices.emplace_back(i);
		}
	}

	template<typename pV_T>
	Graph(int nVertex, std::enable_if_t<!std::is_void_v<pV_T> and std::is_same_v<pV_T, VD_T>, VD_T> val = VD_T()) {
		_vertices.reserve(nVertex);

		for (int i = 0; i < _vertices.size(); i++) {
			_vertices.emplace_back(i,val);
		}
	}

	Graph(Graph const& other) :
		_vertices(other._vertices),
		_edges(other._edges)
	{

	}

	Graph(Graph && other) :
		_vertices(std::move(other._vertices)),
		_edges(std::move(other._edges))
	{

	}

	inline int nVertices() const {
		return _vertices.size();
	}


	inline int nEdges() const {
		return _edges.size();
	}

	inline VertexT const& vertex(int i) const {
		return _vertices[i];
	}

	inline std::conditional_t<std::is_void_v<VD_T>, void, std::conditional_t<std::is_void_v<VD_T>, int, VD_T>&> vertexData(int i) {
		return _vertices[i].data();
	}

	inline EdgeT const& edge(int i) const {
		return _edges[i];
	}

	inline typename EdgeT::WeightType edgeWeight(int i) const {
		return _edges[i].weight();
	}

	inline void setEdgeWeight(int i, typename EdgeT::WeightType w) {
		_edges[i].setWeight(w);
	}

	inline void increaseEdgeWeight(int i, typename EdgeT::WeightType w) {
		typename EdgeT::WeightType oldWeight = _edges[i].weight();
		_edges[i].setWeight(oldWeight + w);
	}

	/*!
	 * \brief edgeBetweenVerticesId
	 * \param idv1
	 * \param idv2
	 * \return the index of the first edge going trough both vertices.
	 *
	 * This function perform an exaustive search, and as such should not be called repeatedly.
	 * If you need to find the index of an edge based on the vertices id repeatedly,
	 * it is better to maintain your own data structure next to the graph.
	 */
	inline std::optional<int> edgeBetweenVerticesId(int idv1, int idv2) {

		for (int i = 0; i < _edges.size(); i++) {

			if (!_edges[i].isValid()) {
				continue;
			}

			if (_edges[i].vertex1() == idv1 and _edges[i].vertex2() == idv2) {
				return i;
			}

			if (edgeType == EdgeDirectedType::UndirectedEdges) {
				if (_edges[i].vertex2() == idv1 and _edges[i].vertex1() == idv2) {
					return i;
				}
			}
		}

		return std::nullopt;
	}

	/*!
	 * \brief linkVertices link two vertices together and return a reference to the corresponding edge
	 * \param idv1 The id of the first vertex.
	 * \param idv2 The id of the second vertex.
	 * \param w The weight of the edge.
	 * \return The index of the newly created edge.
	 *
	 * ! The function will emplace a new edge at the end of the graph, even if an edge already exist between these vertices.
	 */
	inline int linkVertices(int idv1, int idv2, typename EdgeT::WeightType w = 1) {
		int id = _edges.size();
		_edges.emplace_back(idv1, idv2, w);
		return id;
	}

protected:
	std::vector<VertexT> _vertices;
	std::vector<EdgeT> _edges;
};

template<typename GraphType, VertexEdgePosition vType>
/*!
 * \brief The graphVertexIndex class index the vertices in a graph so that the list of corresponding edges id can be accessed in constant time.
 */
class graphVertexIndex {
public:
	graphVertexIndex() {

	}

	graphVertexIndex(graphVertexIndex const& other) :
		_startingIdxs(other._startingIdxs),
		_sortedEdgeIdxs(other._sortedEdgeIdxs)
	{

	}

	graphVertexIndex(graphVertexIndex && other) :
		_startingIdxs(std::move(other._startingIdxs)),
		_sortedEdgeIdxs(std::move(other._sortedEdgeIdxs))
	{

	}

	void buildIndex(GraphType const& g) {

		EdgeDirectedType edgeType = GraphType::edgeType;

		int nEdgeIdxs = g.nEdges();

		if (edgeType == EdgeDirectedType::UndirectedEdges) {
			nEdgeIdxs *= 2;
		}

		_startingIdxs.resize(g.nVertices()+1);
		_sortedEdgeIdxs.resize(nEdgeIdxs);

		//init the vertex edge counts
		for (int i = 0; i < _startingIdxs.size(); i++) {
			_startingIdxs[i] = 0;
		}

		//count the edges per vertex
		for (int i = 0; i < g.nEdges(); i++) {
			auto const& edge = g.edge(i);

			int vId = edge.vertex(vType);
			int aVid = edge.vertex(invertVertexEdgePosition(vType));

			if (vId < 0 or vId >= g.nVertices() or aVid < 0 or aVid >= g.nVertices()) {
				continue; //invalid edge
			}

			_startingIdxs[vId] += 1;

			if (edgeType == EdgeDirectedType::UndirectedEdges) {
				_startingIdxs[aVid] += 1;
			}
		}

		//transform the count into starting idx in the index

		int sum = 0;

		for (int i = 0; i < g.nVertices(); i++) {
			int tmp = sum + _startingIdxs[i];
			_startingIdxs[i] = sum;
			sum = tmp;
		}
		_startingIdxs[g.nVertices()] = sum;

		//build the index
		std::vector<int> currentIdxs = _startingIdxs;

		for (int i = 0; i < g.nEdges(); i++) {
			auto const& edge = g.edge(i);

			int vId = edge.vertex(vType);
			int aVid = edge.vertex(invertVertexEdgePosition(vType));

			if (vId < 0 or vId >= g.nVertices() or aVid < 0 or aVid >= g.nVertices()) {
				continue; //invalid edge
			}

			_sortedEdgeIdxs[currentIdxs[vId]] = i;
			currentIdxs[vId] += 1;

			if (edgeType == EdgeDirectedType::UndirectedEdges) {
				_sortedEdgeIdxs[currentIdxs[aVid]] = i;
				currentIdxs[aVid] += 1;
			}
		}

	}

	inline int nEdges(int vId) {
		if (vId < 0 or vId+1 >= _startingIdxs.size()) {
			return -1;
		}

		return _startingIdxs[vId+1] - _startingIdxs[vId];
	}

	inline int nthEdge(int vId, int nthEdge) {

		if (nthEdge >= nEdges(vId) or nthEdge < 0) {
			return -1;
		}

		int eId = _startingIdxs[vId]+nthEdge;
		return _sortedEdgeIdxs[eId];
	}

protected:
	std::vector<int> _startingIdxs;
	std::vector<int> _sortedEdgeIdxs;
};

template<typename GraphType, VertexEdgePosition vType, typename AssociativeContainer = std::unordered_map<std::array<int, 2>, std::vector<int>>>
/*!
 * \brief The graphEdgeIndex class index the edges of a graph from the possible pairs of vertices.
 */
class graphEdgeIndex {
public:
	graphEdgeIndex() {

	}

	graphEdgeIndex(graphEdgeIndex const& other) :
		_index(other._index)
	{

	}

	graphEdgeIndex(graphEdgeIndex && other) :
		_index(std::move(other._index))
	{

	}

	void buildIndex(GraphType const& g) {

		EdgeDirectedType edgeType = GraphType::edgeType;

		for (int i = 0; i < g.nEdges(); i++) {
			auto const& edge = g.edge(i);

			if (!edge.isValid()) {
				continue;
			}

			int v1 = edge.vertex1();
			int v2 = edge.vertex2();

			if (edgeType == EdgeDirectedType::UndirectedEdges and v1 > v2) {
				int tmp = v1;
				v1 = v2;
				v2 = tmp;
			}

			if(_index.count({v1,v2}) == 0) {
				_index[{v1,v2}] = std::vector<int>();
			}
			_index[{v1,v2}].push_back(i);

		}
	}

	inline std::vector<int> edgeList(std::array<int, 2> const& vertices) {

		EdgeDirectedType edgeType = GraphType::edgeType;

		int v1 = vertices[0];
		int v2 = vertices[1];

		if (edgeType == EdgeDirectedType::UndirectedEdges and v1 > v2) {
			int tmp = v1;
			v1 = v2;
			v2 = tmp;
		}

		if(_index.count({v1,v2}) == 0) {
			return std::vector<int>();
		}
		return _index[{v1,v2}];
	}

protected:
	AssociativeContainer _index;
};

} // namespace GraphProcessing
} // namespace StereoVision

#endif // STEREOVISION_GRAPHS_H
