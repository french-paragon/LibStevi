#ifndef STEREOVISION_GRAPHS_GRAPH_FLOW_H
#define STEREOVISION_GRAPHS_GRAPH_FLOW_H

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

#include "./graphs.h"

#include <queue>
#include <cassert>
#include <limits>

namespace StereoVision {

namespace GraphProcessing {

template<typename F_T>
struct maxFlowMinCutResults {
	F_T maxFlow;
	std::vector<int> minCutEdgesIdxs;
	int sourceVertexId;
	int targetVertexId;
};

template<typename GraphType>
/*!
 * \brief maxFlowMinCut solves both the max flow and min cut problems at the same time
 * \param graph the graph which should be optimized. Must be a directed graph.
 * \param sourceVertexId The index of the source vertex.
 * \param targetVertexId The index of the target vertex.
 * \return a struct containing the max flow, the indices of the edges in the cut and the source and target vertex index.
 */
maxFlowMinCutResults<typename GraphType::EdgeWeightType> maxFlowMinCut(GraphType const& graph,
																	   int sourceVertexId,
																	   int targetVertexId) {

	constexpr EdgeDirectedType edgeType = GraphType::edgeType;
	using WeightType = typename GraphType::EdgeWeightType;

	static_assert (edgeType == EdgeDirectedType::DirectedEdges, "maxFlowMinCut requires a directed graph");
	static_assert (!std::is_void_v<WeightType>, "maxFlowMinCut requires a weighted graph");

	int nVerts = graph.nVertices();
	int nEdges = graph.nEdges();

	graphVertexIndex<GraphType, VertexEdgePosition::SourceVertex> vertexIndex;
	graphVertexIndex<GraphType, VertexEdgePosition::TargetVertex> inverseVertexIndex;
	vertexIndex.buildIndex(graph);
	inverseVertexIndex.buildIndex(graph);

	WeightType flow = 0;

	std::vector<WeightType> flowUsed(nEdges);
	for (int i = 0; i < nEdges; i++) {
		flowUsed[i] = 0.0;
	}

	do {

		std::queue<int> verticesToTreat;
		verticesToTreat.push(sourceVertexId);

		std::vector<int> accessEdge(nVerts);
		for (int i = 0; i < nVerts; i++) {
			accessEdge[i] = -1;
		}

		bool targetReached = false;

		while (!targetReached and verticesToTreat.size() > 0) {

			int vId = verticesToTreat.front();
			verticesToTreat.pop();

			int nEdges = vertexIndex.nEdges(vId);
			int nInverseEdges = inverseVertexIndex.nEdges(vId);

			//forward flow
			for (int i = 0; i < nEdges; i++) {
				int edgeId = vertexIndex.nthEdge(vId, i);
				auto const& edge = graph.edge(edgeId);

				if (!edge.isValid()) {
					continue;
				}

				int target = edge.vertex2();

				assert(target != vId);
				assert(target < nVerts);

				if (accessEdge[target] < 0 and target != sourceVertexId and flowUsed[edgeId] < edge.weight()) {
					accessEdge[target] = edgeId;
					verticesToTreat.push(target);

					if (target == targetVertexId) {
						targetReached = true;
						break;
					}
				}
			}

			//pushback flow
			for (int i = 0; i < nInverseEdges; i++) {
				int edgeId = inverseVertexIndex.nthEdge(vId, i);
				auto const& edge = graph.edge(edgeId);

				if (!edge.isValid()) {
					continue;
				}

				int target = edge.vertex1();

				assert(target != vId);
				assert(target < nVerts);

				if (accessEdge[target] < 0 and target != sourceVertexId and flowUsed[edgeId] > 0) {
					accessEdge[target] = edgeId;
					verticesToTreat.push(target);

					if (target == targetVertexId) {
						targetReached = true;
						break;
					}
				}
			}

		}

		if (!targetReached) {
			break;
		}

		WeightType deltaFlow = std::numeric_limits<WeightType>::max;
		int minFlowEdgeId = -1;

		int vId = targetVertexId;

		//walk the path backwards
		while (vId != sourceVertexId and vId >= 0) {
			int edgeId = accessEdge[vId];
			auto const& edge = graph.edge(edgeId);

			int nextVid = edge.vertex1();

			if (nextVid == vId) {
				nextVid = edge.vertex2();

				if (deltaFlow > flowUsed[edgeId]) {
					deltaFlow = flowUsed[edgeId];
					minFlowEdgeId = edgeId;
				}

			} else {
				if (deltaFlow > edge.weight() - flowUsed[edgeId]) {
					deltaFlow = edge.weight() - flowUsed[edgeId];
					minFlowEdgeId = edgeId;
				}
			}



			vId = nextVid;
		}

		vId = targetVertexId;

		//subtract the flow delta
		while (vId != sourceVertexId and vId >= 0) {
			int edgeId = accessEdge[vId];
			auto const& edge = graph.edge(edgeId);

			int nextVid = edge.vertex1();

			if (nextVid == vId) { //going through the edge in reverse.
				nextVid = edge.vertex2();
				flowUsed[edgeId] -= deltaFlow;

				if (edgeId == minFlowEdgeId) {
					flowUsed[edgeId] = 0;
				}
			} else {
				flowUsed[edgeId] += deltaFlow;

				if (edgeId == minFlowEdgeId) {
					flowUsed[edgeId] = edge.weight();
				}
			}

			if (std::is_floating_point_v<WeightType>) {
				if (flowUsed[edgeId] > edge.weight()) {
					flowUsed[edgeId] = edge.weight();
				}
			}

			deltaFlow = std::min(deltaFlow, edge.weight() - flowUsed[edgeId]);

			vId = nextVid;
		}

		flow += deltaFlow;

	} while (true);


	maxFlowMinCutResults<typename GraphType::EdgeWeightType> ret;
	ret.maxFlow = flow;
	ret.sourceVertexId = sourceVertexId;
	ret.targetVertexId = targetVertexId;

	for (int i = 0; i < nEdges; i++) {
		if (flowUsed[i] >= graph.edgeWeight(i)) {
			ret.minCutEdgesIdxs.push_back(i);
		}
	}

	return ret;

}

} // namespace GraphProcessing
} // namespace StereoVision

#endif // GRAPH_FLOW_H
