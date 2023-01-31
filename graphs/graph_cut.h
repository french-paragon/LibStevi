#ifndef STEREOVISION_GRAPHS_GRAPH_CUT_H
#define STEREOVISION_GRAPHS_GRAPH_CUT_H

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
#include <unordered_set>

namespace StereoVision {

namespace GraphProcessing {


template<typename GraphType, GraphMovingDirection movingDirection = Forward>
std::vector<int> reachableVerticesInCut(GraphType const& graph, std::vector<int> const& cutEdgesId, int startVertex) {

	constexpr EdgeDirectedType edgeType = GraphType::edgeType;

	constexpr VertexEdgePosition ActualSourceVertex = (movingDirection == Forward) ? VertexEdgePosition::SourceVertex : VertexEdgePosition::TargetVertex;
	constexpr VertexEdgePosition ActualTargetVertex = invertVertexEdgePosition(ActualSourceVertex);

	std::unordered_set<int> cutEdgesIdxIndex(cutEdgesId.begin(), cutEdgesId.end());

	graphVertexIndex<GraphType, ActualSourceVertex> vertexIndex;
	vertexIndex.buildIndex(graph);

	if (startVertex < 0 or startVertex >= graph.nVertices()) {
		return {};
	}

	std::vector<bool> reached(graph.nVertices());
	for (int i = 0; i < graph.nVertices(); i++) {
		reached[i] = false;
	}

	reached[startVertex] = true;

	std::queue<int> verticesToTreat;
	verticesToTreat.push(startVertex);

	int count = 0;

	while (verticesToTreat.size() > 0) {
		int vId = verticesToTreat.front();
		verticesToTreat.pop();

		int nEdges = vertexIndex.nEdges(vId);

		//forward flow
		for (int i = 0; i < nEdges; i++) {
			int edgeId = vertexIndex.nthEdge(vId, i);

			if (cutEdgesIdxIndex.contains(edgeId)) {
				continue; //ignore cutted edges
			}

			auto const& edge = graph.edge(edgeId);

			if (!edge.isValid()) {
				continue;
			}

			int target = edge.vertex(ActualTargetVertex);

			if (target == vId and edgeType == EdgeDirectedType::UndirectedEdges) {
				target = edge.vertex(ActualSourceVertex);
			}

			if (target == vId) {
				continue;
			}

			if (reached[target]) {
				continue;
			}

			reached[target] = true;
			verticesToTreat.push(target);
			count++;
		}
	}

	std::vector<int> reachedIdxs;
	reachedIdxs.reserve(count);

	for (int i = 0; i < graph.nVertices(); i++) {
		if (reached[i]) {
			reachedIdxs.push_back(i);
		}
	}

	return reachedIdxs;

}

} // namespace GraphProcessing
} // namespace StereoVision

#endif // STEREOVISION_GRAPHS_GRAPH_CUT_H
