#ifndef LIBSTEVI_FOREGROUNDSEGMENTATION_H
#define LIBSTEVI_FOREGROUNDSEGMENTATION_H

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

#include "../graphs/graphs.h"
#include "../graphs/graph_flow.h"
#include "../graphs/graph_cut.h"

#include "../utils/hash_utils.h"

#include <array>
#include <vector>
#include <limits>
#include <unordered_map>

#include <cassert>

namespace StereoVision {
namespace ImageProcessing {

namespace FgBgSegmentation {

enum MaskInfo {
	Foreground = 1,
	Background = 0
};

}

template<typename T_Cost>
/*!
 * \brief The MaskCostPolicy class correspond to a cost policy for global optimization of foreground/background segmentation.
 *
 * To be usable with graph-cut, the policy should be designed such that
 * globalCost(c1, 0, c2, 0) + globalCost(c1, 1, c2, 1) <= globalCost(c1, 0, c2, 1) + globalCost(c1, 1, c2, 0)
 * [cf Vladimir Kolmogorov and Ramin Zabih, "What Energy Functions can be Minimized via Graph Cuts?"]
 */
class MaskCostPolicy {
public:

	using CoordPairList = std::vector<std::pair<std::array<int,2>, std::array<int,2>>>;
	virtual CoordPairList coordinatePairs4Consideration(std::array<int,2> imageSize) const = 0;
	virtual T_Cost globalCost(std::array<int,2> coord1,
							  FgBgSegmentation::MaskInfo mask_value1,
							  std::array<int,2> coord2,
							  FgBgSegmentation::MaskInfo mask_value2) const = 0;
};

template<typename T_Cost>
class LocalNeighborhoodMaskCostPolicy : public MaskCostPolicy<T_Cost> {
public:

	using CoordPairList = typename MaskCostPolicy<T_Cost>::CoordPairList;
	virtual CoordPairList coordinatePairs4Consideration(std::array<int,2> imageSize) const override {

		int nPairs = 2*(imageSize[0]-1)*(imageSize[1]-1) + imageSize[0]-1 + imageSize[1]-1;

		if (nPairs <= 0) {
			return CoordPairList();
		}

		CoordPairList ret;
		ret.reserve(nPairs);

		for (int i = 0; i < imageSize[0]-1; i++) {
			for (int j = 0; j < imageSize[1]-1; j++) {

				std::array<int,2> base = {i,j};
				std::array<int,2> n1 = {i+1,j};
				std::array<int,2> n2 = {i,j+1};

				ret.emplace_back(base, n1);
				ret.emplace_back(base, n2);

			}
		}


		int j = imageSize[1]-1;

		for (int i = 0; i < imageSize[0]-1; i++) {

			std::array<int,2> base = {i,j};
			std::array<int,2> n1 = {i+1,j};

			ret.emplace_back(base, n1);

		}

		int i = imageSize[0]-1;

		for (int j = 0; j < imageSize[1]-1; j++) {

			std::array<int,2> base = {i,j};
			std::array<int,2> n2 = {i,j+1};

			ret.emplace_back(base, n2);

		}

		return ret;

	}

};

template<typename T_Cost>
class SmoothingMaskCostPolicy : public LocalNeighborhoodMaskCostPolicy<T_Cost> {
public:

	SmoothingMaskCostPolicy(T_Cost switchCost) :
		_switchCost(switchCost)
	{

	}

	virtual T_Cost globalCost(std::array<int,2> coord1,
							  FgBgSegmentation::MaskInfo mask_value1,
							  std::array<int,2> coord2,
							  FgBgSegmentation::MaskInfo mask_value2) const override {
		return costComputationImpl(coord1, mask_value1, coord2, mask_value2);
	}

protected:
	inline T_Cost costComputationImpl(std::array<int,2> coord1,
									  FgBgSegmentation::MaskInfo mask_value1,
									  std::array<int,2> coord2,
									  FgBgSegmentation::MaskInfo mask_value2) const {
		(void) coord1;
		(void) coord2;
		return (mask_value1 != mask_value2) ? _switchCost : 0;
	}

	T_Cost _switchCost;
};


template<typename T_Cost, typename T_Guide>
class GuidedMaskCostPolicy : public SmoothingMaskCostPolicy<T_Cost> {
public:

	GuidedMaskCostPolicy(T_Cost switchCost,
						 Multidim::Array<T_Guide, 3> const& guideRef) :
		SmoothingMaskCostPolicy<T_Cost>(switchCost),
		_guide_ref(guideRef)
	{
		_max_abs_diff = 0;
		_min_abs_diff = std::numeric_limits<float>::infinity();

		std::array<int,3> shape = guideRef.shape();
		std::array<int,2> imShape = {shape[0], shape[1]};

		auto pairs = this->coordinatePairs4Consideration(imShape);

		for (std::pair<std::array<int,2>, std::array<int,2>> const& pair : pairs) {
			float val = this->computeGuideAbsDiff(pair.first, pair.second);

			if (val < _min_abs_diff) {
				_min_abs_diff = val;
			}

			if (val > _max_abs_diff) {
				_max_abs_diff = val;
			}
		}
	}

	virtual T_Cost globalCost(std::array<int,2> coord1,
							   FgBgSegmentation::MaskInfo mask_value1,
							   std::array<int,2> coord2,
							   FgBgSegmentation::MaskInfo mask_value2) const override {

		float delta = this->computeGuideAbsDiff(coord1, coord2);
		float weight = (_max_abs_diff - delta)/(_max_abs_diff - _min_abs_diff);

		return SmoothingMaskCostPolicy<T_Cost>::costComputationImpl(coord1, mask_value1, coord2, mask_value2)*weight;
	}

protected :

	inline float computeGuideAbsDiff(std::array<int,2> coord1, std::array<int,2> coord2) const {
		float absDiff = 0;

		int nChannels = _guide_ref.shape()[2];
		for (int c = 0; c < nChannels; c++) {
			absDiff += std::abs(_guide_ref.valueUnchecked(coord1[0], coord1[1], c) -
								_guide_ref.valueUnchecked(coord2[0], coord2[1], c));
		}

		return absDiff;
	}

	float _max_abs_diff;
	float _min_abs_diff;

	Multidim::Array<T_Guide, 3> const& _guide_ref;
};

namespace FgBgSegmentation {

template<typename T_Cost>
using OptimizableGraph = GraphProcessing::Graph<GraphProcessing::EdgeDirectedType::DirectedEdges, void, T_Cost>;

template<typename T_Cost>
OptimizableGraph<T_Cost> buildGraph(Multidim::Array<T_Cost, 3> const& cost,
									MaskCostPolicy<T_Cost> const& global_cost_policy) {

	auto shape = cost.shape();
	int nVertices = shape[0]*shape[1]+2;

	FgBgSegmentation::OptimizableGraph<T_Cost> graph(nVertices);

	int sourceId = nVertices-2;
	int targetId = nVertices-1;

	std::unordered_map<std::array<int, 2>, int> edgeIndex;
	edgeIndex.reserve(6*nVertices);

	//build base cost edges

	Multidim::IndexConverter<2> idxConverter({shape[0], shape[1]});

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {

			int vertexId = idxConverter.getPseudoFlatIdFromIndex({i,j});

			T_Cost w_0 = cost.valueUnchecked(i,j,Background);
			T_Cost w_1 = cost.valueUnchecked(i,j,Foreground);

			if (w_1 >= w_0) {
				int edgeId = graph.linkVertices(sourceId, vertexId, w_1 - w_0);
				edgeIndex[std::array<int, 2>({sourceId, vertexId})] = edgeId;
			} else {
				int edgeId = graph.linkVertices(vertexId, targetId, w_0 - w_1);
				edgeIndex[std::array<int, 2>({vertexId, targetId})] = edgeId;
			}

		}
	}

	//build global cost edges

	auto pairs = global_cost_policy.coordinatePairs4Consideration({shape[0], shape[1]});

	for (std::pair<std::array<int, 2>, std::array<int, 2>> const & pair : pairs) {

		int vertexId1 = idxConverter.getPseudoFlatIdFromIndex(pair.first);
		int vertexId2 = idxConverter.getPseudoFlatIdFromIndex(pair.second);

		std::array<int, 2> pos1 = pair.first;
		std::array<int, 2> pos2 = pair.second;

		if (vertexId1 > vertexId2) {
			std::swap(vertexId1, vertexId2);
			std::swap(pos1, pos2);
		}

		T_Cost w_00 = global_cost_policy.globalCost(pos1, Background, pos2, Background);
		T_Cost w_10 = global_cost_policy.globalCost(pos1, Foreground, pos2, Background);
		T_Cost w_01 = global_cost_policy.globalCost(pos1, Background, pos2, Foreground);
		T_Cost w_11 = global_cost_policy.globalCost(pos1, Foreground, pos2, Foreground);

		assert(w_10 + w_01 >= w_11 + w_00); //important !

		//quadratic part
		if (w_10 + w_01 - w_11 - w_00 > 0) {
			int edgeId = graph.linkVertices(vertexId1, vertexId2, w_10 + w_01 - w_11 - w_00);
			edgeIndex[std::array<int, 2>({vertexId1, vertexId2})] = edgeId;
		}

		//linear part vertex1

		int vId1;
		int vId2;

		T_Cost cost;

		if (w_10 >= w_00) {

			vId1 = sourceId;
			vId2 = vertexId1;

			cost = w_10 - w_00;

		} else if (w_00 > w_10) {

			vId1 = vertexId1;
			vId2 = targetId;

			cost = w_00 - w_10;
		}

		int nEdgeId;

		if (edgeIndex.count({vId1, vId2}) > 0) {
			nEdgeId = edgeIndex[{vId1, vId2}];
			graph.increaseEdgeWeight(nEdgeId, cost);
		} else {
			nEdgeId = graph.linkVertices(vId1, vId2, cost);
			edgeIndex[std::array<int, 2>({vId1, vId2})] = nEdgeId;
		}

		//linear part vertex2

		if (w_11 >= w_10) {

			vId1 = sourceId;
			vId2 = vertexId2;

			cost = w_11 - w_10;

		} else if (w_10 > w_11) {

			vId1 = vertexId2;
			vId2 = targetId;

			cost = w_10 - w_11;
		}

		if (edgeIndex.count({vId1, vId2}) > 0) {
			nEdgeId = edgeIndex[{vId1, vId2}];
			graph.increaseEdgeWeight(nEdgeId, cost);
		} else {
			nEdgeId = graph.linkVertices(vId1, vId2, cost);
			edgeIndex[std::array<int, 2>({vId1, vId2})] = nEdgeId;
		}

	}

	return graph;

}

}

template<typename T_Cost>
Multidim::Array<FgBgSegmentation::MaskInfo, 2> getGlobalRefinedMask(Multidim::Array<T_Cost, 3> const& cost,
																	MaskCostPolicy<T_Cost> const& global_cost_policy) {

	FgBgSegmentation::OptimizableGraph<T_Cost> graph = FgBgSegmentation::buildGraph(cost, global_cost_policy);

	int nVertices = graph.nVertices();

	std::array<int, 2> shape = {cost.shape()[0], cost.shape()[1]};

	assert(nVertices == shape[0]*shape[1]+2);


	Multidim::IndexConverter<2> idxConverter({shape[0], shape[1]});

	int sourceVertexId = nVertices-2;
	int targetVertexId = nVertices-1;

	auto maxFlowMinCut = GraphProcessing::maxFlowMinCut(graph,
														sourceVertexId,
														targetVertexId);

	std::vector<int> reachableVertices = GraphProcessing::reachableVerticesInCut(graph, maxFlowMinCut.minCutEdgesIdxs, targetVertexId);

	Multidim::Array<FgBgSegmentation::MaskInfo, 2> ret(shape);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			ret.atUnchecked(i,j) = FgBgSegmentation::Background;
		}
	}

	for (int vId : reachableVertices) {

		if (vId == targetVertexId) {
			continue;
		}

		assert(vId != sourceVertexId);

		std::array<int,2> pos = idxConverter.getIndexFromPseudoFlatId(vId);

		ret.atUnchecked(pos) = FgBgSegmentation::Foreground;
	}

	return ret;

}


} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_FOREGROUNDSEGMENTATION_H
