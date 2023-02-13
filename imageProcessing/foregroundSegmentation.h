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

#include "./histogram.h"

#include <array>
#include <vector>
#include <limits>
#include <unordered_map>
#include <cinttypes>
#include <optional>

#include <cassert>

namespace StereoVision {
namespace ImageProcessing {

template<typename ImT>
/*!
 * \brief computeOtsuThreshold compute otsu's threshold
 * \param histogram the histogram to use for computation
 * \return the threshold, or nullopt if an error occur
 */
std::optional<ImT> computeOtsuThreshold(Histogram<ImT> const& histogram) {

	int total = histogram.getTotalCount();
	int nBins = histogram.nBins();

	if (nBins <= 0) {
		return std::nullopt;
	}

	if (histogram.nChannels() != 1) { // support only single channel histogram so far
		return std::nullopt;
	}

	int meanBackground = 0;
	int probBackground = 0;
	int meanTotal = 0;

	for (int i = 0; i < nBins; i++) {
		meanTotal += i*histogram.getBinCount(i);
	}

	int level = 0;
	float maxInterClassVariance = 0.0;

	for (int i = 0; i < nBins; i++) {
		int probForeground = total - probBackground;
		if (probBackground > 0 and probForeground > 0) {

			float mF = float(meanTotal - meanBackground) / probForeground;
			float mDelta = (float(meanBackground) / probBackground) - mF;

			float interClassVariance = probBackground * probForeground * mDelta*mDelta;

			if ( interClassVariance > maxInterClassVariance ) {
				level = i;
				maxInterClassVariance = interClassVariance;
			}
		}
		probBackground += histogram.getBinCount(i);
		meanBackground = meanBackground + i * histogram.getBinCount(i);
	}

	return histogram.getBinLowerBound(level);
}

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

	virtual ~MaskCostPolicy() {}

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
						 Multidim::Array<T_Guide, 3> const& guideRef,
						 T_Cost minSwitchCost = 0) :
		SmoothingMaskCostPolicy<T_Cost>(std::max(switchCost, minSwitchCost) - std::min(switchCost, minSwitchCost)),
		_guide_ref(guideRef),
		_minSwitchCost(std::min(switchCost, minSwitchCost))
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

		return ((mask_value1 != mask_value2) ? _minSwitchCost : 0) +
				SmoothingMaskCostPolicy<T_Cost>::costComputationImpl(coord1, mask_value1, coord2, mask_value2)*weight;
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

	T_Cost _minSwitchCost;
};



namespace FgBgSegmentation {

template<typename T_Cost>
using OptimizableGraph = GraphProcessing::Graph<GraphProcessing::EdgeDirectedType::DirectedEdges, void, T_Cost>;

template<typename T_Cost>
using OptimizableIndexedGraph = GraphProcessing::Graph<GraphProcessing::EdgeDirectedType::DirectedEdges, std::array<int,2>, T_Cost>;

namespace Internal {

template<typename T_Graph, typename T_Cost>
inline void addLinearCost(T_Graph & graph,
						  int vertexId,
						  int sourceVertexId,
						  int targetVertexId,
						  T_Cost costVertexForeground,
						  T_Cost costVertexBackground,
						  std::unordered_map<std::array<int, 2>, int> & edgeIndex) {

	if (costVertexForeground == costVertexBackground) {
		return;
	}

	T_Cost cost;
	int vertex1Id;
	int vertex2Id;

	if (costVertexForeground > costVertexBackground) {
		cost = costVertexForeground - costVertexBackground;
		vertex1Id = sourceVertexId;
		vertex2Id = vertexId;
	} else {
		cost = costVertexBackground - costVertexForeground;
		vertex1Id = vertexId;
		vertex2Id = targetVertexId;
	}

	if (edgeIndex.count(std::array<int, 2>({vertex1Id, vertex2Id})) <= 0) {
		int edgeId = graph.linkVertices(vertex1Id, vertex2Id, cost);
		edgeIndex[std::array<int, 2>({vertex1Id, vertex2Id})] = edgeId;
	} else {
		int edgeId = edgeIndex[std::array<int, 2>({vertex1Id, vertex2Id})];
		graph.increaseEdgeWeight(edgeId, cost);
	}

}

template<typename T_Graph, typename T_Cost>
inline void addQuadraticCost(T_Graph & graph,
							 int vertex1Id,
							 int vertex2Id,
							 int sourceVertexId,
							 int targetVertexId,
							 T_Cost costV1FgV2Fg,
							 T_Cost costV1FgV2Bg,
							 T_Cost costV1BgV2Fg,
							 T_Cost costV1BgV2Bg,
							 std::unordered_map<std::array<int, 2>, int> & edgeIndex) {

	T_Cost& w_00 = costV1BgV2Bg;
	T_Cost& w_10 = costV1FgV2Bg;
	T_Cost& w_01 = costV1BgV2Fg;
	T_Cost& w_11 = costV1FgV2Fg;

	assert(w_10 + w_01 >= w_11 + w_00); //important !

	if (w_10 + w_01 == w_11 + w_00) {
		return;
	}

	//quadratic part
	if (edgeIndex.count(std::array<int, 2>({vertex1Id, vertex2Id})) <= 0) {
		int edgeId = graph.linkVertices(vertex1Id, vertex2Id, w_10 + w_01 - w_11 - w_00);
		edgeIndex[std::array<int, 2>({vertex1Id, vertex2Id})] = edgeId;
	} else {
		int edgeId = edgeIndex[std::array<int, 2>({vertex1Id, vertex2Id})];
		graph.increaseEdgeWeight(edgeId, w_10 + w_01 - w_11 - w_00);
	}

	//linear part vertex1

	int vId1;
	int vId2;

	T_Cost cost;

	if (w_10 >= w_00) {

		vId1 = sourceVertexId;
		vId2 = vertex1Id;

		cost = w_10 - w_00;

	} else if (w_00 > w_10) {

		vId1 = vertex1Id;
		vId2 = targetVertexId;

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

		vId1 = sourceVertexId;
		vId2 = vertex2Id;

		cost = w_11 - w_10;

	} else if (w_10 > w_11) {

		vId1 = vertex2Id;
		vId2 = targetVertexId;

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

}

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

			Internal::addLinearCost(graph, vertexId, sourceId, targetId, w_1, w_0, edgeIndex);

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

		Internal::addQuadraticCost(graph,
								   vertexId1,
								   vertexId2,
								   sourceId,
								   targetId,
								   w_11,
								   w_10,
								   w_01,
								   w_00,
								   edgeIndex);

	}

	return graph;

}

template<typename T_Cost>
OptimizableIndexedGraph<T_Cost> buildMaskedGraph(Multidim::Array<T_Cost, 3> const& cost,
												 MaskCostPolicy<T_Cost> const& global_cost_policy,
												 Multidim::Array<bool, 2> const& optimizablePixels,
												 Multidim::Array<FgBgSegmentation::MaskInfo, 2> const& currentValues) {

	auto shape = cost.shape();
	int nOptimizablePixels = 0;

	if (optimizablePixels.shape()[0] != shape[0] or optimizablePixels.shape()[1] != shape[1]) {
		return OptimizableIndexedGraph<T_Cost>();
	}

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			if (optimizablePixels.valueUnchecked(i,j) == true) {
				nOptimizablePixels++;
			}
		}
	}

	int nVertices = nOptimizablePixels+2;

	FgBgSegmentation::OptimizableIndexedGraph<T_Cost> graph(nVertices);

	int pxCount = 0;
	std::unordered_map<std::array<int, 2>, int> pixelsIndex;
	pixelsIndex.reserve(nOptimizablePixels);

	//indicate which vertex correspond to which pixels.
	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			if (optimizablePixels.valueUnchecked(i,j) == true) {
				graph.template setVertexData<std::array<int,2>>(pxCount, std::array<int,2>{i,j});
				pixelsIndex[{i,j}] = pxCount;
				pxCount++;
			}
		}
	}

	int sourceId = nOptimizablePixels;
	int targetId = nOptimizablePixels+1;

	std::unordered_map<std::array<int, 2>, int> edgeIndex;
	edgeIndex.reserve(6*nVertices);

	//build base cost edges

	Multidim::IndexConverter<2> idxConverter({shape[0], shape[1]});

	for (int vId = 0; vId < nOptimizablePixels; vId++) {

		std::array<int, 2> pixCoord = graph.vertexData(vId);

		T_Cost w_0 = cost.valueUnchecked(pixCoord[0],pixCoord[1],Background);
		T_Cost w_1 = cost.valueUnchecked(pixCoord[0],pixCoord[1],Foreground);

		Internal::addLinearCost(graph, vId, sourceId, targetId, w_1, w_0, edgeIndex);

	}


	//build global cost edges

	auto pairs = global_cost_policy.coordinatePairs4Consideration({shape[0], shape[1]});

	for (std::pair<std::array<int, 2>, std::array<int, 2>> const & pair : pairs) {

		std::array<int, 2> pos1 = pair.first;
		std::array<int, 2> pos2 = pair.second;

		if (pixelsIndex.count(pos1) <= 0 and pixelsIndex.count(pos2) <= 0) {
			continue; //none of the pixels are optimizable

		} else if (pixelsIndex.count(pos1) > 0 and pixelsIndex.count(pos2) <= 0) {

			//linear function for a single pixel (vertex1)
			int vId = pixelsIndex[pos1];

			FgBgSegmentation::MaskInfo pos2Val = currentValues.valueUnchecked(pos2);

			T_Cost w_0 = global_cost_policy.globalCost(pos1, Background, pos2, pos2Val);
			T_Cost w_1 = global_cost_policy.globalCost(pos1, Foreground, pos2, pos2Val);

			Internal::addLinearCost(graph, vId, sourceId, targetId, w_1, w_0, edgeIndex);

		} else if (pixelsIndex.count(pos2) > 0 and pixelsIndex.count(pos1) <= 0) {

			//linear function for a single pixel (vertex2)
			int vId = pixelsIndex[pos2];

			FgBgSegmentation::MaskInfo pos1Val = currentValues.valueUnchecked(pos1);

			T_Cost w_0 = global_cost_policy.globalCost(pos1, pos1Val, pos2, Background);
			T_Cost w_1 = global_cost_policy.globalCost(pos1, pos1Val, pos2, Foreground);

			Internal::addLinearCost(graph, vId, sourceId, targetId, w_1, w_0, edgeIndex);

		} else if (pixelsIndex.count(pos1) > 0 and pixelsIndex.count(pos2) > 0) {

			//quadratic term

			int vertexId1 = pixelsIndex[pos1];
			int vertexId2 = pixelsIndex[pos2];

			if (vertexId1 > vertexId2) {
				std::swap(vertexId1, vertexId2);
				std::swap(pos1, pos2);
			}

			T_Cost w_00 = global_cost_policy.globalCost(pos1, Background, pos2, Background);
			T_Cost w_10 = global_cost_policy.globalCost(pos1, Foreground, pos2, Background);
			T_Cost w_01 = global_cost_policy.globalCost(pos1, Background, pos2, Foreground);
			T_Cost w_11 = global_cost_policy.globalCost(pos1, Foreground, pos2, Foreground);

			Internal::addQuadraticCost(graph,
									   vertexId1,
									   vertexId2,
									   sourceId,
									   targetId,
									   w_11,
									   w_10,
									   w_01,
									   w_00,
									   edgeIndex);

		}

	}

	return graph;

}


struct UpscaledMaskInfos {
	Multidim::Array<FgBgSegmentation::MaskInfo, 2> upscaled_mask;
	Multidim::Array<bool, 2> valueNeedsCheck;
};

inline UpscaledMaskInfos upscaleMask(Multidim::Array<FgBgSegmentation::MaskInfo, 2> const& smallMask, std::array<int, 2> newShape) {

	std::array<int,2> oldShape = smallMask.shape();

	if (newShape[0] < oldShape[0] and newShape[1] < oldShape[1]) {
		return UpscaledMaskInfos{Multidim::Array<FgBgSegmentation::MaskInfo, 2>(), Multidim::Array<bool, 2>()};
	}

	UpscaledMaskInfos ret{Multidim::Array<FgBgSegmentation::MaskInfo, 2>(newShape), Multidim::Array<bool, 2>(newShape)};

	for(int i = 0; i < newShape[0]; i++) {
		for(int j = 0; j < newShape[1]; j++) {

			int si = (i*oldShape[0])/newShape[0];
			int sj = (j*oldShape[1])/newShape[1];

			int countFg = 0;
			int countBg = 0;

			FgBgSegmentation::MaskInfo base = smallMask.valueUnchecked(si,sj);

			for (int di = 0; di <= 1; di++) {
				for (int dj = 0; dj <= 1; dj++) {
					FgBgSegmentation::MaskInfo t = smallMask.valueOrAlt({si+di,sj+dj}, base);

					if (t == Foreground) {
						countFg++;
					} else if (t == Background) {
						countBg++;
					}
				}
			}

			if (countBg == 0) { // all Foreground
				ret.upscaled_mask.atUnchecked(i,j) = Foreground;
				ret.valueNeedsCheck.atUnchecked(i,j) = false;

			} else if (countFg == 0) { // all Background
				ret.upscaled_mask.atUnchecked(i,j) = Background;
				ret.valueNeedsCheck.atUnchecked(i,j) = false;

			} else {
				ret.upscaled_mask.atUnchecked(i,j) = (countFg > countBg) ? Foreground : Background;
				ret.valueNeedsCheck.atUnchecked(i,j) = true;
			}

		}
	}

	return ret;

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

template<typename T_Cost>
Multidim::Array<FgBgSegmentation::MaskInfo, 2> getPartialGlobalRefinedMask(Multidim::Array<T_Cost, 3> const& cost,
																		   MaskCostPolicy<T_Cost> const& global_cost_policy,
																		   Multidim::Array<bool, 2> const& optimizablePixels,
																		   Multidim::Array<FgBgSegmentation::MaskInfo, 2> const& currentValues) {

	FgBgSegmentation::OptimizableIndexedGraph<T_Cost> graph = FgBgSegmentation::buildMaskedGraph(cost, global_cost_policy, optimizablePixels, currentValues);

	int nVertices = graph.nVertices();

	std::array<int, 2> shape = {cost.shape()[0], cost.shape()[1]};

	int sourceVertexId = nVertices-2;
	int targetVertexId = nVertices-1;

	auto maxFlowMinCut = GraphProcessing::maxFlowMinCut(graph,
														sourceVertexId,
														targetVertexId);

	std::vector<int> reachableVertices = GraphProcessing::reachableVerticesInCut(graph, maxFlowMinCut.minCutEdgesIdxs, targetVertexId);

	Multidim::Array<FgBgSegmentation::MaskInfo, 2> ret(shape);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			if (optimizablePixels.valueUnchecked(i,j)) {
				ret.atUnchecked(i,j) = FgBgSegmentation::Background;
			} else {
				ret.atUnchecked(i,j) = currentValues.valueUnchecked(i,j);
			}
		}
	}



	for (int vId : reachableVertices) {
		std::array<int,2> imgPos = graph.vertexData(vId);
		ret.atUnchecked(imgPos[0],imgPos[1]) = FgBgSegmentation::Foreground;
	}

	return ret;

}

template<typename T_Cost, std::size_t depth>
Multidim::Array<FgBgSegmentation::MaskInfo, 2> hierarchicalGlobalRefinedMask(std::array<Multidim::Array<T_Cost, 3> const*, depth> costs,
																			 std::array<MaskCostPolicy<T_Cost> const*, depth> cost_policies) {

	std::array<std::array<int,2>,depth> shapes;

	for (int d = 0; d < depth; d++) {

		shapes[d] = {costs[d]->shape()[0], costs[d]->shape()[1]};
	}

	//check the sizes are indeed decreasing with depth
	for (int d = 0; d < depth-1; d++) {
		if (shapes[d][0] < shapes[d+1][0] or shapes[d][1] < shapes[d+1][1]) {
			return Multidim::Array<FgBgSegmentation::MaskInfo, 2>();
		}
	}

	int currentDepth = depth-1;

	FgBgSegmentation::UpscaledMaskInfos maskInfos;

	maskInfos.upscaled_mask = getGlobalRefinedMask(*costs[currentDepth], *cost_policies[currentDepth]);

	for (int p = 1; p < depth; p++) {
		currentDepth = depth-p-1;
		maskInfos = FgBgSegmentation::upscaleMask(maskInfos.upscaled_mask, shapes[currentDepth]);

		FgBgSegmentation::OptimizableIndexedGraph<T_Cost> graph = FgBgSegmentation::buildMaskedGraph(*costs[currentDepth],
																									 *cost_policies[currentDepth],
																									 maskInfos.valueNeedsCheck,
																									 maskInfos.upscaled_mask);


		int nVertices = graph.nVertices();

		int sourceVertexId = nVertices-2;
		int targetVertexId = nVertices-1;

		auto maxFlowMinCut = GraphProcessing::maxFlowMinCut(graph,
															sourceVertexId,
															targetVertexId);

		std::vector<int> reachableVertices = GraphProcessing::reachableVerticesInCut(graph, maxFlowMinCut.minCutEdgesIdxs, targetVertexId);

		for (int vId = 0; vId < nVertices-2; vId++) {
			std::array<int,2> imgPos = graph.vertexData(vId);
			maskInfos.upscaled_mask.atUnchecked(imgPos[0],imgPos[1]) = FgBgSegmentation::Background;
		}

		for (int vId : reachableVertices) {
			std::array<int,2> imgPos = graph.vertexData(vId);
			maskInfos.upscaled_mask.atUnchecked(imgPos[0],imgPos[1]) = FgBgSegmentation::Foreground;
		}
	}

	return maskInfos.upscaled_mask;

}

template<typename T_Cost>
Multidim::Array<FgBgSegmentation::MaskInfo, 2> hierarchicalGlobalRefinedMask(std::vector<Multidim::Array<T_Cost, 3> const*> const& costs,
																			 std::vector<MaskCostPolicy<T_Cost> const*> const& cost_policies) {

	if (costs.size() != cost_policies.size()) {
		return Multidim::Array<FgBgSegmentation::MaskInfo, 2>();
	}

	int depth = costs.size();
	std::vector<std::array<int,2>> shapes(depth);

	for (int d = 0; d < depth; d++) {

		shapes[d] = {costs[d]->shape()[0], costs[d]->shape()[1]};
	}

	//check the sizes are indeed decreasing with depth
	for (int d = 0; d < depth-1; d++) {
		if (shapes[d][0] < shapes[d+1][0] or shapes[d][1] < shapes[d+1][1]) {
			return Multidim::Array<FgBgSegmentation::MaskInfo, 2>();
		}
	}

	int currentDepth = depth-1;

	FgBgSegmentation::UpscaledMaskInfos maskInfos;

	maskInfos.upscaled_mask = getGlobalRefinedMask(*costs[currentDepth], *cost_policies[currentDepth]);

	for (int p = 1; p < depth; p++) {
		currentDepth = depth-p-1;
		maskInfos = FgBgSegmentation::upscaleMask(maskInfos.upscaled_mask, shapes[currentDepth]);

		FgBgSegmentation::OptimizableIndexedGraph<T_Cost> graph = FgBgSegmentation::buildMaskedGraph(*costs[currentDepth],
																									 *cost_policies[currentDepth],
																									 maskInfos.valueNeedsCheck,
																									 maskInfos.upscaled_mask);


		int nVertices = graph.nVertices();

		int sourceVertexId = nVertices-2;
		int targetVertexId = nVertices-1;

		auto maxFlowMinCut = GraphProcessing::maxFlowMinCut(graph,
															sourceVertexId,
															targetVertexId);

		std::vector<int> reachableVertices = GraphProcessing::reachableVerticesInCut(graph, maxFlowMinCut.minCutEdgesIdxs, targetVertexId);

		for (int vId = 0; vId < nVertices-2; vId++) {
			std::array<int,2> imgPos = graph.vertexData(vId);
			maskInfos.upscaled_mask.atUnchecked(imgPos[0],imgPos[1]) = FgBgSegmentation::Background;
		}

		for (int vId : reachableVertices) {
			std::array<int,2> imgPos = graph.vertexData(vId);
			maskInfos.upscaled_mask.atUnchecked(imgPos[0],imgPos[1]) = FgBgSegmentation::Foreground;
		}
	}

	return maskInfos.upscaled_mask;

}

} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_FOREGROUNDSEGMENTATION_H
