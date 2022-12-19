#ifndef STEREOVISION_DYNAMIC_PROGRAMING_STEREO_H
#define STEREOVISION_DYNAMIC_PROGRAMING_STEREO_H

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

/*
 * see https://ieeexplore.ieee.org/abstract/document/4767639
 */

#include "./correlation_base.h"
#include "./matching_costs.h"

#include "../geometry/core.h"
#include "../geometry/imagecoordinates.h"

#include <functional>

namespace StereoVision {
namespace Correlation {

class DynamicProgramming {

public:

	using Index2d = std::array<int,2>;

	enum JumpType {
		SkippedTarget,
		SkippedSource,
		NoSkip
	};

	template<typename T_Cost>
	/*!
	 * a JumpCostPolicy is a function to determine the additional cost of a certain jump in a dynamic programming matching model
	 *
	 * The function receive as argument the coordinate of the pixel in the source image (might be negative, to account for one virtual line before the image),
	 * as well as the current jump type, the previous jump type and the number of iteration that jump type has gone on.
	 */
	using JumpCostPolicy = std::function<T_Cost(Index2d, JumpType, JumpType, int)>;

private:

	template<typename T, dispExtractionStartegy strategy>
	static bool optionalCompare(T v1, std::optional<T> v2) {
		if (!v2.has_value()) {
			return true;
		}

		if (strategy == dispExtractionStartegy::Cost) {
			return v1 < v2.value();
		}

		return false;
	}

public:

	template<dispExtractionStartegy strategy, class T_CV>
	class SGMLikeJumpCostPolicy {
	public:

		SGMLikeJumpCostPolicy(T_CV costJumpBase, T_CV costNextJumps) :
			_first_jump_cost((strategy == dispExtractionStartegy::Cost) ? costJumpBase : -costJumpBase),
			_next_jumps_cost((strategy == dispExtractionStartegy::Cost) ? costNextJumps : -costNextJumps) {

		}

		T_CV operator()(Index2d pos, JumpType currentJump, JumpType previousJump, int previousCount) {

			(void) pos;

			if (currentJump == NoSkip) {
				return 0;
			}

			if (previousCount > 2) {
				return 0;
			}

			return (currentJump == previousJump) ? _next_jumps_cost : _first_jump_cost;
		}
	private:
		T_CV _first_jump_cost;
		T_CV _next_jumps_cost;
	};

	template<dispExtractionStartegy strategy, class T_G, class T_CV, class T_T = float>
	class SGMLikeWithImageGuideJumpCostPolicy {
	public:

		SGMLikeWithImageGuideJumpCostPolicy(T_CV costJumpBase, T_CV costNextJumps) :
			_first_jump_cost((strategy == dispExtractionStartegy::Cost) ? costJumpBase : -costJumpBase),
			_next_jumps_cost((strategy == dispExtractionStartegy::Cost) ? costNextJumps : -costNextJumps) {

		}

		T_CV operator()(Index2d pos, JumpType currentJump, JumpType previousJump, int previousCount) {

			(void) pos;
			(void) previousCount;

			if (currentJump == NoSkip) {
				return 0;
			}
			return (currentJump == previousJump) ? _next_jumps_cost : _first_jump_cost;
		}
	private:

		Geometry::AffineTransform<T_T>& _refToGuide;
		Multidim::Array<T_G,3>& _guide;

		T_CV _first_jump_cost;
		T_CV _next_jumps_cost;
	};

	template<dispExtractionStartegy strategy, class T_CV>
	static Multidim::Array<disp_t, 2> extractOptimalIndex(Multidim::Array<T_CV, 3> const& costVolume,
														  JumpCostPolicy<T_CV> const& jumpCostPolicy,
														  disp_t invalid_disp = -1) {

		constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

		int searchWidth = costVolume.shape()[2];
		int imWidth = costVolume.shape()[1];
		int imHeight = costVolume.shape()[0];

		if (searchWidth > imWidth) {
			searchWidth = imWidth;
		}

		Multidim::Array<disp_t, 2> disp(imHeight, imWidth);

		Multidim::Array<T_CV, 2> costGrid(searchWidth, imWidth+1);
		Multidim::Array<JumpType, 2> jumpTypeGrid(searchWidth, imWidth+1);
		Multidim::Array<int, 2> jumpTypeCountGrid(searchWidth, imWidth+1);
		Multidim::Array<Index2d, 2> path(searchWidth, imWidth);

		#pragma omp parallel for
		for (int i = 0; i < imHeight; i++) { //for each scanline

			for (int d = 0; d < searchWidth; d++) {
				costGrid.template at<Nc>(d,0) = jumpCostPolicy({-1,-1}, SkippedSource, (d == 0) ? NoSkip : SkippedSource, std::max(0,d-1));
				jumpTypeGrid.template at<Nc>(d,0) = SkippedTarget;
				jumpTypeCountGrid.template at<Nc>(d,0) = d;
			}

			//fill in the grid
			for (int j = 0; j < imWidth; j++) {

				for (int d = 0; d < searchWidth and d+j < imWidth; d++) {

					T_CV noJumpCost = costGrid.template value<Nc>(d,j); //cost of substitution in levenshtein distance
					noJumpCost += costVolume.template value<Nc>(i,j,d); //do a match

					std::optional<T_CV> skipTargetCost; //cost of insertion in levenshtein distance
					if (d > 0) {
						skipTargetCost = costGrid.template value<Nc>(d-1,j+1);
						skipTargetCost.value() += jumpCostPolicy({i,j}, SkippedSource, jumpTypeGrid.value<Nc>(d-1,j), jumpTypeCountGrid.value<Nc>(d-1,j));
					}


					std::optional<T_CV> skipSourceCost;
					if (d < searchWidth-1) { //cost of deletion in levenshtein distance
						skipSourceCost = costGrid.template value<Nc>(d+1,j);
						skipSourceCost.value() += jumpCostPolicy({i,j}, SkippedTarget, jumpTypeGrid.value<Nc>(d+1,j-1), jumpTypeCountGrid.value<Nc>(d+1,j-1));
					}

					//select optimal subpath

					if (optionalCompare<T_CV,strategy>(noJumpCost, skipTargetCost) and
							optionalCompare<T_CV,strategy>(noJumpCost, skipSourceCost)) {

						costGrid.template at<Nc>(d,j+1) = noJumpCost;
						jumpTypeGrid.at<Nc>(d,j+1) = NoSkip;
						jumpTypeCountGrid.at<Nc>(d,j+1) = (costGrid.template value<Nc>(d,j) == NoSkip) ? jumpTypeCountGrid.at<Nc>(d,j)+1 : 0 ;
						path.at<Nc>(d,j) = {d,j};

						continue;

					}

					if (skipTargetCost.has_value()) {
						if (optionalCompare<T_CV,strategy>(skipTargetCost.value(), skipSourceCost)) {

							costGrid.template at<Nc>(d,j+1) = skipTargetCost.value();
							jumpTypeGrid.at<Nc>(d,j+1) = SkippedSource;
							jumpTypeCountGrid.at<Nc>(d,j+1) = (costGrid.template value<Nc>(d-1,j+1) == SkippedSource) ? jumpTypeCountGrid.at<Nc>(d-1,j+1)+1 : 0 ;
							path.at<Nc>(d,j) = {d-1,j+1};

							continue;
						}
					}

					assert(skipSourceCost.has_value());

					costGrid.template at<Nc>(d,j+1) = skipSourceCost.value();
					jumpTypeGrid.at<Nc>(d,j+1) = SkippedTarget;
					jumpTypeCountGrid.at<Nc>(d,j+1) = (costGrid.template value<Nc>(d+1,j) == SkippedTarget) ? jumpTypeCountGrid.at<Nc>(d+1,j)+1 : 0 ;
					path.at<Nc>(d,j) = {d+1,j};

				}
			}

			//walk the path backwards.
			Index2d pathPos = {0,imWidth};

			while (pathPos[1] > 0) {
				Index2d previous_pos = path.at<Nc>(pathPos[0], pathPos[1]-1);

				if (previous_pos == Index2d({pathPos[0], pathPos[1]-1})) {//no jumps, constant disparity
					disp.at<Nc>(i,pathPos[1]-1) = pathPos[0];
					pathPos = previous_pos;
					continue;
				}

				if (previous_pos == Index2d({pathPos[0]-1, pathPos[1]})) {//skip target, no effect on disparity
					pathPos = previous_pos;
					continue;
				}

				if (previous_pos == Index2d({pathPos[0]+1, pathPos[1]-1})) {//skip source, set invalid disparity
					disp.at<Nc>(i,pathPos[1]-1) = invalid_disp;
					pathPos = previous_pos;
					continue;
				}
			}

		}

		return disp;

	}

};

} //namespace Correlation
} //namespace StereoVision

#endif // STEREOVISION_DYNAMIC_PROGRAMING_STEREO_H
