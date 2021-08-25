#ifndef STEREOVISION_CORRELATION_SGM_H
#define STEREOVISION_CORRELATION_SGM_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021  Paragon<french.paragon@gmail.com>

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

#include "../utils/margins.h"

#include "./correlation_base.h"

namespace StereoVision {
namespace Correlation {

enum class sgmDirections{
	Up2Down,
	Down2Up,
	Left2Right,
	Right2Left,
	UpLeft2DownRight,
	DownRight2UpLeft,
	UpRight2DownLeft,
	DownLeft2UpRight,
	UpLeft2Right,
	DownRight2Left,
	UpRight2Left,
	DownLeft2Right,
	UpLeft2Down,
	DownRight2Up,
	UpRight2Down,
	DownLeft2Up
};

namespace Internal {

enum class sgmStartsPos{
	NoStart,
	ZeroPos,
	EndPos,
	Invalid
};

template<sgmDirections direction>
struct directionTraits {
};

template<>
struct directionTraits<sgmDirections::Up2Down> {
	static constexpr std::array<int, 2> stepsVertical = {1,1};
	static constexpr std::array<int, 2> stepsHorizontal = {0,0};
};

template<>
struct directionTraits<sgmDirections::Down2Up> {
	static constexpr std::array<int, 2> stepsVertical = {-1,-1};
	static constexpr std::array<int, 2> stepsHorizontal = {0,0};
};

template<>
struct directionTraits<sgmDirections::Left2Right> {
	static constexpr std::array<int, 2> stepsVertical = {0,0};
	static constexpr std::array<int, 2> stepsHorizontal = {1,1};
};

template<>
struct directionTraits<sgmDirections::Right2Left> {
	static constexpr std::array<int, 2> stepsVertical = {0,0};
	static constexpr std::array<int, 2> stepsHorizontal = {-1,-1};
};

template<>
struct directionTraits<sgmDirections::UpLeft2DownRight> {
	static constexpr std::array<int, 2> stepsVertical = {1,1};
	static constexpr std::array<int, 2> stepsHorizontal = {1,1};
};

template<>
struct directionTraits<sgmDirections::DownRight2UpLeft> {
	static constexpr std::array<int, 2> stepsVertical = {-1,-1};
	static constexpr std::array<int, 2> stepsHorizontal = {-1,-1};
};

template<>
struct directionTraits<sgmDirections::UpRight2DownLeft> {
	static constexpr std::array<int, 2> stepsVertical = {1,1};
	static constexpr std::array<int, 2> stepsHorizontal = {-1,-1};
};

template<>
struct directionTraits<sgmDirections::DownLeft2UpRight> {
	static constexpr std::array<int, 2> stepsVertical = {-1,-1};
	static constexpr std::array<int, 2> stepsHorizontal = {1,1};
};

template<>
struct directionTraits<sgmDirections::UpLeft2Right> {
	static constexpr std::array<int, 2> stepsVertical = {0,1};
	static constexpr std::array<int, 2> stepsHorizontal = {1,1};
};

template<>
struct directionTraits<sgmDirections::DownRight2Left> {
	static constexpr std::array<int, 2> stepsVertical = {0,-1};
	static constexpr std::array<int, 2> stepsHorizontal = {-1,-1};
};

template<>
struct directionTraits<sgmDirections::UpRight2Left> {
	static constexpr std::array<int, 2> stepsVertical = {0,1};
	static constexpr std::array<int, 2> stepsHorizontal = {-1,-1};
};

template<>
struct directionTraits<sgmDirections::DownLeft2Right> {
	static constexpr std::array<int, 2> stepsVertical = {0,-1};
	static constexpr std::array<int, 2> stepsHorizontal = {1,1};
};

template<>
struct directionTraits<sgmDirections::UpLeft2Down> {
	static constexpr std::array<int, 2> stepsVertical = {1,1};
	static constexpr std::array<int, 2> stepsHorizontal = {0,1};
};

template<>
struct directionTraits<sgmDirections::DownRight2Up> {
	static constexpr std::array<int, 2> stepsVertical = {-1,-1};
	static constexpr std::array<int, 2> stepsHorizontal = {0,-1};
};

template<>
struct directionTraits<sgmDirections::UpRight2Down> {
	static constexpr std::array<int, 2> stepsVertical = {1,1};
	static constexpr std::array<int, 2> stepsHorizontal = {0,-1};
};

template<>
struct directionTraits<sgmDirections::DownLeft2Up> {
	static constexpr std::array<int, 2> stepsVertical = {-1,-1};
	static constexpr std::array<int, 2> stepsHorizontal = {0,1};
};

struct StartPosInfos {
	sgmStartsPos colStartPos;
	sgmStartsPos rowStartPos;
};

constexpr StartPosInfos startPostInfos(std::array<int, 2> stepsVertical, std::array<int, 2> stepsHorizontal) {

	sgmStartsPos colStartPos = sgmStartsPos::Invalid;
	sgmStartsPos rowStartPos = sgmStartsPos::Invalid;

	if (stepsVertical[0] == 0 and stepsVertical[1] == 0) {
		colStartPos = sgmStartsPos::NoStart;
	} else if (stepsVertical[0] >= 0 and stepsVertical[1] >= 0) {
		colStartPos = sgmStartsPos::ZeroPos;
	} else if (stepsVertical[0] <= 0 and stepsVertical[1] <= 0) {
		colStartPos = sgmStartsPos::EndPos;
	}

	if (stepsHorizontal[0] == 0 and stepsHorizontal[1] == 0) {
		rowStartPos = sgmStartsPos::NoStart;
	} else if (stepsHorizontal[0] >= 0 and stepsHorizontal[1] >= 0) {
		rowStartPos = sgmStartsPos::ZeroPos;
	} else if (stepsHorizontal[0] <= 0 and stepsHorizontal[1] <= 0) {
		rowStartPos = sgmStartsPos::EndPos;
	}

	return {colStartPos, rowStartPos};
}

template<sgmDirections direction, dispExtractionStartegy extractionStrategy, class T_CV>
void traverseLine(size_t start_i,
				  size_t start_j,
				  Multidim::Array<T_CV, 3> const& cv_base,
				  Multidim::Array<float, 3> & sgm_cv,
				  float P1,
				  float P2,
				  Margins const& margins,
				  float Pout = 100) {

	constexpr auto Nc = Multidim::AccessCheck::Nocheck;

	auto stepsVertical = directionTraits<direction>::stepsVertical;
	auto stepsHorizontal = directionTraits<direction>::stepsHorizontal;

	auto cv_shape = cv_base.shape();

	float* previous_cost = new float[cv_shape[2]];
	float* actual_cost = new float[cv_shape[2]];

	for(long d = 0; d < cv_shape[2]; d++) {
		previous_cost[d] = 0.0;
	}

	int c;
	int i;
	int j;

	for (c = 0, i = start_i, j = start_j;
		 (i >= margins.top() && i < cv_shape[0]-margins.bottom()) && (j >= margins.left() && j < cv_shape[1]-margins.right());
		 i += stepsVertical[c%2], j += stepsHorizontal[c%2], c++) { //traverse line

		if (extractionStrategy == dispExtractionStartegy::Score) {

			float max_p_cost = -std::numeric_limits<float>::infinity();

			for(long d = 0; d < cv_shape[2]; d++) {
				if (previous_cost[d] > max_p_cost) {
					max_p_cost = previous_cost[d];
				}
			}


			for(disp_t nd = 0; nd < cv_shape[2]; nd++) {

				float max_a_cost = -std::numeric_limits<float>::infinity();

				for(disp_t od = 0; od < cv_shape[2]; od++) {

					float c_score = static_cast<float>(cv_base.template value<Nc>(i,j,nd));
					c_score += previous_cost[od];
					if (std::abs(static_cast<int>(od) - static_cast<int>(nd)) == 1) c_score -= P1;
					if (std::abs(static_cast<int>(od) - static_cast<int>(nd)) > 1) c_score -= P2;

					if (c_score > max_a_cost) {
						max_a_cost = c_score;
					}

				}

				if (j + nd >= cv_shape[1]) {
					max_a_cost -= Pout;
				}

				actual_cost[nd] = max_a_cost - max_p_cost;
			}

		} else {

			float min_p_cost = std::numeric_limits<float>::infinity();

			for(long d = 0; d < cv_shape[2]; d++) {
				if (previous_cost[d] < min_p_cost) {
					min_p_cost = previous_cost[d];
				}
			}


			for(disp_t nd = 0; nd < cv_shape[2]; nd++) {

				float min_a_cost = std::numeric_limits<float>::infinity();

				for(disp_t od = 0; od < cv_shape[2]; od++) {

					float c_score = static_cast<float>(cv_base.template value<Nc>(i,j,nd));
					c_score += previous_cost[od];
					if (std::abs(static_cast<int>(od) - static_cast<int>(nd)) == 1) c_score += P1;
					if (std::abs(static_cast<int>(od) - static_cast<int>(nd)) > 1) c_score += P2;

					if (c_score < min_a_cost) {
						min_a_cost = c_score;
					}

				}

				if (j + nd >= cv_shape[1]) {
					min_a_cost += Pout;
				}

				actual_cost[nd] = min_a_cost - min_p_cost;
			}
		}

		for(disp_t d = 0; d < cv_shape[2]; d++) {
			sgm_cv.at<Nc>(i,j,d) += actual_cost[d] - cv_base.template value<Nc>(i,j,d);
		}

		float* tmp = previous_cost;
		previous_cost = actual_cost;
		actual_cost = tmp;

	}

	delete [] previous_cost;
	delete [] actual_cost;

}

template<sgmDirections direction, dispExtractionStartegy extractionStrategy, class T_CV>
void addDirectionalCost(Multidim::Array<T_CV, 3> const& cv_base,
						Multidim::Array<float, 3> & sgm_cv,
						float P1,
						float P2,
						Margins const& margins,
						float Pout = 100) {

	constexpr StartPosInfos sPosInf = startPostInfos(directionTraits<direction>::stepsVertical,
													 directionTraits<direction>::stepsHorizontal);

	static_assert (sPosInf.colStartPos != sgmStartsPos::Invalid, "Invalid column start position");
	static_assert (sPosInf.rowStartPos != sgmStartsPos::Invalid, "Invalid row start position");

	auto cv_shape = cv_base.shape();

	if (sPosInf.rowStartPos != sgmStartsPos::NoStart) { //Needs to iterate over vertical starts points

		size_t start_j = (sPosInf.rowStartPos == sgmStartsPos::ZeroPos) ? margins.left() : cv_shape[1] - margins.right();

		size_t bi = margins.top();
		size_t ei = cv_shape[0]-margins.bottom();

		#pragma omp parallel for
		for(size_t start_i = bi; start_i < ei; start_i++) {
			traverseLine<direction, extractionStrategy>(start_i, start_j, cv_base, sgm_cv, P1, P2, margins, Pout);
		}

	}

	if (sPosInf.colStartPos != sgmStartsPos::NoStart) { //Needs to iterate over horizontal starts points

		size_t start_i = (sPosInf.colStartPos == sgmStartsPos::ZeroPos) ? margins.top() : cv_shape[0]-margins.bottom();

		size_t bj = margins.left();
		size_t ej = cv_shape[1]-margins.right();

		#pragma omp parallel for
		for(size_t start_j = bj; start_j < ej; start_j++) {
			traverseLine<direction, extractionStrategy>(start_i, start_j, cv_base, sgm_cv, P1, P2, margins, Pout);
		}
	}

}

}// namespace Internal

template<int nDirections, dispExtractionStartegy extractionStrategy, class T_CV>
Multidim::Array<float, 3> sgmCostVolume(Multidim::Array<T_CV, 3> const& cv_base,
										float P1,
										float P2,
										Margins const& margins,
										float Pout = 100) {

	static_assert (nDirections == 4 or nDirections == 8 or nDirections == 16, "SGM can only operate with 4, 8 or 16 directions");

	Multidim::Array<float, 3> sgm_cv(cv_base.shape());

	for (int i = 0; i < cv_base.shape()[0]; i++) {
		for (int j = 0; j < cv_base.shape()[1]; j++) {
			for (int d = 0; d < cv_base.shape()[2]; d++) {
				sgm_cv.atUnchecked(i, j, d) = cv_base.valueUnchecked(i, j, d);
			}
		}
	}

	Internal::addDirectionalCost<sgmDirections::Up2Down, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
	Internal::addDirectionalCost<sgmDirections::Down2Up, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
	Internal::addDirectionalCost<sgmDirections::Left2Right, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
	Internal::addDirectionalCost<sgmDirections::Right2Left, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);

	if (nDirections >= 8) {
		Internal::addDirectionalCost<sgmDirections::UpLeft2DownRight, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::DownRight2UpLeft, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::UpRight2DownLeft, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::DownLeft2UpRight, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
	}

	if (nDirections >= 16) {
		Internal::addDirectionalCost<sgmDirections::UpLeft2Down, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::DownRight2Up, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::UpRight2Down, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::DownLeft2Up, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);

		Internal::addDirectionalCost<sgmDirections::UpLeft2Right, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::DownRight2Left, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::UpRight2Left, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
		Internal::addDirectionalCost<sgmDirections::DownLeft2Right, extractionStrategy>(cv_base, sgm_cv, P1, P2, margins, Pout);
	}

	for (int i = 0; i < cv_base.shape()[0]; i++) {
		for (int j = 0; j < cv_base.shape()[1]; j++) {
			for (int d = 0; d < cv_base.shape()[2]; d++) {
				sgm_cv.atUnchecked(i, j, d) += cv_base.valueUnchecked(i, j, d);
			}
		}
	}

	return sgm_cv;
}

} // namespace Correlation
} // namespace StereoVision

#endif // SGM_H
