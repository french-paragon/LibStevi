#ifndef LIBSTEVI_CHECKBOARDDETECTION_H
#define LIBSTEVI_CHECKBOARDDETECTION_H

#include "../utils/types_manipulations.h"
#include "../utils/array_utils.h"
#include "../geometry/imagecoordinates.h"
#include "../correlation/unfold.h"

#include "./finiteDifferences.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "../optimization/l2optimization.h"

#include <MultidimArrays/MultidimArrays.h>

#include <vector>
#include <map>
#include <optional>
#include <set>

namespace StereoVision {

struct discretCheckCornerInfos {

	discretCheckCornerInfos()
	{

	}

	discretCheckCornerInfos(int pixX,
								int pixY,
								float lmbMin,
								float lmbMax,
								float mainDir) :
		pix_coord_x(pixX),
		pix_coord_y(pixY),
		lambda_min(lmbMin),
		lambda_max(lmbMax),
		main_dir(mainDir)
	{

	}

	int pix_coord_x;
	int pix_coord_y;
	float lambda_min;
	float lambda_max;
	float main_dir;
};

struct refinedCornerInfos {

	refinedCornerInfos()
	{

	}

	refinedCornerInfos(int gridX,
					   int gridY,
					   float pixX,
					   float pixY):
		grid_coord_x(gridX),
		grid_coord_y(gridY),
		pix_coord_x(pixX),
		pix_coord_y(pixY)
	{

	}


	int grid_coord_x;
	int grid_coord_y;
	float pix_coord_x;
	float pix_coord_y;


};

class CheckBoardPoints {

public:
	inline CheckBoardPoints() :
		_pointMaps(),
		_transpose(false),
		_rowDirection(1),
		_colDirection(1),
		_rowDelta(0),
		_colDelta(0),
		_nRows(0),
		_nCols(0)
	{
	}


	inline CheckBoardPoints(discretCheckCornerInfos initial) :
		_pointMaps(),
		_transpose(false),
		_rowDirection(1),
		_colDirection(1),
		_rowDelta(0),
		_colDelta(0),
		_nRows(1),
		_nCols(1)
	{
		_pointMaps[{0,0}] = initial;
	}

	inline int rows() const {
		if (_transpose) {
			return _nCols;
		}
		return _nRows;
	}
	inline int cols() const {
		if (_transpose) {
			return _nRows;
		}
		return _nCols;
	}

	inline int nPointsFound() const {
		return _pointMaps.size();
	}

	inline bool hasPointInCoord(int row, int col) const {

		int trow = row;
		int tcol = col;

		if (_transpose) {
			trow = col;
			tcol = row;
		}

		int innerRow = _rowDirection*trow + _rowDelta;
		int innerCol = _colDirection*tcol + _colDelta;

		return _pointMaps.count({innerRow, innerCol}) > 0;
	}

	inline std::optional<discretCheckCornerInfos> pointInCoord(int row, int col) const {

		int trow = row;
		int tcol = col;

		if (_transpose) {
			trow = col;
			tcol = row;
		}

		int innerRow = _rowDirection*trow + _rowDelta;
		int innerCol = _colDirection*tcol + _colDelta;

		if (_pointMaps.count({innerRow, innerCol}) > 0) {
			return _pointMaps.at({innerRow, innerCol});
		}
		return std::nullopt;
	}

	struct Coord {
		int x;
		int y;
	};

protected:

	struct CoordPair {
		int row;
		int col;
	};

	struct coordPairCompare {

		bool operator() (CoordPair p1, CoordPair p2) const{
			if (p1.row == p2.row) {
				return p1.col < p2.col;
			}
			return p1.row < p2.row;
		}
	};

	std::map<CoordPair, discretCheckCornerInfos, coordPairCompare> _pointMaps;

	bool _transpose;

	int _rowDirection;
	int _colDirection;

	int _rowDelta;
	int _colDelta;

	int _nRows;
	int _nCols;

	inline static std::optional<discretCheckCornerInfos> findPoint(std::map<Coord, discretCheckCornerInfos> & candidatesMap,
															int search_center_pix_x,
															int search_center_pix_y,
															int toleranceRadius,
															std::optional<float> previous_detected_direction,
															std::set<Coord> & selectedCandidates,
															float angleTolerance) {

		std::optional<discretCheckCornerInfos> ret = std::nullopt;

		int dist_ret = 4*toleranceRadius*toleranceRadius;

		for (int dx = -toleranceRadius; dx <= toleranceRadius; dx++) {
			for (int dy = -toleranceRadius; dy <= toleranceRadius; dy++) {

				if (candidatesMap.count({search_center_pix_x+dx, search_center_pix_y+dy}) <= 0) {
					continue;
				}

				discretCheckCornerInfos& subcandidate = candidatesMap[{search_center_pix_x+dx, search_center_pix_y+dy}];

				if (selectedCandidates.count({subcandidate.pix_coord_x, subcandidate.pix_coord_y}) > 0) {
					continue; //do not include the same point twice
				}

				if (previous_detected_direction.has_value()) {
					if (std::fabs(std::fabs(previous_detected_direction.value() - subcandidate.main_dir) - M_PI_2) > angleTolerance) {
						continue;
					}
				}


				int dist_cand = dx*dx + dy*dy;

				if (ret.has_value()) {
					if (dist_cand > dist_ret) {
						continue;
					}
				}

				ret = subcandidate;
				dist_ret = dist_cand;
			}
		}

		return ret;
	}

	friend CheckBoardPoints isolateCheckBoard(std::vector<discretCheckCornerInfos> const& candidates,
											  float relDistanceTolerance,
											  float angleTolerance,
											  float observationWeight,
											  int maxDistanceRadius);
};

bool operator<(CheckBoardPoints::Coord c1, CheckBoardPoints::Coord c2) {
	if (c1.x == c2.x) {
		return c1.y < c2.y;
	}
	return  c1.x < c2.x;
}

CheckBoardPoints isolateCheckBoard(std::vector<discretCheckCornerInfos> const& candidates,
								   float relDistanceTolerance = 0.05,
								   float angleTolerance = 0.05,
								   float observationWeight = 1.,
								   int maxDistanceRadius = 20) {

					 using StateVector = Eigen::Matrix<float,4,1>;
					 using ObservationVector = Eigen::Matrix<float,2,1>;

					 using Matrix4x4 = Eigen::Matrix<float,4,4>;
					 using Matrix2x2 = Eigen::Matrix<float,2,2>;
					 using Matrix2x4 = Eigen::Matrix<float,2,4>;
					 using Matrix4x2 = Eigen::Matrix<float,4,2>;

					 CheckBoardPoints currentBests;

					 using Coord = CheckBoardPoints::Coord;

					 Coord minCoord = {candidates[0].pix_coord_x, candidates[1].pix_coord_y};
					 Coord maxCoord = minCoord;
					 std::map<Coord, discretCheckCornerInfos> candidatesMap;

					 for (discretCheckCornerInfos const& candidate : candidates) {
						 candidatesMap[{candidate.pix_coord_x, candidate.pix_coord_y}] = candidate;

						 if (candidate.pix_coord_x < minCoord.x) {
							 minCoord.x = candidate.pix_coord_x;
						 }

						 if (candidate.pix_coord_y < minCoord.y) {
							 minCoord.y = candidate.pix_coord_y;
						 }

						 if (candidate.pix_coord_x > maxCoord.x) {
							 maxCoord.x = candidate.pix_coord_x;
						 }

						 if (candidate.pix_coord_y > maxCoord.y) {
							 maxCoord.y = candidate.pix_coord_y;
						 }
					 }

					 Eigen::Vector2f contRectDiag(maxCoord.x - minCoord.x, maxCoord.y - minCoord.y);
					 float containingRectDiag = contRectDiag.norm();

					 Matrix4x4 TransitionModel = Matrix4x4::Identity();
					 TransitionModel(0,2) = 1;
					 TransitionModel(1,3) = 1;

					 Matrix2x4 ObservationModel = Matrix2x4::Zero();
					 ObservationModel(0,0) = 1;
					 ObservationModel(1,1) = 1;

					 Matrix4x4 MotionModelCov = Matrix4x4::Identity();
					 Matrix2x2 ObservationModelCov = Matrix2x2::Identity()/observationWeight;

					 for (discretCheckCornerInfos const& basePoint : candidates) {

						 for (discretCheckCornerInfos const& candidate : candidates) {

							 //find a possible candidate
							 if (std::fabs(std::fabs(basePoint.main_dir - candidate.main_dir) - M_PI_2) > angleTolerance) {
								 continue;
							 }

							 CheckBoardPoints candidatePointSet(basePoint);
							 candidatePointSet._pointMaps[{0,1}] = candidate;

							 Eigen::Vector2f dist(candidate.pix_coord_x - basePoint.pix_coord_x, candidate.pix_coord_y - basePoint.pix_coord_y);
							 Eigen::Vector2f rotated(-dist.y(), dist.x());

							 //if the grid would be too large avoid straight away
							 if (containingRectDiag/dist.norm() < 0.5*std::min(candidatePointSet._nCols, candidatePointSet._nRows)) {
								 continue;
							 }

							 int toleranceRadius = static_cast<int>(ceil(dist.norm()*relDistanceTolerance));
							 toleranceRadius = std::min(toleranceRadius, maxDistanceRadius);

							 discretCheckCornerInfos rowStart = basePoint;
							 discretCheckCornerInfos rowCurrent = candidate;
							 discretCheckCornerInfos colPrevious;
							 discretCheckCornerInfos colCurrent;

							 Eigen::Vector2f currentRowVec = dist;
							 Eigen::Vector2f currentColVec = rotated;

							 std::set<Coord> selectedCandidates;
							 selectedCandidates.insert({basePoint.pix_coord_x, basePoint.pix_coord_y});
							 selectedCandidates.insert({candidate.pix_coord_x, candidate.pix_coord_y});

							 //find if there is a corresponding point for the next row

							 int search_center_pix_x = static_cast<int>(round(rowStart.pix_coord_x + currentColVec.x()));
							 int search_center_pix_y = static_cast<int>(round(rowStart.pix_coord_y + currentColVec.y()));

							 auto colPreviousCandidate = CheckBoardPoints::findPoint(candidatesMap,
																   search_center_pix_x,
																   search_center_pix_y,
																   toleranceRadius,
																   rowStart.main_dir,
																   selectedCandidates,
																   angleTolerance);

							 if (!colPreviousCandidate.has_value()) {
								 continue; //move to next candidate
							 }

							 colPrevious = colPreviousCandidate.value();

							 search_center_pix_x = static_cast<int>(round(rowStart.pix_coord_x + currentColVec.x() + currentRowVec.x()));
							 search_center_pix_y = static_cast<int>(round(rowStart.pix_coord_y + currentColVec.y() + currentRowVec.y()));

							 auto colCurrentCandidate = CheckBoardPoints::findPoint(candidatesMap,
																   search_center_pix_x,
																   search_center_pix_y,
																   toleranceRadius,
																   (colPrevious.main_dir + rowCurrent.main_dir)/2,
																   selectedCandidates,
																   angleTolerance);

							 if (!colCurrentCandidate.has_value()) {
								 continue; //move to next candidate
							 }

							 colCurrent = colCurrentCandidate.value();

							 selectedCandidates.insert({colPrevious.pix_coord_x, colPrevious.pix_coord_y});
							 selectedCandidates.insert({colCurrent.pix_coord_x, colCurrent.pix_coord_y});

							 candidatePointSet._pointMaps[{1,0}] = colPrevious;
							 candidatePointSet._pointMaps[{1,1}] = colCurrent;
							 candidatePointSet._nCols = 2;
							 candidatePointSet._nRows = 2;


							 std::map<Coord, StateVector, std::less<Coord>> _states_vertical;
							 std::map<Coord, Matrix4x4, std::less<Coord>> _statesVariances_vertical;

							 StateVector v1 = StateVector::Zero();
							 v1[0] = colPrevious.pix_coord_x - rowStart.pix_coord_x;
							 v1[1] = colPrevious.pix_coord_y - rowStart.pix_coord_y;

							 _states_vertical[{1,0}] = v1;
							 _statesVariances_vertical[{1,0}] = Matrix4x4::Identity();

							 StateVector v2 = StateVector::Zero();
							 v2[0] = colCurrent.pix_coord_x - rowCurrent.pix_coord_x;
							 v2[1] = colCurrent.pix_coord_y - rowCurrent.pix_coord_y;

							 _states_vertical[{1,1}] = v2;
							 _statesVariances_vertical[{1,1}] = Matrix4x4::Identity();

							 std::map<Coord, StateVector, std::less<Coord>> _states_horizontal;
							 std::map<Coord, Matrix4x4, std::less<Coord>> _statesVariances_horizontal;

							 v1 = StateVector::Zero();
							 v1[0] = rowCurrent.pix_coord_x - rowStart.pix_coord_x;
							 v1[1] = rowCurrent.pix_coord_y - rowStart.pix_coord_y;

							 _states_horizontal[{0,1}] = v1;
							 _statesVariances_horizontal[{0,1}] = Matrix4x4::Identity();

							 v2 = StateVector::Zero();
							 v2[0] = colCurrent.pix_coord_x - colPrevious.pix_coord_x;
							 v2[1] = colCurrent.pix_coord_y - colPrevious.pix_coord_y;

							 _states_horizontal[{1,1}] = v2;
							 _statesVariances_horizontal[{1,1}] = Matrix4x4::Identity();

							 bool hasGrown = true;
							 bool rowsGrowing = true;
							 bool colsGrowing = true;

							 while (hasGrown) {

								 hasGrown = false;

								 //grow rows
								 if (rowsGrowing) {

									 Coord gridPos = {0,candidatePointSet._nCols-1};
									 discretCheckCornerInfos top = candidatePointSet._pointMaps[{gridPos.x, gridPos.y}];

									 StateVector predicted = TransitionModel*_states_horizontal[gridPos];
									 ObservationVector rowVec = ObservationModel*predicted;

									 search_center_pix_x = static_cast<int>(round(top.pix_coord_x + rowVec.x()));
									 search_center_pix_y = static_cast<int>(round(top.pix_coord_y + rowVec.y()));

									 auto colTopCandidate = CheckBoardPoints::findPoint(candidatesMap,
																	  search_center_pix_x,
																	  search_center_pix_y,
																	  toleranceRadius,
																	  top.main_dir,
																	  selectedCandidates,
																	  angleTolerance);

									 if (colTopCandidate.has_value()) {

										 std::set<Coord> colCandidates = selectedCandidates;
										 std::vector<discretCheckCornerInfos> newCol = {colTopCandidate.value()};
										 newCol.reserve(candidatePointSet._nRows);

										 colCandidates.insert({newCol[0].pix_coord_x, newCol[0].pix_coord_y});

										 Matrix4x4 predictedVar = TransitionModel*_statesVariances_horizontal[{0,candidatePointSet._nCols-1}]*TransitionModel.transpose() + MotionModelCov;

										 ObservationVector innovation;
										 innovation << newCol[0].pix_coord_x - candidatePointSet._pointMaps[{0, candidatePointSet._nCols-1}].pix_coord_x,
												 newCol[0].pix_coord_y - candidatePointSet._pointMaps[{0, candidatePointSet._nCols-1}].pix_coord_y;
										 innovation -= rowVec;

										 Matrix2x2 innovationVar = ObservationModel*predictedVar*ObservationModel.transpose() + ObservationModelCov;

										 Matrix4x2 Gain = predictedVar*ObservationModel.transpose()*innovationVar.completeOrthogonalDecomposition().pseudoInverse();

										 StateVector updatedState = _states_horizontal[{0,candidatePointSet._nCols-1}] + Gain*innovation;
										 Matrix4x4 updatedVar = (Matrix4x4::Identity() - Gain*ObservationModel)*predictedVar;

										 std::vector<StateVector> newHorizontalStates = {updatedState};
										 newHorizontalStates.reserve(candidatePointSet._nRows);

										 std::vector<Matrix4x4> newHorizontalStatesVariance = {updatedVar};
										 newHorizontalStatesVariance.reserve(candidatePointSet._nRows);

										 std::vector<StateVector> newVerticalStates;
										 newHorizontalStates.reserve(candidatePointSet._nRows-1);

										 std::vector<Matrix4x4> newVerticalStatesVariances;
										 newVerticalStatesVariances.reserve(candidatePointSet._nRows-1);

										 for (int i = 1; i < candidatePointSet._nRows; i++) {

											 StateVector previousVert = _states_vertical[{i,candidatePointSet._nCols-1}];

											 if (i > 1) {
												 previousVert = newVerticalStates[i-2];
											 }

											 StateVector predictorHorz = TransitionModel*_states_horizontal[{i,candidatePointSet._nCols-1}];
											 StateVector predictorVert = TransitionModel*previousVert;

											 ObservationVector rowVec = ObservationModel*predictorHorz;
											 ObservationVector colVec = ObservationModel*predictorVert;

											 int search_center_pix_x_h = static_cast<int>(round(candidatePointSet._pointMaps[{i, candidatePointSet._nCols-1}].pix_coord_x + rowVec.x() ));
											 int search_center_pix_y_h = static_cast<int>(round(candidatePointSet._pointMaps[{i, candidatePointSet._nCols-1}].pix_coord_y + rowVec.y() ));

											 int search_center_pix_x_v = static_cast<int>(round(newCol[i-1].pix_coord_x + colVec.x() ));
											 int search_center_pix_y_v = static_cast<int>(round(newCol[i-1].pix_coord_y + colVec.y() ));

											 search_center_pix_x = (search_center_pix_x_h + search_center_pix_x_v)/2;
											 search_center_pix_y = (search_center_pix_y_h + search_center_pix_y_v)/2;

											 int maxErrorRange = std::min(std::abs(search_center_pix_x_h - search_center_pix_x_v),
																		  std::abs(search_center_pix_y_h - search_center_pix_y_v))/2;

											 auto colNextCandidate = CheckBoardPoints::findPoint(candidatesMap,
																			   search_center_pix_x,
																			   search_center_pix_y,
																			   toleranceRadius + std::min(toleranceRadius, maxErrorRange),
																			   newCol[i-1].main_dir,
																			   colCandidates,
																			   angleTolerance);

											 if (!colNextCandidate.has_value()) {
												 break;
											 }

											 newCol.push_back(colNextCandidate.value());

											 colCandidates.insert({newCol[i].pix_coord_x, newCol[i].pix_coord_y});

											 //horizontal KF filter update
											 predictedVar = TransitionModel*_statesVariances_horizontal[{i,candidatePointSet._nCols-1}]*TransitionModel.transpose() + MotionModelCov;

											 innovation << newCol[i].pix_coord_x - candidatePointSet._pointMaps[{i, candidatePointSet._nCols-1}].pix_coord_x,
													 newCol[i].pix_coord_y - candidatePointSet._pointMaps[{i, candidatePointSet._nCols-1}].pix_coord_y;
											 innovation -= rowVec;

											 innovationVar = ObservationModel*predictedVar*ObservationModel.transpose() + ObservationModelCov;

											 Gain = (predictedVar*ObservationModel.transpose()*innovationVar.completeOrthogonalDecomposition().pseudoInverse());

											 updatedState = _states_horizontal[{0,candidatePointSet._nCols-1}] + Gain*innovation;
											 updatedVar = (Matrix4x4::Identity() - Gain*ObservationModel)*predictedVar;

											 newHorizontalStates.push_back(updatedState);
											 newHorizontalStatesVariance.push_back(updatedVar);

											 //vertical KF filter update
											 Matrix4x4 previousVertVar = _statesVariances_vertical[{i,candidatePointSet._nCols-1}];

											 if (i > 1) {
												 previousVertVar = newVerticalStatesVariances[i-2];
											 }

											 predictedVar = TransitionModel*previousVertVar*TransitionModel.transpose() + MotionModelCov;

											 innovation << newCol[i].pix_coord_x - newCol[i-1].pix_coord_x,
													 newCol[i].pix_coord_y - newCol[i-1].pix_coord_y;
											 innovation -= colVec;

											 innovationVar = ObservationModel*predictedVar*ObservationModel.transpose() + ObservationModelCov;

											 Gain = (predictedVar*ObservationModel.transpose()*innovationVar.completeOrthogonalDecomposition().pseudoInverse());

											 updatedState = previousVert + Gain*innovation;
											 updatedVar = (Matrix4x4::Identity() - Gain*ObservationModel)*predictedVar;

											 newVerticalStates.push_back(updatedState);
											 newVerticalStatesVariances.push_back(updatedVar);

										 }

										 if (static_cast<int>(newCol.size()) == candidatePointSet._nRows) {
											 hasGrown = true;

											 for (int i = 0; i < candidatePointSet._nRows; i++) {

												 candidatePointSet._pointMaps[{i,candidatePointSet._nCols}] = newCol[i];

												 _states_horizontal[{i,candidatePointSet._nCols}] = newHorizontalStates[i];
												 _statesVariances_horizontal[{i,candidatePointSet._nCols}] = newHorizontalStatesVariance[i];

												 if (i > 0) {

													 _states_vertical[{i,candidatePointSet._nCols}] = newVerticalStates[i-1];
													 _statesVariances_vertical[{i,candidatePointSet._nCols}] = newVerticalStatesVariances[i-1];
												 }

											 }

											 candidatePointSet._nCols++;
											 selectedCandidates = std::move(colCandidates);

										 } else {
											 rowsGrowing = false;
										 }

									 }
								 }


								 //grow cols
								 if (colsGrowing) {

									 Coord gridPos = {candidatePointSet._nRows-1,0};
									 discretCheckCornerInfos left = candidatePointSet._pointMaps[{gridPos.x, gridPos.y}];

									 StateVector predicted = TransitionModel*_states_vertical[gridPos];
									 ObservationVector colVec = ObservationModel*predicted;

									 search_center_pix_x = static_cast<int>(round(left.pix_coord_x + colVec.x()));
									 search_center_pix_y = static_cast<int>(round(left.pix_coord_y + colVec.y()));

									 auto rowLeftCandidate = CheckBoardPoints::findPoint(candidatesMap,
																	  search_center_pix_x,
																	  search_center_pix_y,
																	  toleranceRadius,
																	  left.main_dir,
																	  selectedCandidates,
																	  angleTolerance);

									 if (rowLeftCandidate.has_value()) {

										 std::set<Coord> rowCandidates = selectedCandidates;
										 std::vector<discretCheckCornerInfos> newRow = {rowLeftCandidate.value()};
										 newRow.reserve(candidatePointSet._nCols);

										 rowCandidates.insert({newRow[0].pix_coord_x, newRow[0].pix_coord_y});

										 Matrix4x4 predictedVar = TransitionModel*_statesVariances_vertical[{candidatePointSet._nRows-1,0}]*TransitionModel.transpose() + MotionModelCov;

										 ObservationVector innovation;
										 innovation << newRow[0].pix_coord_x - candidatePointSet._pointMaps[{candidatePointSet._nRows-1, 0}].pix_coord_x,
												 newRow[0].pix_coord_y - candidatePointSet._pointMaps[{candidatePointSet._nRows-1, 0}].pix_coord_y;
										 innovation -= colVec;

										 Matrix2x2 innovationVar = ObservationModel*predictedVar*ObservationModel.transpose() + ObservationModelCov;

										 Matrix4x2 Gain = predictedVar*ObservationModel.transpose()*innovationVar.completeOrthogonalDecomposition().pseudoInverse();

										 StateVector updatedState = _states_vertical[{candidatePointSet._nRows-1, 0}] + Gain*innovation;
										 Matrix4x4 updatedVar = (Matrix4x4::Identity() - Gain*ObservationModel)*predictedVar;

										 std::vector<StateVector> newVerticalStates = {updatedState};
										 newVerticalStates.reserve(candidatePointSet._nCols);

										 std::vector<Matrix4x4> newVerticalStatesVariances = {updatedVar};
										 newVerticalStatesVariances.reserve(candidatePointSet._nCols);

										 std::vector<StateVector> newHorizontalStates;
										 newHorizontalStates.reserve(candidatePointSet._nCols-1);

										 std::vector<Matrix4x4> newHorizontalStatesVariance;
										 newHorizontalStatesVariance.reserve(candidatePointSet._nCols-1);

										 for (int i = 1; i < candidatePointSet._nCols; i++) {

											 StateVector previousHorz = _states_horizontal[{candidatePointSet._nRows-1,i}];

											 if (i > 1) {
												 previousHorz = newHorizontalStates[i-2];
											 }

											 StateVector predictorVert = TransitionModel*_states_vertical[{candidatePointSet._nRows-1,i}];
											 StateVector predictorHorz = TransitionModel*previousHorz;

											 ObservationVector rowVec = ObservationModel*predictorHorz;
											 ObservationVector colVec = ObservationModel*predictorVert;

											 int search_center_pix_x_v = static_cast<int>(round(candidatePointSet._pointMaps[{candidatePointSet._nRows-1, i}].pix_coord_x + colVec.x() ));
											 int search_center_pix_y_v = static_cast<int>(round(candidatePointSet._pointMaps[{candidatePointSet._nRows-1, i}].pix_coord_y + colVec.y() ));

											 int search_center_pix_x_h = static_cast<int>(round(newRow[i-1].pix_coord_x + rowVec.x() ));
											 int search_center_pix_y_h = static_cast<int>(round(newRow[i-1].pix_coord_y + rowVec.y() ));

											 search_center_pix_x = (search_center_pix_x_h + search_center_pix_x_v)/2;
											 search_center_pix_y = (search_center_pix_y_h + search_center_pix_y_v)/2;

											 int maxErrorRange = std::min(std::abs(search_center_pix_x_h - search_center_pix_x_v),
																		  std::abs(search_center_pix_y_h - search_center_pix_y_v))/2;

											 auto rowNextCandidate = CheckBoardPoints::findPoint(candidatesMap,
																			   search_center_pix_x,
																			   search_center_pix_y,
																			   toleranceRadius + std::min(toleranceRadius, maxErrorRange),
																			   newRow[i-1].main_dir,
																			   rowCandidates,
																			   angleTolerance);

											 if (!rowNextCandidate.has_value()) {
												 break;
											 }

											 newRow.push_back(rowNextCandidate.value());

											 rowCandidates.insert({newRow[i].pix_coord_x, newRow[i].pix_coord_y});

											 //vertical KF filter update
											 predictedVar = TransitionModel*_statesVariances_vertical[{candidatePointSet._nRows-1, i}]*TransitionModel.transpose() + MotionModelCov;

											 innovation << newRow[i].pix_coord_x - candidatePointSet._pointMaps[{candidatePointSet._nRows-1, i}].pix_coord_x,
													 newRow[i].pix_coord_y - candidatePointSet._pointMaps[{candidatePointSet._nRows-1, i}].pix_coord_y;
											 innovation -= colVec;

											 innovationVar = ObservationModel*predictedVar*ObservationModel.transpose() + ObservationModelCov;

											 Gain = (predictedVar*ObservationModel.transpose()*innovationVar.completeOrthogonalDecomposition().pseudoInverse());

											 updatedState = _states_vertical[{candidatePointSet._nRows-1, i}] + Gain*innovation;
											 updatedVar = (Matrix4x4::Identity() - Gain*ObservationModel)*predictedVar;

											 newVerticalStates.push_back(updatedState);
											 newVerticalStatesVariances.push_back(updatedVar);

											 //horizontal KF filter update
											 Matrix4x4 previousHorzVar = _statesVariances_horizontal[{candidatePointSet._nRows-1,i}];

											 if (i > 1) {
												 previousHorzVar = newHorizontalStatesVariance[i-2];
											 }

											 predictedVar = TransitionModel*previousHorzVar*TransitionModel.transpose() + MotionModelCov;

											 innovation << newRow[i].pix_coord_x - newRow[i-1].pix_coord_x,
													 newRow[i].pix_coord_y - newRow[i-1].pix_coord_y;
											 innovation -= rowVec;

											 innovationVar = ObservationModel*predictedVar*ObservationModel.transpose() + ObservationModelCov;

											 Gain = (predictedVar*ObservationModel.transpose()*innovationVar.completeOrthogonalDecomposition().pseudoInverse());

											 updatedState = previousHorz + Gain*innovation;
											 updatedVar = (Matrix4x4::Identity() - Gain*ObservationModel)*predictedVar;

											 newHorizontalStates.push_back(updatedState);
											 newHorizontalStatesVariance.push_back(updatedVar);

										 }

										 if (static_cast<int>(newRow.size()) == candidatePointSet._nCols) {
											 hasGrown = true;

											 for (int i = 0; i < candidatePointSet._nCols; i++) {

												 candidatePointSet._pointMaps[{candidatePointSet._nRows, i}] = newRow[i];

												 _states_vertical[{candidatePointSet._nRows, i}] = newVerticalStates[i];
												 _statesVariances_vertical[{candidatePointSet._nRows, i}] = newVerticalStatesVariances[i];

												 if (i > 0) {

													 _states_horizontal[{candidatePointSet._nRows, i}] = newHorizontalStates[i-1];
													 _statesVariances_horizontal[{candidatePointSet._nRows, i}] = newHorizontalStatesVariance[i-1];
												 }

											 }

											 candidatePointSet._nRows++;
											 selectedCandidates = std::move(rowCandidates);

										 } else {
											 colsGrowing = false;
										 }
									 }
								 }

							 }

							 if (candidatePointSet.nPointsFound() > currentBests.nPointsFound()) {
								 currentBests = candidatePointSet;
							 }
						 }

					 }

					 //orient the checkboard

					 if (currentBests.rows() <= 1 or currentBests.cols() <= 1) { //do not orient if it is not a grid
						 return currentBests;
					 }

					 bool needTranspose = currentBests.rows() > currentBests.cols();
					 int selectedCornerRow = -1;
					 int selectedCornerCol = -1;

					 for (int rowCornerIdx = 0; rowCornerIdx <=1; rowCornerIdx++) {

						 int row = (rowCornerIdx == 0) ? 0 : currentBests._nRows-1;
						 int rowNext = (row > 0) ? row-1 : row+1;

						 for (int colCornerIdx = 0; colCornerIdx <=1; colCornerIdx++) {

							 int col = (colCornerIdx == 0) ? 0 : currentBests._nCols-1;
							 int colNext = (col > 0) ? col-1 : col+1;

							 auto cand = currentBests.pointInCoord(row, col);
							 auto candRow = currentBests.pointInCoord(rowNext, col);
							 auto candCol = currentBests.pointInCoord(row, colNext);

							 if (!cand.has_value() or !candRow.has_value() or !candCol.has_value()) {
								 continue;
							 }

							 discretCheckCornerInfos infos = cand.value();
							 discretCheckCornerInfos nextRow = candRow.value();
							 discretCheckCornerInfos nextCol = candCol.value();

							 Eigen::Vector2f yVec;
							 yVec.x() = nextRow.pix_coord_x - infos.pix_coord_x;
							 yVec.y() = nextRow.pix_coord_y - infos.pix_coord_y;

							 Eigen::Vector2f xVec;
							 xVec.x() = nextCol.pix_coord_x - infos.pix_coord_x;
							 xVec.y() = nextCol.pix_coord_y - infos.pix_coord_y;

							 if (needTranspose) {
								 Eigen::Vector2f tmp = xVec;
								 xVec = yVec;
								 yVec = tmp;
							 }

							 Eigen::Vector2f angleVec;
							 angleVec.x() = std::cos(infos.main_dir);
							 angleVec.y() = std::sin(infos.main_dir);

							 if ((angleVec.dot(yVec) > 0 and angleVec.dot(xVec) > 0) or
								 (angleVec.dot(yVec) < 0 and angleVec.dot(xVec) < 0)) {

								 float cross = xVec.x()*yVec.y() - xVec.y()*yVec.x();

								 if (cross > 0) {

									 selectedCornerRow = row;
									 selectedCornerCol = col;

									 break;
								 }
							 }
						 }

						 if (selectedCornerRow >= 0 and selectedCornerCol >= 0) {
							 break;
						 }
					 }

					 if (selectedCornerRow >= 0 and selectedCornerCol >= 0) {

						 if (selectedCornerRow > 0) {
							 currentBests._rowDelta = selectedCornerRow;
							 currentBests._rowDirection = -1;
						 }

						 if (selectedCornerCol > 0) {
							 currentBests._colDelta = selectedCornerCol;
							 currentBests._colDirection = -1;
						 }

						 if (needTranspose) {
							 currentBests._transpose = true;
						 }
					 }

					 return currentBests;

				 }

template<typename T>
std::vector<discretCheckCornerInfos> checkBoardCornersCandidates(Multidim::Array<T, 2> const& img,
																	 uint8_t smoothRegionRadius= 1,
																	 uint8_t nonMaximumSuppresionRadius = 2,
																	 float lambdaThreshold = 0.0) {

	using ComputeType = std::conditional_t<std::is_integral_v<T>, TypesManipulations::accumulation_extended_t<T>, T>;

	auto shape = img.shape();

	Multidim::Array<ComputeType, 2> xDiff = finiteDifference<T,Geometry::ImageAxis::X, ComputeType>(img);
	Multidim::Array<ComputeType, 2> yDiff = finiteDifference<T,Geometry::ImageAxis::Y, ComputeType>(img);

	Multidim::Array<ComputeType, 2> xxDiff2 = finiteDifference<T,Geometry::ImageAxis::X, ComputeType>(xDiff);
	Multidim::Array<ComputeType, 2> xyDiff2 = finiteDifference<T,Geometry::ImageAxis::Y, ComputeType>(xDiff);
	Multidim::Array<ComputeType, 2> yyDiff2 = finiteDifference<T,Geometry::ImageAxis::Y, ComputeType>(yDiff);

	auto aggregateDiffs = [] (Multidim::Array<ComputeType, 2> & diff, uint8_t smoothRegionRadius) {

		Multidim::Array<ComputeType, 3> unfolded = Correlation::unfold(smoothRegionRadius, smoothRegionRadius, diff);
		Multidim::Array<ComputeType, 2> out (diff.shape());

		for (int i = 0; i < unfolded.shape()[0]; i++) {
			for (int j = 0; j < unfolded.shape()[1]; j++) {

				out.atUnchecked(i,j) = 0;

				for (int f = 0; f < unfolded.shape()[2]; f++) {

					out.atUnchecked(i,j) += unfolded.valueUnchecked(i,j,f);
				}
			}
		}

		return out;

	};

	Multidim::Array<ComputeType, 2> xSq = aggregateDiffs(xxDiff2, smoothRegionRadius);
	Multidim::Array<ComputeType, 2> xy = aggregateDiffs(xyDiff2, smoothRegionRadius);
	Multidim::Array<ComputeType, 2> ySq = aggregateDiffs(yyDiff2, smoothRegionRadius);

	Multidim::Array<float, 2> resp(shape);
	Multidim::Array<float, 2> lambda_min(shape);
	Multidim::Array<float, 2> lambda_max(shape);
	Multidim::Array<float, 2> mainDir(shape);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {

			Eigen::Matrix2f H;
			H(0,0) = xSq.valueUnchecked(i,j);
			H(0,1) = xy.valueUnchecked(i,j);
			H(1,0) = xy.valueUnchecked(i,j);
			H(1,1) = ySq.valueUnchecked(i,j);

			Eigen::EigenSolver<Eigen::Matrix2f> solver(H, true);

			std::complex<float> lmb1 = solver.eigenvalues()[0];
			std::complex<float> lmb2 = solver.eigenvalues()[1];

			if (std::fabs(lmb1.imag()) > 1e-8) {
				//set invalid lambdas
				lambda_min.atUnchecked(i,j) = 1;
				lambda_max.atUnchecked(i,j) = -1;
				continue;
			}

			lambda_min.atUnchecked(i,j) = std::min(lmb1.real(), lmb2.real());
			lambda_max.atUnchecked(i,j) = std::max(lmb1.real(), lmb2.real());

			int maxLambdaIdx = (lmb1.real() == std::max(lmb1.real(), lmb2.real())) ? 0 : 1;
			Eigen::Vector2f maxLambdaEigenvector = solver.eigenvectors().col(maxLambdaIdx).real();
			Eigen::Vector2f minLambdaEigenvector = solver.eigenvectors().col((maxLambdaIdx+1)%2).real();

			int sign = (maxLambdaEigenvector.y() < 0) ? -1 : 1;
			mainDir.atUnchecked(i,j) = std::atan2(sign*maxLambdaEigenvector.y(), sign*maxLambdaEigenvector.x());

			resp.atUnchecked(i,j) = lmb1.real()*lmb2.real();

		}
	}

	Multidim::Array<float, 3> respUnfolded = Correlation::unfold(nonMaximumSuppresionRadius, nonMaximumSuppresionRadius, resp);

	Multidim::Array<float, 2> localMin(shape);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {

			localMin.atUnchecked(i,j) = respUnfolded.valueUnchecked(i,j,0);

			for (int f = 1; f < respUnfolded.shape()[2]; f++) {

				if (localMin.atUnchecked(i,j) > respUnfolded.valueUnchecked(i,j,f)) {
					localMin.atUnchecked(i,j) = respUnfolded.valueUnchecked(i,j,f);
				}
			}

		}
	}

	std::vector<discretCheckCornerInfos> ret;

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {

			if (resp.valueUnchecked(i,j) != localMin.valueUnchecked(i,j)) {
				continue;
			}

			if (lambda_min.valueUnchecked(i,j) > -lambdaThreshold) {
				continue;
			}

			if (lambda_max.valueUnchecked(i,j) < lambdaThreshold) {
				continue;
			}

			ret.emplace_back(j,
							 i,
							 lambda_min.valueUnchecked(i,j),
							 lambda_max.valueUnchecked(i,j),
							 mainDir.valueUnchecked(i,j));

		}
	}

	return ret;

}


template<typename T>
std::vector<discretCheckCornerInfos> checkBoardFilterCandidates(Multidim::Array<T, 2> const& img,
																std::vector<discretCheckCornerInfos> const& candidates,
																float relativeVariationHardTolerance = 0.2,
																float relativeVariationSoftThreshold = 0.6)
{

	std::vector<discretCheckCornerInfos> ret;

	for (discretCheckCornerInfos const& candidate : candidates) {

		if (candidate.pix_coord_x < 3 or candidate.pix_coord_y < 3) {
			continue;
		}

		if (candidate.pix_coord_x > img.shape()[1] - 4 or candidate.pix_coord_y > img.shape()[0] - 4) {
			continue;
		}

		std::array<int, 8> xDeltas = {2, -2, 2, -2, 2, -2, 0, 0};
		std::array<int, 8> yDeltas = {2, -2, 0, 0, -2, 2, 2, -2};

		std::array<float, 8> meanSectionVals;

		int minSectionid = 0;
		int maxSectionid = 0;

		for (int s = 0; s < 8; s++) {

			std::array<T,9> vals;

			int p = 0;

			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					vals[p] = img.valueUnchecked(candidate.pix_coord_y+yDeltas[s]+i, candidate.pix_coord_x+xDeltas[s]+j);
					p++;
				}
			}

			std::sort(vals.begin(), vals.end());

			float interQuantMean = 0;

			for (int i = 2; i <7; i++) {
				interQuantMean +=vals[i];
			}

			meanSectionVals[s] = interQuantMean;

			if (interQuantMean < meanSectionVals[minSectionid]) {
				minSectionid = s;
			}

			if (interQuantMean > meanSectionVals[maxSectionid]) {
				maxSectionid = s;
			}

		}

		int failureCount = 0;
		int errorCount = 0;
		float range = meanSectionVals[maxSectionid] - meanSectionVals[minSectionid];

		for (int sp = 0; sp < 4; sp++) {
			if (std::fabs(meanSectionVals[2*sp] - meanSectionVals[2*sp+1]) > relativeVariationHardTolerance*range) {
				failureCount++;
			}
			if (std::fabs(meanSectionVals[2*sp] - meanSectionVals[2*sp+1]) > relativeVariationSoftThreshold*range) {
				errorCount++;
			}
		}

		if (failureCount <= 3 and errorCount <= 1) {
			ret.push_back(candidate);
		}
	}

	return ret;

}


template<typename T>
Eigen::Matrix<float, 4, 1> fitCheckboardCornerCenterModelOptParameters(Multidim::Array<T, 2> const& img,
																	   Eigen::Vector2i discretCenterEstimate,
																	   float mainDir,
																	   int windowRadius = 3,
																	   int nIter = 5) {

	constexpr int nParams = 4;

	typedef Eigen::Matrix<float, nParams, 1> ParamVector;
	typedef Eigen::Matrix<float, Eigen::Dynamic, nParams> MatrixA;

	int nObs = 2*windowRadius+1;
	nObs = nObs*nObs;

	MatrixA A;
	A.resize(nObs, nParams);

	Eigen::VectorXf obs;
	obs.resize(nObs);

	Eigen::VectorXf xsCords;
	xsCords.resize(nObs);
	Eigen::VectorXf ysCords;
	ysCords.resize(nObs);

	std::vector<T> vals;
	vals.reserve(nObs);

	int idx = 0;
	for (int i = -windowRadius; i <= windowRadius; i++) {

		int py = discretCenterEstimate.y()+i;
		if (py < 0) {
			py = 0;
		}
		if (py >= img.shape()[0]) {
			py = img.shape()[0]-1;
		}

		for (int j = -windowRadius; j <= windowRadius; j++) {

			int px = discretCenterEstimate.x()+j;
			if (px < 0) {
				px = 0;
			}
			if (px >= img.shape()[1]) {
				px = img.shape()[1]-1;
			}

			vals.push_back(img.valueUnchecked(py,px));
			idx++;
		}
	}

	std::vector<T> valsSorted(nObs);
	std::copy(vals.begin(), vals.end(), valsSorted.begin());

	std::sort(valsSorted.begin(), valsSorted.end());

	int blackIdx = nObs/10;
	int whiteIdx = nObs-blackIdx-1;

	float black = valsSorted[blackIdx];
	float white = valsSorted[whiteIdx];

	idx = 0;
	for (int i = -windowRadius; i <= windowRadius; i++) {

		for (int j = -windowRadius; j <= windowRadius; j++) {

			obs[idx] = 2*(vals[idx] - black)/(white-black) * M_PI_2 - M_PI_2; //scale by M_PI_2, so we do not have to scale the atan function

			xsCords[idx] = j;
			ysCords[idx] = i;

			idx++;
		}
	}

	float theta = mainDir - M_PI_4; //adjust coordinate system so that main angle is 45 degree

	ParamVector X;
	X[0] = 0; //x translation
	X[1] = 0; //y translation
	X[2] = std::cos(-theta); //rigid transform diag
	X[3] = std::sin(-theta); //rigid transform non diag

	Eigen::VectorXf estimate;
	estimate.resize(nObs);

	auto computeEstimateVector = [&X, &estimate, windowRadius] () {

		int idx = 0;
		for (int pos_y = -windowRadius; pos_y <= windowRadius; pos_y++) {

			for (int pos_x = -windowRadius; pos_x <= windowRadius; pos_x++) {

				float trsfX = X[2]*pos_x - X[3]*pos_y + X[0];
				float trsfY = X[3]*pos_x + X[2]*pos_y + X[1];

				estimate[idx] = std::atan(trsfX*trsfY);

				idx++;
			}
		}
	};

	auto buildMatrixA = [&X, &A, windowRadius] () {

		int idx = 0;
		for (int pos_y = -windowRadius; pos_y <= windowRadius; pos_y++) {

			for (int pos_x = -windowRadius; pos_x <= windowRadius; pos_x++) {

				float trsfX = X[2]*pos_x - X[3]*pos_y + X[0];
				float trsfY = X[3]*pos_x + X[2]*pos_y + X[1];

				float attenuation = 1/((trsfX*trsfY)*(trsfX*trsfY) + 1); //diff atan

				A(idx,0) = attenuation*trsfY;
				A(idx,1) = attenuation*trsfX;
				A(idx,2) = attenuation*(pos_y*trsfX + pos_x*trsfY);
				A(idx,3) = attenuation*(pos_x*trsfX - pos_y*trsfY);

				idx++;
			}
		}

	};

	for (int i = 0; i < nIter; i++) {
		computeEstimateVector();

		Eigen::VectorXf error = obs - estimate;

		buildMatrixA();

		ParamVector delta = Optimization::leastSquares(A,error);

		X = X+delta;

	}

	return X;

}

template<typename T>
Eigen::Matrix<float, 4, 1> fitCheckboardCornerCenterModelOptParameters(Multidim::Array<T, 2> const& img,
																	   Eigen::Vector2i discretCenterEstimate,
																	   Eigen::Vector2f const& initialCoordinateTransform,
																	   int windowRadius = 3,
																	   int nIter = 5) {

	constexpr int nParams = 4;

	typedef Eigen::Matrix<float, nParams, 1> ParamVector;
	typedef Eigen::Matrix<float, Eigen::Dynamic, nParams> MatrixA;

	int nObs = 2*windowRadius+1;
	nObs = nObs*nObs;

	MatrixA A;
	A.resize(nObs, nParams);

	Eigen::VectorXf obs;
	obs.resize(nObs);

	Eigen::VectorXf xsCords;
	xsCords.resize(nObs);
	Eigen::VectorXf ysCords;
	ysCords.resize(nObs);

	std::vector<T> vals;
	vals.reserve(nObs);

	int yCenter;
	int xCenter;

	int idx = 0;
	for (int i = -windowRadius; i <= windowRadius; i++) {

		int py = discretCenterEstimate.y()+i;
		if (py < 0) {
			py = 0;
		}
		if (py >= img.shape()[0]) {
			py = img.shape()[0]-1;
		}

		for (int j = -windowRadius; j <= windowRadius; j++) {

			int px = discretCenterEstimate.x()+j;
			if (px < 0) {
				px = 0;
			}
			if (px >= img.shape()[1]) {
				px = img.shape()[1]-1;
			}

			vals.push_back(img.valueUnchecked(py,px));
			idx++;
		}
	}

	std::vector<T> valsSorted(nObs);
	std::copy(vals.begin(), vals.end(), valsSorted.begin());

	std::sort(valsSorted.begin(), valsSorted.end());

	int blackIdx = nObs/10;
	int whiteIdx = nObs-blackIdx-1;

	float black = valsSorted[blackIdx];
	float white = valsSorted[whiteIdx];

	idx = 0;
	for (int i = -windowRadius; i <= windowRadius; i++) {

		for (int j = -windowRadius; j <= windowRadius; j++) {

			obs[idx] = 2*(vals[idx] - black)/(white-black) * M_PI_2 - M_PI_2; //scale by M_PI_2, so we do not have to scale the atan function

			xsCords[idx] = j;
			ysCords[idx] = i;

			idx++;
		}
	}

	ParamVector X;
	X[0] = 0; //x translation
	X[1] = 0; //y translation
	X[2] = initialCoordinateTransform[0]; //rigid transform diag
	X[3] = initialCoordinateTransform[1]; //rigid transform non diag

	Eigen::VectorXf estimate;
	estimate.resize(nObs);

	auto computeEstimateVector = [&X, &estimate, windowRadius] () {

		int idx = 0;
		for (int pos_y = -windowRadius; pos_y <= windowRadius; pos_y++) {

			for (int pos_x = -windowRadius; pos_x <= windowRadius; pos_x++) {

				float trsfX = X[2]*pos_x - X[3]*pos_y + X[0];
				float trsfY = X[3]*pos_x + X[2]*pos_y + X[1];

				estimate[idx] = std::atan(trsfX*trsfY);

				idx++;
			}
		}
	};

	auto buildMatrixA = [&X, &A, windowRadius] () {

		int idx = 0;
		for (int pos_y = -windowRadius; pos_y <= windowRadius; pos_y++) {

			for (int pos_x = -windowRadius; pos_x <= windowRadius; pos_x++) {

				float trsfX = X[2]*pos_x - X[3]*pos_y + X[0];
				float trsfY = X[3]*pos_x + X[2]*pos_y + X[1];

				float attenuation = 1/((trsfX*trsfY)*(trsfX*trsfY) + 1); //diff atan

				A(idx,0) = attenuation*trsfY;
				A(idx,1) = attenuation*trsfX;
				A(idx,2) = attenuation*(pos_y*trsfX + pos_x*trsfY);
				A(idx,3) = attenuation*(pos_x*trsfX - pos_y*trsfY);

				idx++;
			}
		}

	};

	for (int i = 0; i < nIter; i++) {
		computeEstimateVector();

		Eigen::VectorXf error = obs - estimate;

		buildMatrixA();

		ParamVector delta = Optimization::leastSquares(A,error);

		X = X+delta;

	}

	return X;

}

inline Eigen::Vector2f deltaFromCornerFitParams(Eigen::Matrix<float, 4, 1> const& X) {

	Eigen::Vector2f Invret;
	Invret.x() = X[0];
	Invret.y() = X[1];

	Eigen::Matrix2f InvModel;
	InvModel(0,0) = X[2];
	InvModel(1,0) = X[3];
	InvModel(0,1) = -X[3];
	InvModel(1,1) = X[2];

	return -InvModel.completeOrthogonalDecomposition().pseudoInverse()*Invret;

}

template<typename T>
Eigen::Vector2f fitCheckboardCornerCenter(Multidim::Array<T, 2> const& img,
										  Eigen::Vector2i discretCenterEstimate,
										  float mainDir,
										  int windowRadius = 3,
										  int nIter = 5) {

	Eigen::Matrix<float, 4, 1> X = fitCheckboardCornerCenterModelOptParameters(img, discretCenterEstimate, mainDir, windowRadius, nIter);

	return deltaFromCornerFitParams(X);

}

template<typename T, int nLevels>
Eigen::Vector2f fitCheckboardCornerCenterHiearchical(std::array<Multidim::Array<T, 2> const*, nLevels> imgsHierarchy,
													 Eigen::Vector2i level0DiscretCenterEstimate,
													 float mainDir,
													 float upscalingFactor,
													 std::array<int, nLevels> searchWindowRadiuses = constantArray<int, nLevels>(3),
													 std::array<int, nLevels> nIters = constantArray<int, nLevels>(5)) {
	Eigen::Vector2i discretePos = level0DiscretCenterEstimate;

	Eigen::Matrix<float, 4, 1> X = fitCheckboardCornerCenterModelOptParameters(*imgsHierarchy[0],
			discretePos,
			mainDir,
			searchWindowRadiuses[0],
			nIters[0]);

	Eigen::Vector2f delta = deltaFromCornerFitParams(X);

	Eigen::Vector2f pos = discretePos.cast<float>() + delta;

	for (int level = 1; level < nLevels; level++) {

		discretePos[0] = static_cast<int>(std::round(upscalingFactor*pos[0]));
		discretePos[1] = static_cast<int>(std::round(upscalingFactor*pos[1]));

		X = fitCheckboardCornerCenterModelOptParameters(*imgsHierarchy[level],
														discretePos,
														X.block<2,1>(2, 0)/upscalingFactor,
														searchWindowRadiuses[level],
														nIters[level]);

		delta = deltaFromCornerFitParams(X);

		pos = discretePos.cast<float>() + delta;
	}

	return pos;

}

template<typename T>
std::vector<refinedCornerInfos> refineCheckBoardCorners(Multidim::Array<T, 2> const& img,
														CheckBoardPoints const& discretePoints) {

	constexpr int searchWindowRadius = 3;
	constexpr int nIter = 5;

	std::vector<refinedCornerInfos> ret;
	ret.reserve(discretePoints.rows()*discretePoints.cols());

	for (int i = 0; i < discretePoints.rows(); i++) {
		for (int j = 0; j < discretePoints.cols(); j++) {

			auto cornerInfos = discretePoints.pointInCoord(i,j);

			if (!cornerInfos.has_value()) {
				continue;
			}

			discretCheckCornerInfos corner = cornerInfos.value();

			Eigen::Vector2i discretePos;
			discretePos.x() = corner.pix_coord_x;
			discretePos.y() = corner.pix_coord_y;

			Eigen::Vector2f delta = fitCheckboardCornerCenter<T>(img,
																 discretePos,
																 corner.main_dir,
																 searchWindowRadius,
																 nIter);

			ret.emplace_back(j,i,cornerInfos->pix_coord_x+delta.x(), cornerInfos->pix_coord_y+delta.y());
		}
	}

	return ret;

}

template<typename T, int nLevels>
std::vector<refinedCornerInfos> upsampleRefineCheckBoardCorners(std::array<Multidim::Array<T, 2> const&, nLevels> imgsHierarchy,
																CheckBoardPoints const& discretePointsLvl0,
																float upscalingFactor,
																std::array<int, nLevels> searchWindowRadiuses = constantArray<int, nLevels>(3)) {

	constexpr int nIter = 5;

	std::vector<refinedCornerInfos> ret;
	ret.reserve(discretePointsLvl0.rows()*discretePointsLvl0.cols());

	for (int i = 0; i < discretePointsLvl0.rows(); i++) {
		for (int j = 0; j < discretePointsLvl0.cols(); j++) {

			auto cornerInfos = discretePointsLvl0.pointInCoord(i,j);

			if (!cornerInfos.has_value()) {
				continue;
			}

			discretCheckCornerInfos corner = cornerInfos.value();

			Eigen::Vector2i discretePos;
			discretePos.x() = corner.pix_coord_x;
			discretePos.y() = corner.pix_coord_y;

			Eigen::Vector2f pos = fitCheckboardCornerCenterHiearchical(imgsHierarchy,
																	   discretePos,
																	   corner.main_dir,
																	   upscalingFactor,
																	   searchWindowRadiuses,
																	   constantArray<int, nLevels>(nIter));

			ret.emplace_back(j,i,pos.x(), pos.y());
		}
	}

	return ret;

}

} //namespace StereoVision

#endif // CHECKBOARDDETECTION_H
