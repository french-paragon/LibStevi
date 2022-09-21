#include "checkBoardDetection.h"

#include <math.h>
#include <eigen3/Eigen/Core>
#include<eigen3/Eigen/StdVector>

#include <map>
#include <set>

using namespace StereoVision;


CheckBoardPoints::CheckBoardPoints() :
	_pointMaps(),
	_rowDelta(0),
	_colDelta(0),
	_nRows(0),
	_nCols(0)
{
}


CheckBoardPoints::CheckBoardPoints(discretCheckCornerInfos initial) :
	_pointMaps(),
	_rowDelta(0),
	_colDelta(0),
	_nRows(1),
	_nCols(1)
{
	_pointMaps[{0,0}] = initial;
}

int CheckBoardPoints::rows() {
	return _nRows;
}
int CheckBoardPoints::cols() {
	return _nCols;
}

int CheckBoardPoints::nPointsFound() {
	return _pointMaps.size();
}

bool CheckBoardPoints::hasPointInCoord(int row, int col) {
	return _pointMaps.count({row, col}) > 0;
}

std::optional<discretCheckCornerInfos> CheckBoardPoints::pointInCoord(int row, int col) {
	if (hasPointInCoord(row, col)) {
		return _pointMaps[{row, col}];
	}
	return std::nullopt;
}

struct Coord {
	int x;
	int y;
};

bool operator<(Coord c1, Coord c2) {
	if (c1.x == c2.x) {
		return c1.y < c2.y;
	}
	return  c1.x < c2.x;
}

inline std::optional<discretCheckCornerInfos> findPoint(std::map<Coord, discretCheckCornerInfos> & candidatesMap,
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

CheckBoardPoints StereoVision::isolateCheckBoard(std::vector<discretCheckCornerInfos> const& candidates,
												 float relDistanceTolerance,
												 float angleTolerance,
												 float observationWeight,
												 int maxDistanceRadius) {

	using StateVector = Eigen::Matrix<float,4,1>;
	using ObservationVector = Eigen::Matrix<float,2,1>;

	using Matrix4x4 = Eigen::Matrix<float,4,4>;
	using Matrix2x2 = Eigen::Matrix<float,2,2>;
	using Matrix2x4 = Eigen::Matrix<float,2,4>;
	using Matrix4x2 = Eigen::Matrix<float,4,2>;

	CheckBoardPoints currentBests;

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

			auto colPreviousCandidate = findPoint(candidatesMap,
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

			auto colCurrentCandidate = findPoint(candidatesMap,
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


			std::map<Coord, StateVector, std::less<Coord>, Eigen::aligned_allocator<std::pair<Coord, StateVector>>> _states_vertical;
			std::map<Coord, Matrix4x4, std::less<Coord>, Eigen::aligned_allocator<std::pair<Coord, Matrix4x4>>> _statesVariances_vertical;

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

			std::map<Coord, StateVector, std::less<Coord>, Eigen::aligned_allocator<std::pair<Coord, StateVector>>> _states_horizontal;
			std::map<Coord, Matrix4x4, std::less<Coord>, Eigen::aligned_allocator<std::pair<Coord, Matrix4x4>>> _statesVariances_horizontal;

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

					auto colTopCandidate = findPoint(candidatesMap,
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

						std::vector<StateVector,Eigen::aligned_allocator<StateVector>> newHorizontalStates = {updatedState};
						newHorizontalStates.reserve(candidatePointSet._nRows);

						std::vector<Matrix4x4,Eigen::aligned_allocator<Matrix4x4>> newHorizontalStatesVariance = {updatedVar};
						newHorizontalStatesVariance.reserve(candidatePointSet._nRows);

						std::vector<StateVector,Eigen::aligned_allocator<StateVector>> newVerticalStates;
						newHorizontalStates.reserve(candidatePointSet._nRows-1);

						std::vector<Matrix4x4,Eigen::aligned_allocator<Matrix4x4>> newVerticalStatesVariances;
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

							auto colNextCandidate = findPoint(candidatesMap,
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

					auto rowLeftCandidate = findPoint(candidatesMap,
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

						std::vector<StateVector,Eigen::aligned_allocator<StateVector>> newVerticalStates = {updatedState};
						newVerticalStates.reserve(candidatePointSet._nCols);

						std::vector<Matrix4x4,Eigen::aligned_allocator<Matrix4x4>> newVerticalStatesVariances = {updatedVar};
						newVerticalStatesVariances.reserve(candidatePointSet._nCols);

						std::vector<StateVector,Eigen::aligned_allocator<StateVector>> newHorizontalStates;
						newHorizontalStates.reserve(candidatePointSet._nCols-1);

						std::vector<Matrix4x4,Eigen::aligned_allocator<Matrix4x4>> newHorizontalStatesVariance;
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

							auto rowNextCandidate = findPoint(candidatesMap,
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

	std::cout << "current best: " << currentBests.nPointsFound() << std::endl;

	return currentBests;

}
