#include "checkBoardDetection.h"

#include <math.h>
#include <Eigen/Core>

#include <map>

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

CheckBoardPoints StereoVision::isolateCheckBoard(std::vector<discretCheckCornerInfos> const& candidates,
												 float relDistanceTolerance,
												 float angleTolerance) {

	CheckBoardPoints currentBests;

	std::map<Coord, discretCheckCornerInfos> candidatesMap;

	for (discretCheckCornerInfos const& candidate : candidates) {
		candidatesMap[{candidate.pix_coord_x, candidate.pix_coord_y}] = candidate;
	}

	for (discretCheckCornerInfos const& basePoint : candidates) {

		for (discretCheckCornerInfos const& candidate : candidates) {

			if (std::fabs(std::fabs(basePoint.main_dir - candidate.main_dir) - M_PI_2) > angleTolerance) {
				continue;
			}

			CheckBoardPoints candidatePointSet(basePoint);
			candidatePointSet._pointMaps[{0,1}] = candidate;
			candidatePointSet._nCols = 2;
			candidatePointSet._nRows = 1;

			Eigen::Vector2f dist(candidate.pix_coord_x - basePoint.pix_coord_x, candidate.pix_coord_y - basePoint.pix_coord_y);
			Eigen::Vector2f rotated(-dist.y(), dist.x());

			int toleranceRadius = static_cast<int>(ceil(dist.norm()*relDistanceTolerance));

			bool goonRow = true;
			bool goonCol = true;

			int currentRow = 0;
			int nInRow = 2;

			discretCheckCornerInfos rowStart = basePoint;
			discretCheckCornerInfos rowCurrent = candidate;

			Eigen::Vector2f currentRowVec = dist;
			Eigen::Vector2f currentColVec = rotated;

			while (goonCol) {

				while (goonRow) {

					bool found = false;

					int search_center_pix_x = static_cast<int>(round(rowCurrent.pix_coord_x + currentRowVec.x()));
					int search_center_pix_y = static_cast<int>(round(rowCurrent.pix_coord_y + currentRowVec.y()));

					for (int dx = -toleranceRadius; dx <= toleranceRadius; dx++) {
						for (int dy = -toleranceRadius; dy <= toleranceRadius; dy++) {

							if (candidatesMap.count({search_center_pix_x+dx, search_center_pix_y+dy}) <= 0) {
								continue;
							}

							discretCheckCornerInfos& subcandidate = candidatesMap[{search_center_pix_x+dx, search_center_pix_y+dy}];

							if (std::fabs(std::fabs(rowCurrent.main_dir - subcandidate.main_dir) - M_PI_2) > angleTolerance) {
								continue;
							}


							Eigen::Vector2f move(subcandidate.pix_coord_x - rowCurrent.pix_coord_x, subcandidate.pix_coord_y - rowCurrent.pix_coord_y);

							if ((move - currentRowVec).norm() > toleranceRadius) {
								continue;
							}

							rowCurrent = subcandidate;
							currentRowVec = move;

							candidatePointSet._pointMaps[{currentRow,nInRow}] = subcandidate;

							if (candidatePointSet._nCols <= nInRow) {
								candidatePointSet._nCols++;
							}

							nInRow++;
							found = true;

							break;
						}

						if (found) {
							break;
						}
					}

					if (!found) {
						break;
					}
				}

				bool newRowFound = false;

				int search_center_pix_x = static_cast<int>(round(rowStart.pix_coord_x + currentColVec.x()));
				int search_center_pix_y = static_cast<int>(round(rowStart.pix_coord_y + currentColVec.y()));

				for (int dx = -toleranceRadius; dx <= toleranceRadius; dx++) {
					for (int dy = -toleranceRadius; dy <= toleranceRadius; dy++) {

						if (candidatesMap.count({search_center_pix_x+dx, search_center_pix_y+dy}) <= 0) {
							continue;
						}

						discretCheckCornerInfos& subcandidate = candidatesMap[{search_center_pix_x+dx, search_center_pix_y+dy}];

						if (std::fabs(std::fabs(rowStart.main_dir - subcandidate.main_dir) - M_PI_2) > angleTolerance) {
							continue;
						}

						Eigen::Vector2f move(subcandidate.pix_coord_x - rowStart.pix_coord_x, subcandidate.pix_coord_y - rowStart.pix_coord_y);

						if ((move - currentColVec).norm() > toleranceRadius) {
							continue;
						}

						rowStart = subcandidate;
						rowCurrent = subcandidate;
						currentColVec = move;

						currentRow++;
						nInRow = 0;

						candidatePointSet._pointMaps[{currentRow,nInRow}] = subcandidate;

						candidatePointSet._nRows++;

						nInRow++;
						newRowFound = true;

						break;
					}

					if (newRowFound) {
						break;
					}
				}

				if (!newRowFound) {
					break;
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
