#ifndef LIBSTEVI_CHECKBOARDDETECTION_H
#define LIBSTEVI_CHECKBOARDDETECTION_H

#include "../utils/types_manipulations.h"
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
	CheckBoardPoints();
	CheckBoardPoints(discretCheckCornerInfos initial);

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


	friend CheckBoardPoints isolateCheckBoard(std::vector<discretCheckCornerInfos> const& candidates,
											  float relDistanceTolerance,
											  float angleTolerance,
											  float observationWeight,
											  int maxDistanceRadius);
};

CheckBoardPoints isolateCheckBoard(std::vector<discretCheckCornerInfos> const& candidates,
								   float distanceTolerance = 0.05,
								   float angleTolerance = 0.05,
								   float observationWeight = 1.,
								   int maxDistanceRadius = 20);

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

			interQuantMean;

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
Eigen::Vector2f fitCheckboardCornerCenter(Multidim::Array<T, 2> const& img,
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

} //namespace StereoVision

#endif // CHECKBOARDDETECTION_H
