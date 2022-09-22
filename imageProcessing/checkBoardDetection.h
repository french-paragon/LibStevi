#ifndef LIBSTEVI_CHECKBOARDDETECTION_H
#define LIBSTEVI_CHECKBOARDDETECTION_H

#include "../utils/types_manipulations.h"
#include "../geometry/imagecoordinates.h"
#include "../correlation/unfold.h"

#include "./finiteDifferences.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

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
		return _nRows;
	}
	inline int cols() const {
		return _nCols;
	}

	inline int nPointsFound() const {
		return _pointMaps.size();
	}

	inline bool hasPointInCoord(int row, int col) const {
		return _pointMaps.count({row, col}) > 0;
	}

	inline std::optional<discretCheckCornerInfos> pointInCoord(int row, int col) const {
		if (hasPointInCoord(row, col)) {
			return _pointMaps.at({row, col});
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
std::vector<refinedCornerInfos> refineCheckBoardCorners(Multidim::Array<T, 2> const& img,
														CheckBoardPoints const& discretePoints) {

	constexpr int searchWindowRadius = 2;
	constexpr int nParams = 5;
	constexpr int nObs = (2*searchWindowRadius+1)*(2*searchWindowRadius+1);

	using DerivativeMatrixType = Eigen::Matrix<float, nObs, nParams>;
	using ObsVectorType = Eigen::Matrix<float, nObs, 1>;

	std::vector<refinedCornerInfos> ret;
	ret.reserve(discretePoints.rows()*discretePoints.cols());

	DerivativeMatrixType derivative;

	ObsVectorType obs;
	obs.resize(nObs);

	int idx = 0;
	for (int dr = -searchWindowRadius; dr <= searchWindowRadius; dr++) {
		for (int dc = -searchWindowRadius; dc <= searchWindowRadius; dc++) {
			derivative(idx,0) = dr*dr;
			derivative(idx,1) = dr*dc;
			derivative(idx,2) = dc*dc;
			derivative(idx,3) = dr;
			derivative(idx,4) = dc;
			idx++;
		}
	}

	Eigen::FullPivHouseholderQR<DerivativeMatrixType> qr = derivative.fullPivHouseholderQr();

	for (int i = 0; i < discretePoints.rows(); i++) {
		for (int j = 0; j < discretePoints.cols(); j++) {

			auto cornerInfos = discretePoints.pointInCoord(i,j);

			if (!cornerInfos.has_value()) {
				continue;
			}

			float mean = 0;
			for (int dr = -searchWindowRadius-1; dr <= searchWindowRadius+1; dr++) {
				for (int dc = -searchWindowRadius-1; dc <= searchWindowRadius+1; dc++) {
					int coordX = cornerInfos->pix_coord_x + dr;
					int coordY = cornerInfos->pix_coord_y + dc;

					if (coordX < 0) {
						coordX = 0;
					}

					if (coordY < 0) {
						coordY = 0;
					}

					if (coordX >= img.shape()[1]) {
						coordX = img.shape()[1]-1;
					}

					if (coordY < img.shape()[0]) {
						coordY = img.shape()[0]-1;
					}

					mean += static_cast<float>(img.valueUnchecked(coordY, coordX));
				}
			}

			mean /= nObs;

			int idx = 0;
			for (int dr = -searchWindowRadius; dr <= searchWindowRadius; dr++) {
				for (int dc = -searchWindowRadius; dc <= searchWindowRadius; dc++) {
					int coordX = cornerInfos->pix_coord_x + dr;
					int coordY = cornerInfos->pix_coord_y + dc;

					if (coordX < 1) {
						coordX = 1;
					}

					if (coordY < 1) {
						coordY = 1;
					}

					if (coordX >= img.shape()[1]-1) {
						coordX = img.shape()[1]-2;
					}

					if (coordY < img.shape()[0]-1) {
						coordY = img.shape()[0]-2;
					}

					float val = 0;


					for (int d1 = -1; d1 <= 1; d1++) {
						for (int d2 = -1; d2 <= 1; d2++) {

							val += static_cast<float>(img.valueUnchecked(coordY+d1, coordX+d2)) - mean;
						}
					}

					obs(idx) = val;
					idx++;
				}
			}


			Eigen::Matrix<float, nParams, 1> paramsEst = qr.solve(obs);

			Eigen::Matrix2f gradMat;
			gradMat(0,0) = 2*paramsEst(0);
			gradMat(0,1) = paramsEst(1);
			gradMat(1,0) = paramsEst(1);
			gradMat(1,1) = 2*paramsEst(2);

			Eigen::Vector2f gradConst;
			gradConst(0) = paramsEst(3);
			gradConst(1) = paramsEst(4);

			Eigen::Vector2f delta = -gradMat.completeOrthogonalDecomposition().pseudoInverse()*gradConst;

			ret.emplace_back(j,i,cornerInfos->pix_coord_x+delta.x(), cornerInfos->pix_coord_y+delta.y());
		}
	}

	return ret;

}

} //namespace StereoVision

#endif // CHECKBOARDDETECTION_H
