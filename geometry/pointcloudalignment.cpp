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

#include "pointcloudalignment.h"

#include "geometricexception.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <iostream>

namespace StereoVision {
namespace Geometry {

AffineTransform estimateAffineMap(Eigen::VectorXf const& obs,
												Eigen::Matrix3Xf const& pts,
												std::vector<int> const& idxs,
												std::vector<Axis> const& coordinate) {

	typedef Eigen::Matrix<float, 12, 1, Eigen::ColMajor> ParamVector;
	typedef Eigen::Matrix<float, 12, 12> ParamMatrix;
	typedef Eigen::Matrix<float, Eigen::Dynamic, 12, Eigen::RowMajor> MatrixA;


	int n_obs = obs.rows();

	ParamVector x = ParamVector::Zero();
	ParamVector offset = ParamVector::Zero();
	offset[0] = 1;
	offset[4] = 1;
	offset[8] = 1;

	AffineTransform transform;

	MatrixA A;
	A.setZero(n_obs, 12);

	for (size_t i = 0; i < idxs.size(); i++) {
		switch (coordinate[i]) {
		case Axis::X:
			A.block<1,3>(i,0) = pts.col(idxs[i]).transpose();
			A(i,9) = 1;
			break;
		case Axis::Y:
			A.block<1,3>(i,3) = pts.col(idxs[i]).transpose();
			A(i,10) = 1;
			break;
		case Axis::Z:
			A.block<1,3>(i,6) = pts.col(idxs[i]).transpose();
			A(i,11) = 1;
			break;
		}
	}

	ParamMatrix invQxx = A.transpose()*A;

	auto svd = invQxx.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

	ParamMatrix pseudoInverse = svd.matrixV() * (svd.singularValues().array().abs() > 1e-4).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().transpose();

	x = pseudoInverse*A.transpose()*(obs - A*offset);
	x += offset;

	transform.R.row(0) = x.block<3,1>(0,0);
	transform.R.row(1) = x.block<3,1>(3,0);
	transform.R.row(2) = x.block<3,1>(6,0);
	transform.t = x.block<3,1>(9,0);

	return transform;

}

AffineTransform estimateQuasiShapePreservingMap(Eigen::VectorXf const& obs,
												Eigen::Matrix3Xf const& pts,
												std::vector<int> const& idxs,
												std::vector<Axis> const& coordinate,
												float damping,
												IterativeTermination * status,
												float incrLimit,
												int iterationLimit,
												bool verbose) {

	typedef Eigen::Matrix<float, 12, 1, Eigen::ColMajor> ParamVector;
	typedef Eigen::Matrix<float, 12, 12> ParamMatrix;
	typedef Eigen::Matrix<float, Eigen::Dynamic, 12, Eigen::RowMajor> MatrixA;


	if (verbose) {
		std::cout << "Start estimating QuasiShapePreservingMap:" << std::endl;
	}

	int n_obs = obs.rows();
	int n_eqs = n_obs + 5;

	Eigen::VectorXf extObs;
	extObs.resize(n_eqs,1);
	extObs.block(0,0,n_obs,1) = obs;
	extObs.block(n_obs,0,5,1).setConstant(0);

	ParamVector x = ParamVector::Zero();
	ParamVector offset = ParamVector::Zero();
	offset[0] = 1;
	offset[4] = 1;
	offset[8] = 1;

	MatrixA A;
	A.setZero(n_eqs, 12);

	for (int i = 0; i < n_obs; i++) {
		switch (coordinate[i]) {
		case Axis::X:
			A.block<1,3>(i,0) = pts.col(idxs[i]).transpose();
			A(i,9) = 1;
			break;
		case Axis::Y:
			A.block<1,3>(i,3) = pts.col(idxs[i]).transpose();
			A(i,10) = 1;
			break;
		case Axis::Z:
			A.block<1,3>(i,6) = pts.col(idxs[i]).transpose();
			A(i,11) = 1;
			break;
		}
	}

	IterativeTermination s = IterativeTermination::MaxStepReached;

	for (int i = 0; i < iterationLimit; i++) {

		//<R1, R2> = 0
		A.block<1,3>(n_obs, 0) = x.block<3,1>(3,0).transpose() + Eigen::Vector3f(0,1,0).transpose();
		A.block<1,3>(n_obs, 3) = x.block<3,1>(0,0).transpose() + Eigen::Vector3f(1,0,0).transpose();

		//<R1, R3> = 0
		A.block<1,3>(n_obs+1, 0) = x.block<3,1>(6,0).transpose() + Eigen::Vector3f(0,0,1).transpose();
		A.block<1,3>(n_obs+1, 6) = x.block<3,1>(0,0).transpose() + Eigen::Vector3f(1,0,0).transpose();

		//<R2, R3> = 0
		A.block<1,3>(n_obs+2, 3) = x.block<3,1>(6,0).transpose() + Eigen::Vector3f(0,0,1).transpose();
		A.block<1,3>(n_obs+2, 6) = x.block<3,1>(3,0).transpose() + Eigen::Vector3f(0,1,0).transpose();

		//<R1,R1> - <R2,R2> = 0
		A.block<1,3>(n_obs+3, 0) = x.block<3,1>(0,0).transpose() + Eigen::Vector3f(1,0,0).transpose();
		A.block<1,3>(n_obs+3, 3) = -x.block<3,1>(3,0).transpose() - Eigen::Vector3f(0,1,0).transpose();

		//<R1,R1> - <R3,R3> = 0
		A.block<1,3>(n_obs+4, 0) = x.block<3,1>(0,0).transpose() + Eigen::Vector3f(1,0,0).transpose();
		A.block<1,3>(n_obs+4, 6) = -x.block<3,1>(6,0).transpose() - Eigen::Vector3f(0,0,1).transpose();


		ParamMatrix invQxx = A.transpose()*A;

		auto svd = invQxx.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

		ParamMatrix pseudoInverse = svd.matrixV() * (svd.singularValues().array().abs() > 1e-6).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().transpose();

		ParamVector dx = pseudoInverse*A.transpose()*(extObs - A*offset - A*x);
		x += damping*dx;

		float n = dx.norm()/12.;

		if (verbose) {
			std::cout << "\t" << "Iteration " << i << ": incr_rms = " << n << std::endl;
		}

		if (n < incrLimit) {
			s = IterativeTermination::Converged;
			break;
		}

	}

	x += offset;

	if (status != nullptr) {
		*status = s;
	}

	if (verbose) {
		std::cout << ((s == IterativeTermination::Converged) ? "Convverged" : "Terminated before convergence reached") << std::endl << std::endl;
	}

	AffineTransform transform;

	transform.R.row(0) = x.block<3,1>(0,0);
	transform.R.row(1) = x.block<3,1>(3,0);
	transform.R.row(2) = x.block<3,1>(6,0);
	transform.t = x.block<3,1>(9,0);

	return transform;

}

ShapePreservingTransform affine2ShapePreservingMap(AffineTransform const & initial) {

	ShapePreservingTransform T;
	T.t = initial.t;

	auto svd = initial.R.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

	Eigen::Matrix3f U = svd.matrixU();
	Eigen::Matrix3f V = svd.matrixV();

	float Udet = U.determinant();
	float Vdet = V.determinant();

	if (Udet*Vdet < 0) {
		U = -U;
	}

	T.r = inverseRodriguezFormula(U*V.transpose());

	T.s = svd.singularValues().mean();

	if (Udet*Vdet < 0) {
		T.s = -T.s;
	}

	return T;

}


ShapePreservingTransform estimateTranslationMap(Eigen::VectorXf const& obs,
												Eigen::Matrix3Xf const& pts,
												std::vector<int> const& idxs,
												std::vector<Axis> const& coordinate,
												float * residual,
												bool verbose) {

	typedef Eigen::Matrix<float, 3, 1, Eigen::ColMajor> ParamVector;
	typedef Eigen::Matrix<float, 3, 3> ParamMatrix;
	typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixA;


	if (verbose) {
		std::cout << "Start estimating QuasiShapePreservingMap:" << std::endl;
	}

	int n_obs = obs.rows();

	Eigen::VectorXf deltaObs;
	deltaObs.resize(n_obs,1);
	deltaObs.block(0,0,n_obs,1) = obs;

	MatrixA A;
	A.setZero(n_obs, 3);

	for (int i = 0; i < n_obs; i++) {
		switch (coordinate[i]) {
		case Axis::X:
			A(i,0) = 1;
			deltaObs[i] -= pts(0,idxs[i]);
			break;
		case Axis::Y:
			A(i,1) = 1;
			deltaObs[i] -= pts(1,idxs[i]);
			break;
		case Axis::Z:
			A(i,2) = 1;
			deltaObs[i] -= pts(2,idxs[i]);
			break;
		}
	}


	ParamMatrix invQxx = A.transpose()*A;

	auto svd = invQxx.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

	ParamMatrix pseudoInverse = svd.matrixV() * (svd.singularValues().array().abs() > 1e-6).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().transpose();

	ParamVector opt = pseudoInverse*A.transpose()*deltaObs;

	if (residual != nullptr) {
		*residual = (A*opt - deltaObs).norm()/n_obs;
	}

	ShapePreservingTransform optimal(Eigen::Vector3f::Zero(), opt, 1.);
	return optimal;
}



ShapePreservingTransform estimateScaleMap(Eigen::VectorXf const& obs,
										  Eigen::Matrix3Xf const& pts,
										  std::vector<int> const& idxs,
										  std::vector<Axis> const& coordinate,
										  float *residual,
										  bool verbose) {

	typedef Eigen::Matrix<float, Eigen::Dynamic, 1> MatrixA;


	if (verbose) {
		std::cout << "Start estimating QuasiShapePreservingMap:" << std::endl;
	}

	int n_obs = obs.rows();

	MatrixA A;
	A.setZero(n_obs, 1);

	for (int i = 0; i < n_obs; i++) {
		switch (coordinate[i]) {
		case Axis::X:
			A[i] = pts(0,idxs[i]);
			break;
		case Axis::Y:
			A[i] = pts(1,idxs[i]);
			break;
		case Axis::Z:
			A[i] = pts(2,idxs[i]);
			break;
		}
	}

	int n = 0;
	float s = 0;

	for (int i = 0; i < n_obs; i++) {
		if (std::fabs(A[i]) > 1e-6) {
			s += obs[i] / A[i];
			n++;
		}
	}

	if (n == 0) {
		s = 1;
	} else {
		s /= n;
	}

	if (residual != nullptr) {
		*residual = (A*s - obs).norm()/n_obs;
	}

	ShapePreservingTransform optimal(Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), s);
	return optimal;
}

ShapePreservingTransform estimateRotationMap(Eigen::VectorXf const& obs,
											 Eigen::Matrix3Xf const& pts,
											 std::vector<int> const& idxs,
											 std::vector<Axis> const& coordinate,
											 float *residual,
											 IterativeTermination * status,
											 bool verbose,
											 int n_steps,
											 float incrLimit) {

	typedef Eigen::Matrix<float, 3, 1, Eigen::ColMajor> ParamVector;
	typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixA;


	if (verbose) {
		std::cout << "Start estimating Rotation Map:" << std::endl;
	}
	int nobs = obs.rows();

	MatrixA A;

	Eigen::VectorXf f0;

	ShapePreservingTransform current(ParamVector::Zero(), Eigen::Vector3f::Zero(), 1);


	IterativeTermination stat = IterativeTermination::MaxStepReached;

	for(int s = 0; s < n_steps; s++) {

		A.setZero(nobs, 3);
		f0.setZero(nobs);

		//compute f0 (functional model is rodriguez formula
		Eigen::Matrix3Xf const& tpts = current*pts;

		for (int i = 0; i < nobs; i++) {

			switch (coordinate[i]) {
			case Axis::X:
				f0[i] = tpts(0,idxs[i]);
				break;
			case Axis::Y:
				f0[i] = tpts(1,idxs[i]);
				break;
			case Axis::Z:
				f0[i] = tpts(2,idxs[i]);
				break;
			}
		}

		//rodriguez formula is:
		//v cos(norm(theta)) + (theta cross v) * sin(norm(theta))/norm(theta) + theta (theta dot v)(1 - cos(norm(theta)))

		//compute A
		for (int i = 0; i < nobs; i++) {

			Eigen::Vector3f p = pts.col(idxs[i]);
			Eigen::Matrix3f skew = StereoVision::Geometry::skew(p);

			switch (coordinate[i]) { //small angle approximation to make life easier
				case Axis::X:
					A.row(i) = -skew.row(0);
					break;
				case Axis::Y:
					A.row(i) = -skew.row(1);
					break;
				case Axis::Z:
					A.row(i) = -skew.row(2);
					break;
			}

		}

		Eigen::MatrixXf invQxx = A.transpose()*A;

		auto svd = invQxx.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

		Eigen::MatrixXf pseudoInverse = svd.matrixV() * (svd.singularValues().array().abs() > 1e-6).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().transpose();

		Eigen::VectorXf r = obs - f0;
		Eigen::Vector3f delta = pseudoInverse*A.transpose()*r;

		ShapePreservingTransform change(delta, Eigen::Vector3f::Zero(), 1);
		current = change*current;

		if (verbose) {
			std::cout << "\t" << "Iteration " << s << ": incr_rms = " << delta.norm()/(3) << std::endl;
		}

		if (delta.norm()/(3) < incrLimit) {
			stat = IterativeTermination::Converged;
			break;
		}

	}

	if (status != nullptr) {
		*status = stat;
	}

	ShapePreservingTransform& r = current;

	Eigen::Matrix3Xf tpts = r*pts;

	if (residual != nullptr) {

		float res = 0;
		for(int i = 0; i < nobs; i++) {

			int col;

			if (coordinate[i] == Axis::X) {
				col = 0;
			}

			if (coordinate[i] == Axis::Y) {
				col = 1;
			}

			if (coordinate[i] == Axis::Z) {
				col = 2;
			}

			float diff = obs[i] - tpts(col, idxs[i]);

			res += diff*diff;
		}

		res = sqrt(res)/nobs;

		*residual = res;
	}

	return r;

}


ShapePreservingTransform estimateShapePreservingMap(Eigen::VectorXf const& obs,
													Eigen::Matrix3Xf const& pts,
													std::vector<int> const& idxs,
													std::vector<Axis> const& coordinate,
													IterativeTermination *status,
													int n_steps,
													float incrLimit,
													float damping,
													float dampingScale) {

	typedef Eigen::Matrix<float, 7, 1, Eigen::ColMajor> ParamVector;
	typedef Eigen::Matrix<float, 7, 7> MatrixQxx ;
	typedef Eigen::Matrix<float, Eigen::Dynamic, 7, Eigen::RowMajor> MatrixA ;

	ShapePreservingTransform current(Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), 1);

	//check if the initializer can do most of the job already.
	auto test = initShapePreservingMapEstimate(obs, pts, idxs, coordinate);

	if (test.has_value()) {
		ShapePreservingTransform possible = test.value();

		if (possible.isFinite()) {
			current = possible;
		}
	}

	int n_obs = obs.rows();

	IterativeTermination stat = IterativeTermination::MaxStepReached;

	MatrixA A;

	Eigen::VectorXf f0 = obs;

	for(int i = 0; i < n_steps; i++) {

		Eigen::Matrix3Xf tpts = current*pts;

		A.setZero(n_obs, 7);

		for (int i = 0; i < static_cast<int>(idxs.size()); i++) {

			int id_row;

			if (coordinate[i] == Axis::X) {
				id_row = 0;
			}

			if (coordinate[i] == Axis::Y) {
				id_row = 1;
			}

			if (coordinate[i] == Axis::Z) {
				id_row = 2;
			}

			f0[i] = tpts(id_row,idxs[i]);

			Eigen::Vector3f p = tpts.col(idxs[i]);
			Eigen::Matrix3f skew = StereoVision::Geometry::skew(p);

			A.block<1,3>(i,0) = -skew.row(id_row); //Param rx, ry, rz, using the small angle approximation for rodriguez formula

			A(i, 3) = (coordinate[i] == Axis::X) ? 1 : 0; //Param x;
			A(i, 4) = (coordinate[i] == Axis::Y) ? 1 : 0; //Param y;
			A(i, 5) = (coordinate[i] == Axis::Z) ? 1 : 0; //Param z;

			A(i, 6) = p[id_row]; //Param s;
		}

		MatrixQxx invQxx = A.transpose()*A;

		auto svd = invQxx.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

		Eigen::MatrixXf pseudoInverse = svd.matrixV() * (svd.singularValues().array().abs() > 1e-6).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().transpose();

		Eigen::VectorXf r = obs - f0;
		ParamVector delta = pseudoInverse*A.transpose()*r;
		delta.block<6,1>(0,0) *= damping;
		delta[6] *= dampingScale;

		ShapePreservingTransform change(delta.block<3,1>(0,0), delta.block<3,1>(3,0), exp(delta[6]));
		current = change*current;

		float n = delta.norm();
		if (n/damping < incrLimit) {
			stat = IterativeTermination::Converged;
			break;
		}

	}

	if (status != nullptr) {
		*status = stat;
	}

	return current;

}

std::optional<ShapePreservingTransform> initShapePreservingMapEstimate(Eigen::VectorXf const& obs,
																	   Eigen::Matrix3Xf const& pts,
																	   std::vector<int> const& idxs,
																	   std::vector<Axis> const& coordinate) {
	enum axisFlags {
		X = 1,
		Y = 2,
		Z = 4
	};

	std::vector<int> observed(pts.size());
	std::fill(observed.begin(), observed.end(), 0);

	int nobs = obs.rows();
	int nFullAxisPoints = 0;
	std::vector<int> fullAxisPointsIdxs;
	fullAxisPointsIdxs.reserve(pts.cols());

	for (int i = 0; i < nobs; i++) {
		axisFlags f = (coordinate[i] == Axis::X) ? X : (coordinate[i] == Axis::Y) ? Y : Z;
		observed[idxs[i]] = observed[idxs[i]] | f;

		if (observed[idxs[i]] == (X | Y | Z)) {
			auto p = std::find(fullAxisPointsIdxs.begin(), fullAxisPointsIdxs.end(), idxs[i]);
			if (p == fullAxisPointsIdxs.end()) {
				nFullAxisPoints++;
				fullAxisPointsIdxs.push_back(idxs[i]);
			}
		}
	}

	if (nFullAxisPoints < 3) {
		return std::nullopt;
	}

	int ptRefId = fullAxisPointsIdxs[0];
	int ptPivot1 = fullAxisPointsIdxs[1];
	int ptPivot2 = fullAxisPointsIdxs[2];

	Eigen::Vector3f ptRef_orig = pts.col(ptRefId);
	Eigen::Vector3f ptPiv1_orig = pts.col(ptPivot1);
	Eigen::Vector3f ptPiv2_orig = pts.col(ptPivot2);

	Eigen::Vector3f ptRef_obs;
	Eigen::Vector3f ptPiv1_obs;
	Eigen::Vector3f ptPiv2_obs;

	for (int i = 0; i < nobs; i++) {

		int rowId = (coordinate[i] == Axis::X) ? 0 : (coordinate[i] == Axis::Y) ? 1 : 2;

		if (idxs[i] == ptRefId) {
			ptRef_obs[rowId] = obs[i];
		} else if (idxs[i] == ptPivot1) {
			ptPiv1_obs[rowId] = obs[i];
		} else if (idxs[i] == ptPivot2) {
			ptPiv2_obs[rowId] = obs[i];
		}
	}

	ShapePreservingTransform translate(Eigen::Vector3f::Zero(), -ptRef_obs, 1);
	ShapePreservingTransform current(Eigen::Vector3f::Zero(), ptRef_obs - ptRef_orig, 1);

	Eigen::Vector3f trsfm = current*ptPiv1_orig - ptRef_obs;
	Eigen::Vector3f axis_alignement = ptPiv1_obs - ptRef_obs;
	axis_alignement.normalize();

	Eigen::Vector3f rot1 = (trsfm/trsfm.norm()).cross(axis_alignement);
	float scale = std::asin(rot1.norm())/rot1.norm();

	ShapePreservingTransform R1(scale*rot1, Eigen::Vector3f::Zero(), 1);

	trsfm = current*ptPiv2_orig - ptRef_obs;
	trsfm = R1*trsfm;

	Eigen::Vector3f axis_orientation = ptPiv2_obs - ptRef_obs;

	Eigen::Vector3f proj1 = trsfm - trsfm.dot(axis_alignement) * axis_alignement;
	Eigen::Vector3f proj2 = axis_orientation - axis_orientation.dot(axis_alignement) * axis_alignement;

	Eigen::Vector3f rot2 = (proj1/proj1.norm()).cross(proj2/proj2.norm());
	scale = std::asin(rot2.norm())/rot2.norm();

	ShapePreservingTransform R2(scale*rot2, Eigen::Vector3f::Zero(), 1);

	ShapePreservingTransform total = R2*R1*translate*current;

	Eigen::Matrix3Xf tpts = total*pts;

	Eigen::VectorXf obs_offst = obs;

	for (int i = 0; i < nobs; i++) {

		int rowId = (coordinate[i] == Axis::X) ? 0 : (coordinate[i] == Axis::Y) ? 1 : 2;

		obs_offst[i] -= ptRef_obs[rowId];
	}

	ShapePreservingTransform scaling = estimateScaleMap(obs_offst, tpts, idxs, coordinate, nullptr, false);

	return translate.inverse()*scaling*total;

}

} // namespace Geometry
}; //namespace StereoVision
