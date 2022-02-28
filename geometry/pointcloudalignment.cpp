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

	float s = (obs.array()/A.array()).mean();

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

	(void)(verbose);

	struct EqLine{
		EqLine() :
			ptIdx(-1),
			value(0) {

		}
		int ptIdx;
		float value;
	};

	struct Eq {
		EqLine axisX;
		EqLine axisY;
		EqLine axisZ;
	};

	int nobs = obs.rows();
	int npts = pts.cols();

	std::vector<Eq> eqs(npts);

	int nFreeParameters = 3*npts - nobs;

	int ParamPos = 3;

	for (int i = 0; i < npts; i++) {

		int xobs = -1;
		int yobs = -1;
		int zobs = -1;

		for (int j = 0; j < nobs; j++) {
			if (idxs[j] == i and coordinate[j] == Axis::X) {
				xobs = j;
			}
			if (idxs[j] == i and coordinate[j] == Axis::Y) {
				yobs = j;
			}
			if (idxs[j] == i and coordinate[j] == Axis::Z) {
				zobs = j;
			}
		}

		if (xobs >= 0) {
			eqs[i].axisX.value = obs[xobs];
			eqs[i].axisX.ptIdx = -1;
		} else {
			eqs[i].axisX.value = 0;
			eqs[i].axisX.ptIdx = ParamPos;
			ParamPos++;
		}

		if (yobs >= 0) {
			eqs[i].axisY.value = obs[yobs];
			eqs[i].axisY.ptIdx = -1;
		} else {
			eqs[i].axisY.value = 0;
			eqs[i].axisY.ptIdx = ParamPos;
			ParamPos++;
		}

		if (zobs >= 0) {
			eqs[i].axisZ.value = obs[zobs];
			eqs[i].axisZ.ptIdx = -1;
		} else {
			eqs[i].axisZ.value = 0;
			eqs[i].axisZ.ptIdx = ParamPos;
			ParamPos++;
		}
	}

	Eigen::MatrixXf A;

	Eigen::VectorXf l;
	l.setZero(3*npts);


	Eigen::VectorXf x;
	x.setZero(3 + nFreeParameters);


	IterativeTermination stat = IterativeTermination::MaxStepReached;

	for(int s = 0; s < n_steps; s++) {

		A.setZero(3*npts, 3 + nFreeParameters);

		for (int i = 0; i < npts; i++) {

			float valX;
			if (eqs[i].axisX.ptIdx < 0) {
				valX = eqs[i].axisX.value;
			} else {
				valX = x[eqs[i].axisX.ptIdx];
			}

			float valY;
			if (eqs[i].axisY.ptIdx < 0) {
				valY = eqs[i].axisY.value;
			} else {
				valY = x[eqs[i].axisY.ptIdx];
			}

			float valZ;
			if (eqs[i].axisZ.ptIdx < 0) {
				valZ = eqs[i].axisZ.value;
			} else {
				valZ = x[eqs[i].axisZ.ptIdx];
			}

			Eigen::Vector3f P = pts.col(i);
			Eigen::Vector3f O(valX, valY, valZ);

			float Pnorm = P.norm();
			float Onorm = O.norm();

			if (eqs[i].axisX.ptIdx < 0 and
					eqs[i].axisY.ptIdx < 0 and
					eqs[i].axisZ.ptIdx < 0) {

				//if all measurements are there, then an approximation of Theta can be computer as P/Pnorm cross O/Onorm

				if (s == 0) {
					Eigen::Vector3f Rot = (P/Pnorm).cross(O/Onorm);
					l[3*i] = Rot[0];
					l[3*i+1] = Rot[1];
					l[3*i+2] = Rot[2];
				}

				A(3*i,0) = 1;
				A(3*i+1,1) = 1;
				A(3*i+2,2) = 1;

			} else {
				//Else the equation is Onorm*Theta_x - (1/Pnorm)(P_y O_z - P_z O_y) = 0
				//and Onorm*Theta_y - (1/Pnorm)(P_z O_x - P_x O_z) = 0
				//and Onorm*Theta_z - (1/Pnorm)(P_x O_y - P_y O_x) = 0

				A(3*i,0) = Onorm;
				A(3*i+1,1) = Onorm;
				A(3*i+2,2) = Onorm;

				if (eqs[i].axisX.ptIdx >= 0) {
					A(3*i,eqs[i].axisX.ptIdx) = O[0]/Onorm * x[0];
					A(3*i+1,eqs[i].axisX.ptIdx) = -P[2]/Pnorm;
					A(3*i+2,eqs[i].axisX.ptIdx) = P[1]/Pnorm;
				}

				if (eqs[i].axisY.ptIdx >= 0) {
					A(3*i,eqs[i].axisY.ptIdx) = P[2]/Pnorm;
					A(3*i+1,eqs[i].axisY.ptIdx) = O[1]/Onorm * x[1];
					A(3*i+2,eqs[i].axisY.ptIdx) = -P[0]/Pnorm;
				}

				if (eqs[i].axisZ.ptIdx >= 0) {
					A(3*i,eqs[i].axisZ.ptIdx) = -P[1]/Pnorm;
					A(3*i+1,eqs[i].axisZ.ptIdx) = P[0]/Pnorm;
					A(3*i+2,eqs[i].axisZ.ptIdx) = O[2]/Onorm * x[2];
				}

			}

		}

		Eigen::MatrixXf invQxx = A.transpose()*A;

		auto svd = invQxx.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

		Eigen::MatrixXf pseudoInverse = svd.matrixV() * (svd.singularValues().array().abs() > 1e-6).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().transpose();

		Eigen::VectorXf r = l - A*x;
		Eigen::VectorXf delta = pseudoInverse*A.transpose()*r;

		x = x + delta;

		if (delta.norm()/(3 + nFreeParameters) < incrLimit) {
			stat = IterativeTermination::Converged;
			break;
		}

	}

	if (status != nullptr) {
		*status = stat;
	}

	Eigen::Vector3f theta = x.block<3,1>(0,0);
	theta *= std::asin(theta.norm())/theta.norm();

	ShapePreservingTransform r(theta, Eigen::Vector3f::Zero(), 1.);

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

AffineTransform estimateShapePreservingMap(Eigen::VectorXf const& obs,
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

	AffineTransform initial = estimateQuasiShapePreservingMap(obs,
															  pts,
															  idxs,
															  coordinate,
															  2e-1,
															  nullptr,
															  1e-6,
															  500);

	ShapePreservingTransform trans = affine2ShapePreservingMap(initial);

	if (trans.s < 0) {
		throw GeometricException("Unable to estimate rigid transformation starting from mirrored transformation.");
	}

	float s = std::log(trans.s);
	Eigen::Vector3f r = trans.r;
	Eigen::Vector3f t = trans.t;

	AffineTransform transform;

	int n_obs = obs.rows();

	if (n_obs < 7) {//underdetermined
		if (status != nullptr) {
			*status = IterativeTermination::Error;
		}
		return transform;
	}

	ParamVector x;
	x << r, t, s;

	*status = IterativeTermination::MaxStepReached;

	Eigen::Matrix<float, Eigen::Dynamic, 7, Eigen::RowMajor> A;
	A.resize(n_obs, 7);

	Eigen::VectorXf e = obs;

	for(int i = 0; i < n_steps; i++) {

		r << x[0],x[1],x[2];
		t << x[3],x[4],x[5];
		s = x[6];

		if (fabs(s) > 5) {//limit the scale growth to avoid scale explosion
			s = (s > 0) ? 3. : -3;
			x[6] = s;
		}

		Eigen::Matrix3f R = rodriguezFormula(r);
		Eigen::Matrix3f DxR = diffRodriguez(r, Axis::X);
		Eigen::Matrix3f DyR = diffRodriguez(r, Axis::Y);
		Eigen::Matrix3f DzR = diffRodriguez(r, Axis::Z);

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
			Eigen::Block<Eigen::Matrix3f,1,3> Rrow = R.block<1,3>(id_row,0);

			float l0i = exp(s)*(Rrow*pts.block<3,1>(0,idxs[i]))(0,0) + t[id_row];
			e[i] = obs[i] - l0i;

			A(i, 0) = exp(s)*(DxR.block<1,3>(id_row,0)*pts.block<3,1>(0,idxs[i]))(0,0); //Param rx;
			A(i, 1) = exp(s)*(DyR.block<1,3>(id_row,0)*pts.block<3,1>(0,idxs[i]))(0,0); //Param ry;
			A(i, 2) = exp(s)*(DzR.block<1,3>(id_row,0)*pts.block<3,1>(0,idxs[i]))(0,0); //Param rz;

			A(i, 3) = (coordinate[i] == Axis::X) ? 1 : 0; //Param x;
			A(i, 4) = (coordinate[i] == Axis::Y) ? 1 : 0; //Param y;
			A(i, 5) = (coordinate[i] == Axis::Z) ? 1 : 0; //Param z;

			A(i, 6) = exp(s)*(Rrow*pts.block<3,1>(0,idxs[i]))(0,0); //Param s;
		}

		MatrixQxx invQxx = A.transpose()*A;
		auto QR = Eigen::ColPivHouseholderQR<MatrixQxx>(invQxx);

		if (!QR.isInvertible()) {
			if (status != nullptr) {
				*status = IterativeTermination::Error;
			}
			return transform;
		}

		ParamVector incr = QR.solve(A.transpose()*e);
		incr[6] *= dampingScale;

		x = x + damping*incr;

		float n = incr.norm();
		if (n < incrLimit) {
			if (status != nullptr) {
				*status = IterativeTermination::Converged;
			}
			break;
		}

	}

	r << x[0],x[1],x[2];
	t << x[3],x[4],x[5];
	s = x[6];

	transform.R = exp(s)*rodriguezFormula(r);
	transform.t = t;

	return transform;

}

} // namespace Geometry
}; //namespace StereoVision
