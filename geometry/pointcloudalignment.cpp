#include "pointcloudalignment.h"

#include "geometricexception.h"

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
