#ifndef LIBSTEVI_SHAPEFROMSHADING_H
#define LIBSTEVI_SHAPEFROMSHADING_H

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

namespace StereoVision {
namespace ImageProcessing {

template<typename ComputeType>
Multidim::Array<ComputeType, 3> normalMapFromSingleShadingImage(Multidim::Array<ComputeType, 2> const& shading,
																Eigen::Matrix<ComputeType, 3, 1> const& lightDirection,
																ComputeType lambdaNorm = 1.0,
																ComputeType lambdaDiff = 0.25,
																int nIter = 50,
																ComputeType incrTol = 1e-6) {

	using MatrixAType = Eigen::SparseMatrix<ComputeType>;
	using VectorBType = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;
	using VectorSolType = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;

	using SolverType = Eigen::ConjugateGradient<MatrixAType, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<ComputeType>>;
	//using SolverType = Eigen::BiCGSTAB<MatrixAType, Eigen::IncompleteLUT<ComputeType>>;

	int VectorPlen = shading.flatLenght();
	int VectorNlen = 3*VectorPlen;

	if (VectorPlen <= 0) {
		return Multidim::Array<ComputeType, 3>();
	}

	Eigen::Matrix<ComputeType, 3, 1> normalizedLightDirection = lightDirection;
	ComputeType lightDirNorm = normalizedLightDirection.norm();

	ComputeType maxShadingVal = shading.valueUnchecked(0,0);

	std::array<int,2> inSize = shading.shape();
	std::array<int,3> outSize;
	outSize[0] = inSize[0];
	outSize[1] = inSize[1];
	outSize[2] = 2;

	for (int i = 0; i < inSize[0]; i++) {
		for (int j = 0; j < inSize[1]; j++) {

			ComputeType val = shading.valueUnchecked(i,j);

			if (val > maxShadingVal) {
				maxShadingVal = val;
			}

		}
	}

	lightDirNorm *= maxShadingVal/lightDirNorm;

	std::array<int,3> fullOutSize = outSize;
	fullOutSize[2] = 3;

	Multidim::IndexConverter<2> idxConverterIn(inSize);
	Multidim::IndexConverter<3> idxConverterOut(outSize);

	MatrixAType P;
	VectorBType p = VectorBType::Zero(VectorPlen);

	P.resize(VectorPlen, VectorNlen);
	P.reserve(Eigen::VectorXi::Constant(VectorNlen, 1));

	for (int j = 0; j < idxConverterOut.numberOfPossibleIndices(); j++) {
		auto idxOut = idxConverterOut.getIndexFromPseudoFlatId(j);

		std::array<int,2> idxIn = {idxOut[0], idxOut[1]};

		int i = idxConverterIn.getPseudoFlatIdFromIndex(idxIn);

		ComputeType PVal = normalizedLightDirection[idxOut[2]];

		P.coeffRef(i,j) -= PVal;

		ComputeType pVal = shading.valueUnchecked(idxIn);

		p[i] = pVal;
	}

	MatrixAType Dx;
	Dx.resize(VectorNlen, VectorNlen);
	Dx.reserve(Eigen::VectorXi::Constant(VectorNlen, 2));

	MatrixAType Dy;
	Dy.resize(VectorNlen, VectorNlen);
	Dy.reserve(Eigen::VectorXi::Constant(VectorNlen, 2));

	for (int i = 0; i < idxConverterOut.numberOfPossibleIndices(); i++) {
		auto idxOut = idxConverterOut.getIndexFromPseudoFlatId(i);

		auto idxOutM1 = idxOut;
		auto idxOutP1 = idxOut;

		if (idxOut[0]-1 >= 0 or idxOut[0]+1 < outSize[0]) {
			if (idxOut[0]-1 >= 0 and idxOut[0]+1 < outSize[0]) {
				idxOutM1[0] -= 1;
				idxOutP1[0] += 1;
			} else if (idxOut[0]-1 >= 0) {
				idxOutM1[0] -= 1;
			} else if (idxOut[0]+1 < outSize[0]) {
				idxOutP1[0] += 1;
			}

			int j = idxConverterOut.getPseudoFlatIdFromIndex(idxOutM1);
			Dx.coeffRef(i,j) -= 1;

			j = idxConverterOut.getPseudoFlatIdFromIndex(idxOutP1);
			Dx.coeffRef(i,j) += 1;
		}

		idxOutM1 = idxOut;
		idxOutP1 = idxOut;

		if (idxOut[1]-1 >= 0 or idxOut[1]+1 < outSize[1]) {
			if (idxOut[1]-1 >= 0 and idxOut[1]+1 < outSize[1]) {
				idxOutM1[1] -= 1;
				idxOutP1[1] += 1;
			} else if (idxOut[1]-1 >= 0) {
				idxOutM1[1] -= 1;
			} else if (idxOut[1]+1 < outSize[1]) {
				idxOutP1[1] += 1;
			}

			int j = idxConverterOut.getPseudoFlatIdFromIndex(idxOutM1);
			Dx.coeffRef(i,j) -= 1;

			j = idxConverterOut.getPseudoFlatIdFromIndex(idxOutP1);
			Dx.coeffRef(i,j) += 1;
		}
	}

	MatrixAType Abase = P.transpose()*P;
	Abase += lambdaDiff*(Dx.transpose()*Dx);
	Abase += lambdaDiff*(Dy.transpose()*Dy);

	VectorSolType solution = VectorSolType::Zero(VectorNlen);

	MatrixAType N;
	N.resize(VectorNlen, VectorNlen);
	N.reserve(Eigen::VectorXi::Constant(VectorNlen, 3));

	VectorBType normDiffs = VectorSolType::Zero(VectorNlen);

	for (int iter = 0; iter < nIter; iter++) {

		for (int i = 0; i < VectorPlen; i++) {
			int i1 = 3*i;
			int i2 = 3*i+1;
			int i3 = 3*i+2;

			ComputeType x1 = solution[i1];
			ComputeType x2 = solution[i2];
			ComputeType x3 = solution[i3];

			ComputeType quadr = (x1*x1 + x2*x2 + x3*x3 - 1);

			normDiffs[i1] = 4*quadr*x1;
			normDiffs[i2] = 4*quadr*x2;
			normDiffs[i3] = 4*quadr*x3;

			N.coeffRef(i1,i1) = 4*quadr + 8*x1;
			N.coeffRef(i1,i2) = 8*x1*x2;
			N.coeffRef(i1,i3) = 8*x1*x3;

			N.coeffRef(i2,i2) = 4*quadr + 8*x2;
			N.coeffRef(i2,i1) = 8*x2*x1;
			N.coeffRef(i2,i3) = 8*x2*x3;

			N.coeffRef(i3,i3) = 4*quadr + 8*x3;
			N.coeffRef(i3,i1) = 8*x3*x1;
			N.coeffRef(i3,i2) = 8*x3*x2;
		}

		MatrixAType A = Abase;
		VectorBType b = P.transpose()*p - A*solution - lambdaNorm*normDiffs; //rhs = c - (A*x_0 + g(x_0))
		A += lambdaNorm*N; //Diff = A + Diff(g)

		SolverType solver;
		solver.compute(A);

		VectorSolType delta = solver.solve(b);

		if(solver.info()!=Eigen::Success) {
			return Multidim::Array<ComputeType, 3>();
		}

		solution += delta;

		if (delta.norm()/VectorNlen < incrTol) {
			break;
		}
	}

	Multidim::Array<ComputeType, 3> ret(outSize);

	for (int i = 0; i < idxConverterOut.numberOfPossibleIndices(); i++) {
		auto idxOut = idxConverterOut.getIndexFromPseudoFlatId(i);

		ret.atUnchecked(idxOut) = solution[i];
	}

	for (int i = 0; i < outSize[0]; i++) {
		for (int j = 0; j < outSize[1]; j++) {
			ComputeType norm = 0;

			for (int c = 0; c < 3; c++) {
				ComputeType v = ret.valueUnchecked(i,j,c);
				norm += v*v;
			}

			norm = std::sqrt(norm);

			for (int c = 0; c < 3; c++) {
				ret.atUnchecked(i,j,c) /= norm;
			}
		}
	}

	return ret;
}

} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_SHAPEFROMSHADING_H
