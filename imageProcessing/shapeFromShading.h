#ifndef LIBSTEVI_SHAPEFROMSHADING_H
#define LIBSTEVI_SHAPEFROMSHADING_H

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

#include "./convolutions.h"

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

	//using SolverType = Eigen::ConjugateGradient<MatrixAType, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<ComputeType>>;
	//using SolverType = Eigen::BiCGSTAB<MatrixAType, Eigen::IncompleteLUT<ComputeType>>;
	using SolverType = Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >;

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

/*!
 * \brief normalMapFromIntrinsicDecomposition compute a normal map using the intrinsic image decompisition
 * \param shading the shading image
 * \param reflectance
 * \param lambdaNorm
 * \param lambdaDiff
 * \param nIter
 * \param incrTol
 * \return
 */
template<typename ComputeType>
Multidim::Array<ComputeType, 3> normalMapFromIntrinsicDecomposition(Multidim::Array<ComputeType, 2> const& shading,
                                                                    Multidim::Array<ComputeType, 3> const& reflectance,
                                                                    Eigen::Matrix<ComputeType, 3, 1> const& lightDirection,
                                                                    ComputeType lambdaNorm = 1.0,
                                                                    ComputeType lambdaDiff = 0.25,
                                                                    ComputeType lambdaDir = 0.25,
                                                                    ComputeType propEdges = 0.05,
                                                                    int nIter = 50,
                                                                    ComputeType incrTol = 1e-6) {

    using MatrixAType = Eigen::SparseMatrix<ComputeType>;
    using VectorBType = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;
    using VectorSolType = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;

    //using SolverType = Eigen::ConjugateGradient<MatrixAType, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<ComputeType>>;
    //using SolverType = Eigen::BiCGSTAB<MatrixAType, Eigen::IncompleteLUT<ComputeType>>;
    using SolverType = Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >;

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

    normalizedLightDirection *= lightDirNorm;

    std::array<int,3> fullOutSize = outSize;
    fullOutSize[2] = 3;

    Multidim::IndexConverter<2> idxConverterIn(inSize);
    Multidim::IndexConverter<3> idxConverterOut(outSize);

    MatrixAType P;
    VectorBType p = VectorBType::Zero(VectorPlen);

    //Constraint S = <D,N>
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

    //Constraint Gradiant N = 0
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

    //Constraint H(||\nabla R||-c)\dotprod{S\nabla R}{N} = 0

    constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

    constexpr int nGradientDir = 2;
    Multidim::Array<ComputeType,3> coefficients(3,3,nGradientDir);

    coefficients.atUnchecked(0,0,0) = 1;
    coefficients.atUnchecked(1,0,0) = 2;
    coefficients.atUnchecked(2,0,0) = 1;

    coefficients.atUnchecked(0,1,0) = 0;
    coefficients.atUnchecked(1,1,0) = 0;
    coefficients.atUnchecked(2,1,0) = 0;

    coefficients.atUnchecked(0,2,0) = -1;
    coefficients.atUnchecked(1,2,0) = -2;
    coefficients.atUnchecked(2,2,0) = -1;

    coefficients.atUnchecked(0,0,1) = 1;
    coefficients.atUnchecked(0,1,1) = 2;
    coefficients.atUnchecked(0,2,1) = 1;

    coefficients.atUnchecked(1,0,1) = 0;
    coefficients.atUnchecked(1,1,1) = 0;
    coefficients.atUnchecked(1,2,1) = 0;

    coefficients.atUnchecked(2,0,1) = -1;
    coefficients.atUnchecked(2,1,1) = -2;
    coefficients.atUnchecked(2,2,1) = -1;

    using Maxis = Convolution::MovingWindowAxis;
    using BIaxis = Convolution::BatchedInputAxis;
    using BOaxis = Convolution::BatchedOutputAxis;

    Convolution::Filter<ComputeType, Maxis, Maxis, BIaxis, BOaxis> gradientFilter(coefficients, Maxis(), Maxis(), BIaxis(), BOaxis());

    Multidim::Array<ComputeType, 4> gradientsPerChannels = gradientFilter.convolve(reflectance);

    Multidim::Array<ComputeType, 3> aggregatedGradients(inSize[0], inSize[1], nGradientDir);
    Multidim::Array<ComputeType, 2> gradientsAmplitude(inSize[0], inSize[1]);

    std::vector<ComputeType> sortedGradientsAmpl;
    sortedGradientsAmpl.reserve(inSize[0]*inSize[1]);

    ComputeType maxAmpl = 0;

    for (int i = 0; i < inSize[0]; i++) {
        for (int j = 0; j < inSize[1]; j++) {

            ComputeType d0 = 0;
            ComputeType d1 = 0;

            for (int c = 0; c < inSize[2]; c++) {

                ComputeType coeff = 1;

                if (gradientsPerChannels.atUnchecked(i,j,c,0) < 0) {
                    coeff = -1;
                }

                d0 += coeff*gradientsPerChannels.atUnchecked(i,j,c,0);
                d1 += coeff*gradientsPerChannels.atUnchecked(i,j,c,1);
            }

            aggregatedGradients.atUnchecked(i,j,0) = d0;
            aggregatedGradients.atUnchecked(i,j,1) = d1;

            ComputeType ampl = d0*d0 + d1*d1;

            gradientsAmplitude.atUnchecked(i,j) = ampl;
            sortedGradientsAmpl.push_back(ampl);

            maxAmpl = std::max(ampl, maxAmpl);

        }
    }

    maxAmpl = std::sqrt(maxAmpl);

    int qant = (sortedGradientsAmpl.size()-1)*(1.-propEdges);

    if (qant < 0) {
        qant = 0;
    }

    if (qant >= sortedGradientsAmpl.size()) {
        qant = sortedGradientsAmpl.size()-1;
    }

    std::nth_element(sortedGradientsAmpl.begin(), sortedGradientsAmpl.begin()+qant, sortedGradientsAmpl.end());

    ComputeType threshold = sortedGradientsAmpl[qant];

    MatrixAType D;

    D.resize(VectorPlen, VectorNlen);
    D.reserve(Eigen::VectorXi::Constant(VectorNlen, 1));

    for (int j = 0; j < idxConverterOut.numberOfPossibleIndices(); j++) {
        auto idxOut = idxConverterOut.getIndexFromPseudoFlatId(j);

        if (gradientsAmplitude.atUnchecked(idxOut[0], idxOut[1]) < threshold) {
            continue;
        }

        std::array<int,2> idxIn = {idxOut[0], idxOut[1]};

        int i = idxConverterIn.getPseudoFlatIdFromIndex(idxIn);

        ComputeType DVal = 0;

        if (idxOut[2] < 2) {

            int idx = idxOut[2] == 1 ? 0 : 1;
            int scale = idxOut[2] == 1 ? -1 : 1;
            ComputeType d = aggregatedGradients.atUnchecked(idxOut[0], idxOut[1], idx);

            DVal = scale*d/maxAmpl;
        }

        D.coeffRef(i,j) -= DVal;
    }

    //Constraint |N| = 1 and iterations
    MatrixAType Abase = P.transpose()*P;
    Abase += lambdaDiff*(Dx.transpose()*Dx);
    Abase += lambdaDiff*(Dy.transpose()*Dy);

    ComputeType rescale = static_cast<ComputeType>(sortedGradientsAmpl.size())/(sortedGradientsAmpl.size() - qant);
    Abase += rescale*lambdaDir*(D.transpose()*D);

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

    //return normal map
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

template<typename ComputeType>
Multidim::Array<ComputeType, 2> heightFromNormalMap(Multidim::Array<ComputeType, 3> const& normalmap, ComputeType maxDiff = 50.) {


	using MatrixAType = Eigen::SparseMatrix<ComputeType>;
	using VectorBType = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;
	using VectorSolType = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;

	using SolverType = Eigen::LeastSquaresConjugateGradient<MatrixAType, Eigen::LeastSquareDiagonalPreconditioner<ComputeType>>;

	if (normalmap.shape()[2] != 3) {
		return Multidim::Array<ComputeType, 2>();
	}

	std::array<int,2> shape{normalmap.shape()[0], normalmap.shape()[1]};

	Multidim::Array<ComputeType, 3> diffMap(normalmap.shape()[0], normalmap.shape()[1], 2);

	for (int i = 0; i < normalmap.shape()[0]; i++) {
		for (int j = 0; j < normalmap.shape()[1]; j++) {
			ComputeType nx = normalmap.valueUnchecked(i,j,0);
			ComputeType ny = normalmap.valueUnchecked(i,j,1);
			ComputeType nz = normalmap.valueUnchecked(i,j,2);

			ComputeType sx = (nx > 0) ? 1 : (sx < 0) ? -1 : 0;
			ComputeType sy = (ny > 0) ? 1 : (sx < 0) ? -1 : 0;

			ComputeType dx = nx/nz;

			if (std::isnan(dx) or std::isinf(dx) or std::abs(dx) > maxDiff) {
				dx = sx*maxDiff;
			}

			ComputeType dy = ny/nz;

			if (std::isnan(dy) or std::isinf(dy) or std::abs(dy) > maxDiff) {
				dy = sy*maxDiff;
			}
		}
	}

	Multidim::IndexConverter<2> outIdxs(shape);
	Multidim::IndexConverter<3> inIdxs(diffMap.shape());

	int nObs = inIdxs.numberOfPossibleIndices() + 1;
	int nVars = outIdxs.numberOfPossibleIndices();

	MatrixAType A;

	A.resize(nObs, nVars);
	A.reserve(Eigen::VectorXi::Constant(nVars, 5));

	VectorBType b;
	b.resize(nObs);

	for (int i = 0; i < inIdxs.numberOfPossibleIndices(); i++) {

		std::array<int,3> idxIn = inIdxs.getIndexFromPseudoFlatId(i);
		std::array<int,2> outCenterIdx = {idxIn[0], idxIn[1]};

		if (idxIn[2] == 0) { //xDiff
			int xPosM1 = (outCenterIdx[1] > 0) ? outCenterIdx[1]-1 : 0;
			int xPosP1 = (outCenterIdx[1] < shape[1]-1) ? outCenterIdx[1]+1 : shape[1]-1;

			int jM1 = outIdxs.getPseudoFlatIdFromIndex({outCenterIdx[0], xPosM1});
			int jP1 = outIdxs.getPseudoFlatIdFromIndex({outCenterIdx[0], xPosP1});

			A.coeffRef(i,jM1) = -1;
			A.coeffRef(i,jP1) = 1;
		} else if (idxIn[2] == 1) { //yDiff
			int yPosM1 = (outCenterIdx[0] < shape[0]-1) ? outCenterIdx[0]+1 : shape[1]-1;
			int yPosP1 = (outCenterIdx[0] > 0) ? outCenterIdx[0]-1 : 0;

			int jM1 = outIdxs.getPseudoFlatIdFromIndex({yPosM1, outCenterIdx[1]});
			int jP1 = outIdxs.getPseudoFlatIdFromIndex({yPosP1, outCenterIdx[1]});

			A.coeffRef(i,jM1) = -1;
			A.coeffRef(i,jP1) = 1;
		}

		b[i] = diffMap.valueUnchecked(idxIn);

	}

	int zeroPosIdx = outIdxs.getPseudoFlatIdFromIndex({0,0});

	A.coeffRef(nObs-1,zeroPosIdx) = 1;
	b[nObs-1] = 0;

	A.makeCompressed();

	SolverType solver;
	solver.compute(A);

	VectorSolType solution = solver.solve(b);


	Multidim::Array<ComputeType, 2> ret(shape);

	for (int i = 0; i < outIdxs.numberOfPossibleIndices(); i++) {
		auto idxOut = outIdxs.getIndexFromPseudoFlatId(i);

		ret.atUnchecked(idxOut) = solution[i];
	}

	return ret;
}

} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_SHAPEFROMSHADING_H
