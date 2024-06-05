#ifndef LIBSTEVI_SHAPEFROMSHADING_H
#define LIBSTEVI_SHAPEFROMSHADING_H

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <Eigen/QR>

#include "./convolutions.h"
#include "./edgesDetection.h"

#include "../geometry/rotations.h"

namespace StereoVision {
namespace ImageProcessing {

/*!
 * \brief initialNormalMapEstimate compute a rough estimate of the normal map given the shading.
 * \param shading The shading map
 * \param lightDirection The incoming light direction
 * \return an estimate of the normal map
 *
 * This function lift the ambiguity in the normal estimation by assuming that the normal is colinear with the gradient.
 *
 * This yield three equations for the three unknowns (unit length, light dot normal = shading and colinear with gradient).
 */
template<typename ComputeType>
Multidim::Array<ComputeType, 3> initialNormalMapEstimate(Multidim::Array<ComputeType, 2> const& shading,
                                                         Eigen::Matrix<ComputeType, 3, 1> const& lightDirection) {

    constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

    ComputeType check = 1/lightDirection.z();
    if (std::isinf(check) or std::isnan(check)) {
        return Multidim::Array<ComputeType, 3>();
    }

    std::array<int,2> inShape = shading.shape();

    Multidim::Array<ComputeType, 4> possibleEstimates(inShape[0], inShape[1], 3, 2);

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
    using BOaxis = Convolution::BatchedOutputAxis;

    Convolution::Filter<ComputeType, Maxis, Maxis, BOaxis> gradientFilter(coefficients, Maxis(), Maxis(), BOaxis());

	//compute the gradients
    Multidim::Array<ComputeType, 3> gradients = gradientFilter.convolve(shading);

    ComputeType maxShading = shading.valueUnchecked(0,0);

    for (int i = 0; i < inShape[0]; i++) {
        for (int j = 0; j < inShape[1]; j++) {

            ComputeType val = shading.valueUnchecked(i,j);

            if (val > maxShading) {
                maxShading = val;
            }
        }
    }

    Eigen::Matrix<ComputeType, 3, 1> ld = lightDirection;
    ld.normalize();
    ld *= maxShading;

    for (int i = 0; i < inShape[0]; i++) {
        for (int j = 0; j < inShape[1]; j++) {

            ComputeType gx = gradients.atUnchecked(i,j,0);
            ComputeType gy = gradients.atUnchecked(i,j,1);

            ComputeType s = shading.valueUnchecked(i,j);

            ComputeType x;
            ComputeType y;
            ComputeType z;

            ComputeType scale;

            //branch x = f(y), d = x
            if (std::abs(gx) < std::abs(gy)) {
                scale = gx/gy; //x = scale*y;
            } else { //branch y = f(x)
                scale = gy/gx; //y = scale*x;
            }

            if (std::isinf(scale) or std::isnan(scale)) {
                scale = 1;
            }

            ComputeType lv;

            if (std::abs(gx) < std::abs(gy)) {
                lv = scale*ld.x() + ld.y();
            } else {
                lv = scale*ld.y() + ld.x();
            }

            ComputeType lz = ld.z();

            ComputeType tr = -lv/lz; //z = s + tr*v

            ComputeType a = tr*tr + scale*scale + 1; //a*v*v + b*v + c = 0
            ComputeType b = 2*tr*s;
            ComputeType c = s*s - 1;

            ComputeType delta = b*b - 4*a*c;

            if (std::abs(gx) < std::abs(gy)) {
                y = (-b + sqrt(delta))/(2*a);
                z = tr*y + s;
                x = scale*y;
            } else {
                x = (-b + sqrt(delta))/(2*a);
                z = tr*x + s;
                y = scale*x;
            }

            possibleEstimates.atUnchecked(i,j,0,0) = x;
            possibleEstimates.atUnchecked(i,j,1,0) = y;
            possibleEstimates.atUnchecked(i,j,2,0) = z;

            if (std::abs(gx) < std::abs(gy)) {
                y = (-b - sqrt(delta))/(2*a);
                z = tr*y + s;
                x = scale*y;
            } else {
                x = (-b - sqrt(delta))/(2*a);
                z = tr*x + s;
                y = scale*x;
            }

            possibleEstimates.atUnchecked(i,j,0,1) = x;
            possibleEstimates.atUnchecked(i,j,1,1) = y;
            possibleEstimates.atUnchecked(i,j,2,1) = z;

        }
    }

    Multidim::Array<ComputeType, 3> estimates(inShape[0], inShape[1], 3);

    for (int i = 0; i < inShape[0]; i++) {
        for (int j = 0; j < inShape[1]; j++) {

            std::array<ComputeType, 2> deltas = {0, 0};

            for (int d = 0; d < 2; d++) {
                ComputeType x = possibleEstimates.atUnchecked(i,j,0,d);
                ComputeType y = possibleEstimates.atUnchecked(i,j,1,d);
                ComputeType z = possibleEstimates.atUnchecked(i,j,2,d);

                if (i > 0) {
                    ComputeType dx = estimates.atUnchecked(i-1,j,0) - x;
                    ComputeType dy = estimates.atUnchecked(i-1,j,1) - y;
                    ComputeType dz = estimates.atUnchecked(i-1,j,2) - z;

                    deltas[d] += std::sqrt(dx*dx + dy*dy + dz*dz);
                }

                if (j > 0) {
                    ComputeType dx = estimates.atUnchecked(i,j-1,0) - x;
                    ComputeType dy = estimates.atUnchecked(i,j-1,1) - y;
                    ComputeType dz = estimates.atUnchecked(i,j-1,2) - z;

                    deltas[d] += std::sqrt(dx*dx + dy*dy + dz*dz);
                }
            }

            int selected = 0;
            if (deltas[1] < deltas[0]) {
                selected = 1;
            }

			if (possibleEstimates.atUnchecked(i,j,2,selected) < 0) {
				selected = (selected == 0) ? 1 : 0;
			}

            for (int a = 0; a < 3; a++) {
                estimates.atUnchecked(i,j,a) = possibleEstimates.atUnchecked(i,j,a,selected);
            }

        }
    }

    return estimates;

}

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
 * \param guide the guide used to detect edges, where it is assumed the normal align with the gradient
 * \param lambdaNorm the coefficient weight for enforcing the normal to be unit length
 * \param lambdaDiff the coefficient weight for smoothing the solution
 * \param nIter the number of iterations for non linear optimization
 * \param incrTol The tolerance threshold on the solution increment to exit the non linear optimization loop.
 * \return The estimated normal map.
 */
template<typename ComputeType>
Multidim::Array<ComputeType, 3> normalMapFromIntrinsicDecomposition(Multidim::Array<ComputeType, 2> const& shading,
																	Multidim::Array<ComputeType, 3> const& guide,
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

    using SolverType = Eigen::ConjugateGradient<MatrixAType, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<ComputeType>>;
    //using SolverType = Eigen::BiCGSTAB<MatrixAType, Eigen::IncompleteLUT<ComputeType>>;
    //using SolverType = Eigen::SparseLU<Eigen::SparseMatrix<ComputeType>, Eigen::COLAMDOrdering<int> >;
    //using SolverType = Eigen::SimplicialLDLT<Eigen::SparseMatrix<ComputeType>>;

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
    outSize[2] = 3;

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

        P.coeffRef(i,j) += PVal;

        ComputeType pVal = shading.valueUnchecked(idxIn);

        p[i] = pVal;
    }

	//Constraint Gradient N = 0
    MatrixAType Dx;
    Dx.resize(2*VectorNlen, VectorNlen);
    Dx.reserve(Eigen::VectorXi::Constant(VectorNlen, 4));

    MatrixAType Dy;
    Dy.resize(2*VectorNlen, VectorNlen);
    Dy.reserve(Eigen::VectorXi::Constant(VectorNlen, 4));

    for (int i = 0; i < idxConverterOut.numberOfPossibleIndices(); i++) {
        auto idxOut = idxConverterOut.getIndexFromPseudoFlatId(i);

        auto idxOutM1 = idxOut;
        auto idxOutP1 = idxOut;

        if (idxOut[0]-1 >= 0) {
            idxOutM1[0] -= 1;

            int j = idxConverterOut.getPseudoFlatIdFromIndex(idxOutM1);
            Dx.coeffRef(i,j) -= 1;

            j = idxConverterOut.getPseudoFlatIdFromIndex(idxOut);
            Dx.coeffRef(i,j) += 1;
        }

        if (idxOut[0]+1 < outSize[0]) {
            idxOutP1[0] += 1;

            int j = idxConverterOut.getPseudoFlatIdFromIndex(idxOut);
            Dx.coeffRef(i+VectorNlen,j) -= 1;

            j = idxConverterOut.getPseudoFlatIdFromIndex(idxOutP1);
            Dx.coeffRef(i+VectorNlen,j) += 1;
        }

        idxOutM1 = idxOut;
        idxOutP1 = idxOut;

        if (idxOut[1]-1 >= 0) {
            idxOutM1[1] -= 1;

            int j = idxConverterOut.getPseudoFlatIdFromIndex(idxOutM1);
            Dy.coeffRef(i,j) -= 1;

            j = idxConverterOut.getPseudoFlatIdFromIndex(idxOut);
            Dy.coeffRef(i,j) += 1;
        }

        if (idxOut[1]+1 < outSize[1]) {
            idxOutP1[1] += 1;

            int j = idxConverterOut.getPseudoFlatIdFromIndex(idxOut);
            Dy.coeffRef(i+VectorNlen,j) -= 1;

            j = idxConverterOut.getPseudoFlatIdFromIndex(idxOutP1);
            Dy.coeffRef(i+VectorNlen,j) += 1;
        }
    }

    //Constraint H(||\nabla R||-c)\dotprod{S\nabla R}{N} = 0

	std::vector<std::tuple<std::array<int,2>, std::array<ComputeType,2>>> coords = gradientBasedEdges(guide, propEdges);

    MatrixAType D;

    D.resize(VectorPlen, VectorNlen);
    D.reserve(Eigen::VectorXi::Constant(VectorNlen, 1));

    for (auto const& [idxIn, gradient] : coords) {

        int i = idxConverterIn.getPseudoFlatIdFromIndex(idxIn);

        ComputeType ampl = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);

        for (int c = 0; c < 2; c++) {
            std::array<int,3> idxOut = {idxIn[0], idxIn[1], c};
            int j = idxConverterOut.getPseudoFlatIdFromIndex(idxOut);


            ComputeType DVal = 0;

            //turn the gradient 90°, so that cross product with normal is 0!
            int idx = idxOut[2] == 1 ? 1 : 0;
            int scale = idxOut[2] == 1 ? 1 : 1;

            ComputeType d = gradient[idx];

            DVal = scale*d/ampl;

            D.coeffRef(i,j) -= DVal;
        }
    }

    //Constraint |N| = 1 and iterations
    MatrixAType Abase = P.transpose()*P;
    Abase += lambdaDiff*(Dx.transpose()*Dx);
    Abase += lambdaDiff*(Dy.transpose()*Dy);

    ComputeType rescale = static_cast<ComputeType>(inSize[0]*inSize[1])/coords.size();
    Abase += rescale*lambdaDir*(D.transpose()*D);

    VectorSolType solution = VectorSolType::Zero(VectorNlen);
	std::cout << "Base matrix computed" << std::endl;

	//Initialize the solution with a guess
	Multidim::Array<ComputeType,3> guess = initialNormalMapEstimate(shading, normalizedLightDirection);

	for (int i = 0; i < idxConverterOut.numberOfPossibleIndices(); i++) {
		auto idxOut = idxConverterOut.getIndexFromPseudoFlatId(i);

        ComputeType val = guess.atUnchecked(idxOut);

        if (!std::isfinite(val)) {
            val = (idxOut[2] == 2) ? 1 : 0; //if the guess failed initialize with the normal pointing up
        }

        solution[i] = val;
    }

    MatrixAType N;
	N.resize(VectorPlen, VectorNlen);
    N.reserve(Eigen::VectorXi::Constant(VectorNlen, 3));

	VectorBType nClosure = VectorSolType::Zero(VectorPlen);

    VectorSolType res = VectorSolType::Zero(VectorPlen);
    VectorSolType delta = VectorSolType::Zero(VectorPlen);

    for (int iter = 0; iter < nIter; iter++) {

        std::cout << "\tStarting iterations " << iter << std::endl;

		for (int i = 0; i < outSize[0]; i++) {
			for (int j = 0; j < outSize[1]; j++) {
				int i1 = idxConverterOut.getPseudoFlatIdFromIndex({i,j,0});
				int i2 = idxConverterOut.getPseudoFlatIdFromIndex({i,j,1});
				int i3 = idxConverterOut.getPseudoFlatIdFromIndex({i,j,2});

				int i_n = idxConverterIn.getPseudoFlatIdFromIndex({i,j});

				ComputeType x1 = solution[i1];
				ComputeType x2 = solution[i2];
				ComputeType x3 = solution[i3];

                ComputeType quadr = (x1*x1 + x2*x2 + x3*x3 - 1);

                nClosure[i_n] = quadr + 4*quadr*x1*x1 + 4*quadr*x2*x2 + 4*quadr*x3*x3;

                N.coeffRef(i_n,i1) = 4*quadr*x1;
                N.coeffRef(i_n,i2) = 4*quadr*x2;
                N.coeffRef(i_n,i3) = 4*quadr*x3;
			}
        }

        std::cout << "\tFinished rebuilding N " << std::endl;

        MatrixAType A = Abase;
        VectorBType b = P.transpose()*p + lambdaNorm*N.transpose()*nClosure;
        A += lambdaNorm*N.transpose()*N;

        A.makeCompressed();

        //SolverType solver;

		std::cout << "\tPreparing iterations " << iter << std::endl;

        // Compute the ordering permutation vector from the structural pattern of A
        //solver.analyzePattern(A);
        // Compute the numerical factorization
        //solver.factorize(A);
        //solver.compute(A);

        /*if(solver.info() != Eigen::Success) {

            std::cout << "\tFailure to factorize " << iter << std::endl;
            return Multidim::Array<ComputeType, 3>();
        }*/

        //since we will iterate anyway, do a single step of conjugate gradient method each time.
        res = b - A*solution;
        ComputeType scale = res.dot(A*res);
        ComputeType alpha = res.dot(res)/scale;

        delta = alpha*res;
        std::cout << "\titerations " << iter << std::endl;

        /*if(solver.info() != Eigen::Success) {

            std::cout << "\tFailure to optimize " << iter << std::endl;
			return Multidim::Array<ComputeType, 3>();
        }*/

        if(!delta.array().isFinite().all()) {
			std::cout << "Non finite increments in solution!" << std::endl;
			return Multidim::Array<ComputeType, 3>();
		}

        solution += delta;

		if (delta.norm()/VectorNlen < incrTol) {
            break;
        }

        std::cout << "\titerations ended " << iter << std::endl;
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
Multidim::Array<ComputeType, 3> rectifyNormalMap(Multidim::Array<ComputeType, 3> const& normalmap, Multidim::Array<bool, 2> const& mask) {

	using NormalT = Eigen::Matrix<ComputeType, 3, 1>;

	NormalT sum = NormalT::Zero();

	auto shape = normalmap.shape();

	if (mask.shape()[0] != shape[0] or mask.shape()[1] != shape[1]) {
		return Multidim::Array<ComputeType, 3>();
	}

	int nPixs = 0;

	for (int i = 0; i < mask.shape()[0]; i++) {
		for (int j = 0; j < mask.shape()[1]; j++) {

			if (mask.valueUnchecked(i,j)) {
				nPixs++;

				NormalT normal;

				normal[0] = normalmap.valueUnchecked(i,j,0);
				normal[1] = normalmap.valueUnchecked(i,j,1);
				normal[2] = normalmap.valueUnchecked(i,j,2);

				sum += normal;
			}

		}
	}

	NormalT mean = sum / nPixs;

	NormalT vertical(0,0,1);

	NormalT dir = mean.cross(vertical);

	double alpha = std::acos(mean.dot(vertical)/mean.norm());

	NormalT axisAngle = alpha*dir.normalized();

	Eigen::Matrix<ComputeType, 3, 3> R = Geometry::rodriguezFormula<ComputeType>(axisAngle);

	Multidim::Array<ComputeType, 3> ret(shape);

	for (int i = 0; i < mask.shape()[0]; i++) {
		for (int j = 0; j < mask.shape()[1]; j++) {

			NormalT normal;

			normal[0] = normalmap.valueUnchecked(i,j,0);
			normal[1] = normalmap.valueUnchecked(i,j,1);
			normal[2] = normalmap.valueUnchecked(i,j,2);

			NormalT recitifed = R*normal;

			ret.atUnchecked(i,j,0) = recitifed[0];
			ret.atUnchecked(i,j,1) = recitifed[1];
			ret.atUnchecked(i,j,2) = recitifed[2];
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

			diffMap.atUnchecked(i,j,0) = dx;
			diffMap.atUnchecked(i,j,1) = dy;
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

			A.coeffRef(i,jM1) += -1;
			A.coeffRef(i,jP1) += 1;

		} else if (idxIn[2] == 1) { //yDiff
			int yPosM1 = (outCenterIdx[0] < shape[0]-1) ? outCenterIdx[0]+1 : shape[0]-1;
			int yPosP1 = (outCenterIdx[0] > 0) ? outCenterIdx[0]-1 : 0;

			int jM1 = outIdxs.getPseudoFlatIdFromIndex({yPosM1, outCenterIdx[1]});
			int jP1 = outIdxs.getPseudoFlatIdFromIndex({yPosP1, outCenterIdx[1]});

			A.coeffRef(i,jM1) += -1;
			A.coeffRef(i,jP1) += 1;
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

template<typename ComputeType>
Multidim::Array<ComputeType, 2> flattenHeightMapInAreaOfInterest(Multidim::Array<ComputeType, 2> const& baseHeightMap, Multidim::Array<bool, 2> const& mask) {

	using MatAT = Eigen::Matrix<ComputeType, Eigen::Dynamic, 3>;
	using VecbT = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;
	using VecSolT = Eigen::Matrix<ComputeType, 3, 1>;

	auto shape = baseHeightMap.shape();

	if (mask.shape() != shape) {
		return Multidim::Array<ComputeType, 2>();
	}

	int nPixs = 0;

	for (int i = 0; i < mask.shape()[0]; i++) {
		for (int j = 0; j < mask.shape()[1]; j++) {

			if (mask.valueUnchecked(i,j)) {
				nPixs++;
			}

		}
	}

	MatAT A;
	A.resize(nPixs, 3);

	VecbT b;
	b.resize(nPixs);

	int r = 0;

	for (int i = 0; i < mask.shape()[0]; i++) {
		for (int j = 0; j < mask.shape()[1]; j++) {

			if (mask.valueUnchecked(i,j)) {

				A(r,0) = i;
				A(r,1) = j;
				A(r,2) = 1;

				b[r] = baseHeightMap.valueUnchecked(i,j);

				r++;
			}

		}
	}

	VecSolT coeffs = A.colPivHouseholderQr().solve(b); //least square approximation

	Multidim::Array<ComputeType, 2> ret(shape);

	ComputeType minVal = std::numeric_limits<ComputeType>::max();

	for (int i = 0; i < mask.shape()[0]; i++) {
		for (int j = 0; j < mask.shape()[1]; j++) {

			if (mask.valueUnchecked(i,j)) {
				ComputeType trend = i*coeffs[0] + j*coeffs[1] + coeffs[2];
				ComputeType val = baseHeightMap.valueUnchecked(i,j) - trend;
				ret.atUnchecked(i,j) = val;

				if (minVal > val) {
					minVal = val;
				}
			}

		}
	}

	for (int i = 0; i < mask.shape()[0]; i++) {
		for (int j = 0; j < mask.shape()[1]; j++) {

			if (mask.valueUnchecked(i,j)) {
				ret.atUnchecked(i,j) -= minVal;
			} else {
				ret.atUnchecked(i,j) = 0;
			}

		}
	}

	return ret;

}

} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_SHAPEFROMSHADING_H
