#ifndef LIBSTEVI_L1_OPTIMIZATION_H
#define LIBSTEVI_L1_OPTIMIZATION_H

#include <vector>
#include <set>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>

#include "./l2optimization.h"

namespace StereoVision {
namespace Optimization {

template<typename T>
int weighted_median_index(std::vector<T> const& elements, std::vector<float> const& weights) {

	//TODO implement the O(n) version of the algorithm, current version is the O(nlog(n)) complexity algorithm.
	std::vector<int> sorting_key(elements.size());
	for(int i = 0; i < static_cast<int>(sorting_key.size()); i++) {sorting_key[i] = i;}

	float sumWeights = 0;
	for(int i = 0; i < static_cast<int>(weights.size()); i++) {sumWeights += weights[i];}

	std::sort(sorting_key.begin(), sorting_key.end(), [&elements] (int i1, int i2) {
		return elements[i1] < elements[i2];
	});

	float partialSum = 0;
	for(int i = 0; i < static_cast<int>(sorting_key.size()); i++) {

		float tmp = 2*weights[sorting_key[i]];

		if (partialSum + tmp > sumWeights) {
			return sorting_key[i];
		}

		partialSum += tmp;
	}

	return sorting_key.back();
}

template<typename T>
T weighted_median(std::vector<T> const& elements, std::vector<float> const& weights) {

	return elements[weighted_median_index(elements, weights)];
}


template<int dims>
inline Eigen::Matrix<float,dims,1> constrainVectorToSubspace(Eigen::Matrix<float,dims,1> const& unconstrained,
															 Eigen::Matrix<float,dims-1,dims> const& constaints) {
	static_assert (dims >= 2, "The vector space should have a least two dimensions !");

	Eigen::FullPivHouseholderQR<Eigen::Matrix<float,dims-1,dims>> QRZ(constaints.transpose());
	Eigen::Matrix<float,dims,1> result = unconstrained - constaints.transpose()*QRZ.solve(unconstrained);

	return result;
}

template<>
inline Eigen::Matrix<float,2,1> constrainVectorToSubspace<2>(Eigen::Matrix<float,2,1> const& unconstrained,
															 Eigen::Matrix<float,1,2> const& constraints) {

	float norm = constraints.norm();
	if (norm < 1e-6) {
		return unconstrained;
	}

	float dot = unconstrained.dot(constraints.transpose());

	Eigen::Matrix<float,2,1> result = unconstrained - constraints.transpose()*(dot/norm);

	return result;
}

/*!
 * leastAbsoluteDifferences solve a problem of the form argmin_x || Ax - b ||_1 using dimensionality reduction and weigthed median.
 * \return The optimal x.
 */
template<int dimsIn, int dimsOuts = Eigen::Dynamic>
Eigen::Matrix<float,dimsIn,1> leastAbsoluteDifferences(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
													   Eigen::Matrix<float,dimsOuts,1> const& b,
													   float tol=1e-6,
													   int maxIters = 100) {

	static_assert (dimsIn >= 2, "leastAbsoluteDifferences expect x dimension to be greather or equal to 2.");

	auto softSignFunc = [tol] (float val) -> float {
		if (val > tol) {
			return 1;
		} else if (val < -tol) {
			return -1;
		}
		return 0;
	};

	typedef Eigen::Matrix<float,dimsOuts,dimsIn> TypeMatrixA;
	typedef Eigen::Matrix<float,dimsIn,1> TypeVectorX;
	typedef Eigen::Matrix<float,1,dimsIn> TypeConstraintRow;

	constexpr int maxNConstraints = dimsIn-1;
	typedef Eigen::Matrix<float,dimsIn-1,dimsIn> TypeMatrixZ;
	typedef std::array<int, maxNConstraints> TypeConstraintsInputs;
	typedef std::set<int> TypeUnconstrainedInputs;

	auto nextIncrement = [] (TypeMatrixA const& A,
								TypeVectorX const& x,
								Eigen::VectorXf const& b,
								TypeMatrixZ const& constraints,
								TypeUnconstrainedInputs const& unconstrained,
								float tol,
								int & nextConstraint) -> TypeVectorX {

		TypeVectorX d = TypeVectorX::Random();
		d = constrainVectorToSubspace<dimsIn>(d, constraints).eval();

		float d_norm = d.template lpNorm<1>();

		while (d_norm < dimsIn*tol) {
			//if the algorithm get stuck before ending we just take an optimization step in a random direction
			//TODO: test if this is clever and implement a better variable selection if need there is.
			d = TypeVectorX::Random();
			d = constrainVectorToSubspace<dimsIn>(d, constraints).eval();
			d_norm = d.template lpNorm<1>();
		}

		d = d/d_norm; //cost time, but improve numerical stability.

		int nInputs = A.rows();

		Eigen::VectorXf Ax = A*x;
		Eigen::VectorXf Ad = A*d;

		std::vector<float> t_breaks;
		t_breaks.reserve(nInputs);
		std::vector<float> t_weights;
		t_weights.reserve(nInputs);
		std::vector<int> indices;
		indices.reserve(nInputs);

		for (int c : unconstrained) {

			float o = Ad(c);
			t_weights.push_back(std::abs(o));
			t_breaks.push_back(-(Ax(c) - b(c))/o);
			if (std::isnan(t_breaks.back()) or std::isinf(t_breaks.back())) {
				t_weights.back() = 0;
				t_breaks.back() = -std::numeric_limits<float>::infinity();
			}
			indices.push_back(c);

		}

		int t_hat_index = Optimization::weighted_median_index(t_breaks, t_weights);
		nextConstraint = indices[t_hat_index];
		return d*t_breaks[t_hat_index];

	};

	TypeVectorX x = leastSquares(A, b); //start with the l2 solution.

	TypeMatrixZ Z = TypeMatrixZ::Zero();
	TypeMatrixZ prevZ = TypeMatrixZ::Zero();
	TypeConstraintsInputs constraintInputs;
	std::fill(constraintInputs.begin(), constraintInputs.end(), -1);
	int idNextConstraint = 0;
	bool constraintsFull = false;

	TypeVectorX d = TypeVectorX::Zero();

	int nInputs = A.rows();

	TypeUnconstrainedInputs unconstrainedInputs;
	for (int i=0; i < nInputs; i++) { unconstrainedInputs.insert(i); }

	for (int i = 0; i < maxIters; i++) {

		int nextConstraint;

		TypeVectorX incr = nextIncrement(A,x,b,Z,unconstrainedInputs, tol, nextConstraint);

		if (constraintsFull) {
			float incr_norm = incr.template lpNorm<1>();
			if (incr_norm < dimsIn*tol) {// check all possible edges, if all lead to 0 increment then global minimum is reached.
				prevZ = Z;
				bool sucess = true;
				unconstrainedInputs.erase(nextConstraint);

				for (int n = 0; n < maxNConstraints; n++) {
					Z = prevZ;
					int outConstraint = constraintInputs[n];
					unconstrainedInputs.insert(outConstraint);
					Z.row(n) = A.row(nextConstraint);

					int nnextConstraint;

					TypeVectorX incr2 = nextIncrement(A,x,b,Z,unconstrainedInputs, tol, nnextConstraint);
					float incr2_norm = incr2.template lpNorm<1>();

					if (incr2_norm > dimsIn*tol) {
						sucess = false;
						constraintInputs[n] = nextConstraint;
						idNextConstraint = (n+1)%maxNConstraints;
						nextConstraint = nnextConstraint;
						incr += incr2;
						break;
					}

					unconstrainedInputs.erase(outConstraint);
				}

				if (sucess) { //global minima reached
					break;
				}
			}
		}

		x = x + incr;
		Z.row(idNextConstraint) = A.row(nextConstraint);

		int outConstraint = constraintInputs[idNextConstraint];
		if (outConstraint >= 0) {unconstrainedInputs.insert(outConstraint);}
		unconstrainedInputs.erase(nextConstraint);
		constraintInputs[idNextConstraint] = nextConstraint;

		idNextConstraint++;

		if (idNextConstraint >= maxNConstraints) {
			idNextConstraint = 0;
			constraintsFull = true;
		}

	}

	return x;

}

template<int dimsOuts = Eigen::Dynamic>
Eigen::Matrix<float,1,1> leastAbsoluteDifferences(Eigen::Matrix<float,dimsOuts,1> const& A,
												  Eigen::Matrix<float,dimsOuts,1> const& b,
												  float tol=1e-6,
												  int maxIters = 100) {

	(void) tol;
	(void) maxIters;

	int nInputs = A.rows();

	std::vector<float> t_breaks(nInputs);
	std::vector<float> t_weights(nInputs);

	for (int i = 0; i < b.rows(); i++) {
		float o = A(i);
		t_weights[i] = std::abs(o);
		t_breaks[i] = b(i)/o;
		if (std::isnan(t_breaks.back()) or std::isinf(t_breaks.back())) {
			t_weights[i] = 0;
			t_breaks[i] = -std::numeric_limits<float>::infinity();
		}
	}

	float alpha = weighted_median(t_breaks, t_weights);
	return Eigen::Matrix<float,1,1>(alpha);

}

/*!
 * affineBestL1Approximation solve a problem of the form argmin_x || Ax - b ||_1 with the constraint that sum(x) == 1 using a dense solver.
 * The function is optimized for small problems.
 * The function, while it is guaranteed to reach the global optimum in finite time, is not perfectly robust to numerical imprecision and might converge in a really long time.
 * This is why a maximal number of iterations, as well as a floating point tolerance, have been added.
 * \return The optimal x.
 */
template<int dimsIn, int dimsOuts = Eigen::Dynamic>
inline Eigen::Matrix<float,dimsIn,1> affineBestL1Approximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
															   Eigen::Matrix<float,dimsOuts,1> const& b,
															   float tol=1e-6,
															   int maxIters = 100) {

	static_assert (dimsIn >= 2, "affineBestL1Approximation expect x dimension to be greather or equal to 2.");

	constexpr int referenceId = -1;
	typedef AffineSpace<dimsIn, dimsOuts, float, referenceId> AffineSpaceA ;

	typedef Eigen::Matrix<float,dimsOuts,dimsIn> TypeMatrixA;
	typedef typename AffineSpaceA::TypeVectorCoeffs TypeVectorAlpha;

	AffineSpaceA AffineA(A);
	TypeVectorAlpha alpha = leastAbsoluteDifferences(AffineA.M(), (b - AffineA.b()).eval(), tol, maxIters);
	return AffineSpaceA::fullCoeffs(alpha);
}

} // namespace Optimization
} // namespace StereoVision

#endif // LIBSTEVI_L1_OPTIMIZATION_H
