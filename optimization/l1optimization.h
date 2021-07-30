#ifndef LIBSTEVI_L1_OPTIMIZATION_H
#define LIBSTEVI_L1_OPTIMIZATION_H

#include <vector>
#include <set>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>

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
template<int dims>
Eigen::Matrix<float,dims,1> leastAbsoluteDifferences(Eigen::Matrix<float,Eigen::Dynamic,dims> const& A,
													 Eigen::VectorXf const& b,
													 float tol=1e-6,
													 int maxIters = 100) {

	auto softSignFunc = [tol] (float val) -> float {
		if (val > tol) {
			return 1;
		} else if (val < -tol) {
			return -1;
		}
		return 0;
	};

	typedef Eigen::Matrix<float,Eigen::Dynamic,dims> TypeMatrixA;
	typedef Eigen::Matrix<float,dims,1> TypeVectorX;
	typedef Eigen::Matrix<float,1,dims> TypeConstraintRow;

	constexpr int maxNConstraints = dims-1;
	typedef Eigen::Matrix<float,dims-1,dims> TypeMatrixZ;
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
		d = constrainVectorToSubspace<dims>(d, constraints).eval();

		float d_norm = d.template lpNorm<1>();

		while (d_norm < dims*tol) {
			//if the algorithm get stuck before ending we just take an optimization step in a random direction
			//TODO: test if this is clever and implement a better variable selection if need there is.
			d = TypeVectorX::Random();
			d = constrainVectorToSubspace<dims>(d, constraints).eval();
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

	Eigen::FullPivHouseholderQR<TypeMatrixA> QRA(A);

	TypeVectorX x = QRA.solve(b); //start on the reference feature vector.

	TypeMatrixZ Z = TypeMatrixZ::Zero();
	TypeMatrixZ prevZ = TypeMatrixZ::Zero();
	TypeConstraintsInputs constraintInputs;
	std::fill(constraintInputs.begin(), constraintInputs.end(), -1);
	int idNextConstraint = 0;
	int idPrevConstraint = 0;
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
			if (incr_norm < dims*tol) {// check all possible edges, if all lead to 0 increment then global minimum is reached.
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

					if (incr2_norm > dims*tol) {
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

} // namespace Optimization
} // namespace StereoVision

#endif // LIBSTEVI_L1_OPTIMIZATION_H
