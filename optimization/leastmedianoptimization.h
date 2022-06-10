#ifndef STEREOVISION_LEASTMEDIANOPTIMIZATION_H
#define STEREOVISION_LEASTMEDIANOPTIMIZATION_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022  Paragon<french.paragon@gmail.com>

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

#include <eigen3/Eigen/Core>
#include <cmath>
#include <random>

#include "./l2optimization.h"

namespace StereoVision {
namespace Optimization {

/*!
 * leastSquares solve a problem of the form argmin_x Median(|Ax - b|) using a robust dense solver. If the problem is underdetermined the minimal norm solution is returned.
 * The function is optimized for small problems
 * \return The optimal x.
 */
template<int dimsIn, int dimsOuts = Eigen::Dynamic>
inline Eigen::Matrix<float,dimsIn,1> leastAbsoluteMedian(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
														 Eigen::Matrix<float,dimsOuts,1> const& b,
														 float prob_optimal = 0.99,
														 float prob_outlier = 0.3,
														 int maxiter = 10000) {

	using IntermMatType = Eigen::Matrix<float, dimsIn, dimsIn>;
	using SolVecType = Eigen::Matrix<float, dimsIn, 1>;
	using InType = Eigen::Matrix<float,dimsOuts,1>;

	if (A.rows() <= A.cols()) {
		return leastSquares(A,b);
	}

	int medianPos = A.rows()/2;

	int p = A.cols();

	double pout_single = 1 - std::pow(1-prob_outlier, p);
	double nd = std::ceil(std::log(pout_single)/std::log(1 - prob_optimal));
	int n = static_cast<int>(nd);

	if (n > maxiter) {
		n = maxiter;
	}

	if (n < A.rows()) {
		n = A.rows();
	}

	SolVecType sol;
	float medErr = std::numeric_limits<float>::infinity();

	std::vector<int> idxs(A.rows());

	for (int i = 0; i < A.rows(); i++) {
		idxs[i] = i;
	}

	for (int i = 0; i < n; i++) {
		std::random_shuffle(idxs.begin(), idxs.end());

		SolVecType sub_b(A.cols());
		IntermMatType sub_A(A.cols(), A.cols());

		for (int j = 0; j < A.cols(); j++) {
			sub_b(j) = b(idxs[j]);
			sub_A.row(j) = A.row(idxs[j]);
		}

		SolVecType tmp = leastSquares(sub_A, sub_b); //reuse the least square solver, which avoid numerical errors

		InType s = A*tmp;
		InType err = b - s;

		std::vector<int> abs_errs(A.rows());

		for (int j = 0; j < A.rows(); j++) {
			abs_errs[j] = std::fabs(err[j]);
		}

		std::nth_element(abs_errs.begin(), abs_errs.begin()+medianPos, abs_errs.end());
		float median_err = abs_errs[medianPos];

		if (median_err < medErr) {
			medErr = median_err;
			sol = tmp;
		}
	}

	return sol;

}

/*!
 * affineBestLeastMedianApproximation solve a problem of the form argmin_x Median(|Ax - b|) with the constraint that sum(x) == 1 using a robust dense solver.
 * The function is optimized for small problems
 * \return The optimal x.
 */
template<int dimsIn, int dimsOuts = Eigen::Dynamic>
inline Eigen::Matrix<float,dimsIn,1> affineBestLeastMedianApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
																		Eigen::Matrix<float,dimsOuts,1> const& b,
																		float prob_optimal = 0.99,
																		float prob_outlier = 0.3,
																		int maxiter = 10000) {

	static_assert (dimsIn >= 2, "affineBestL2Approximation expect x dimension to be greather or equal to 2.");

	constexpr int referenceId = -1;
	typedef AffineSpace<dimsIn, dimsOuts, float, referenceId> AffineSpaceA ;

	typedef Eigen::Matrix<float,dimsOuts,dimsIn> TypeMatrixA;
	typedef typename AffineSpaceA::TypeVectorCoeffs TypeVectorAlpha;

	AffineSpaceA AffineA(A);
	TypeVectorAlpha alpha = leastAbsoluteMedian(AffineA.M(), (b - AffineA.b()).eval(), prob_optimal, prob_outlier, maxiter);
	return AffineSpaceA::fullCoeffs(alpha);
}

} // namespace StereoVision
} // namespace Optimization

#endif // STEREOVISION_LEASTMEDIANOPTIMIZATION_H
