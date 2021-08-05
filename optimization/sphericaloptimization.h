#ifndef SPHERICALOPTIMIZATION_H
#define SPHERICALOPTIMIZATION_H

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

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>

#include "./affine_utils.h"

namespace StereoVision {
namespace Optimization {

/*!
 * sphericalAffineBestApproximation solve a problem of the form argmin_x || Ax/||Ax||_2 - b/||b||_2 ||_2 with the constraint that sum(x) == 1 using a robust dense solver.
 * The function is optimized for small problems
 * \return The optimal x.
 */
template<int dimsIn, int dimsOuts = Eigen::Dynamic>
inline Eigen::Matrix<float,dimsIn,1> sphericalAffineBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
																	  Eigen::Matrix<float,dimsOuts,1> const& b) {

	static_assert (dimsIn >= 2, "sphericalAffineBestApproximation expect x dimension to be greather or equal to 2.");

	constexpr int referenceId = -1;
	typedef AffineSpace<dimsIn, dimsOuts, float, referenceId> AffineSpaceA ;

	typedef Eigen::Matrix<float,dimsOuts,dimsIn> TypeMatrixA;
	typedef typename AffineSpaceA::TypeMatrixM TypeMatrixM;
	typedef Eigen::Matrix<float,dimsIn,1> TypeVectorX;
	typedef Eigen::Matrix<float,dimsOuts,1> TypeVectorB;
	typedef typename AffineSpaceA::TypeVectorCoeffs TypeVectorAlpha;

	AffineSpaceA AffineA(A);

	Eigen::FullPivHouseholderQR<TypeMatrixA> QRA(A);

	TypeVectorB bPerp = A*QRA.solve(b);

	Eigen::FullPivHouseholderQR<TypeMatrixM> QRM(AffineA.M());

	Eigen::VectorXf aPerp = AffineA.b() - AffineA.M()*QRM.solve(AffineA.b());

	float g = (aPerp.dot(aPerp))/(aPerp.dot(bPerp));

	TypeVectorAlpha alpha = QRM.solve(g*bPerp - AffineA.b());
	return AffineSpaceA::fullCoeffs(alpha);
}

} // namespace Optimization
} // namespace StereoVision

#endif // SPHERICALOPTIMIZATION_H
