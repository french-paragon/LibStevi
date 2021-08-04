#ifndef SPHERICALOPTIMIZATION_H
#define SPHERICALOPTIMIZATION_H

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
