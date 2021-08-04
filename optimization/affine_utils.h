#ifndef LIBSTEVI_AFFINE_UTILS_H
#define LIBSTEVI_AFFINE_UTILS_H

#include <eigen3/Eigen/Core>

namespace StereoVision {
namespace Optimization {

template<int nVec, int spaceDim, typename Scalar = float, int referenceId = -1>
class AffineSpace
{
public:

	static_assert (nVec >= 2, "You need at least two vectors to span an affine subspace");
	static_assert (std::abs(referenceId) < nVec, "Reference column id out of bound, must respect -nVec < referenceId < nVec");

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef Eigen::Matrix<Scalar, spaceDim, nVec> TypeMatrixA;
	typedef Eigen::Matrix<Scalar, spaceDim, nVec-1> TypeMatrixM;
	typedef Eigen::Matrix<Scalar, spaceDim, 1> TypeVectorb;
	typedef Eigen::Matrix<Scalar, nVec-1, 1> TypeVectorCoeffs;
	typedef Eigen::Matrix<Scalar, nVec, 1> TypeVectorCoeffsFull;

	constexpr static int offsetId = (nVec + referenceId)%nVec; //compute the column in A which will be the offset of the affine space.

	explicit AffineSpace(TypeMatrixA const& A) {

		_b = A.col(offsetId);

		int t = 0;
		for (int i = 0; i<nVec; i++) {
			if (i == offsetId) {
				continue;
			}
			_M.col(t) = A.col(i) - _b;
			t++;
		}
	}

	TypeMatrixM& M() {
		return _M;
	}

	TypeMatrixM const& M() const{
		return _M;
	}

	TypeVectorb& b() {
		return _b;
	}

	TypeVectorb const& b() const{
		return _b;
	}

	static inline TypeVectorCoeffsFull fullCoeffs(TypeVectorCoeffs const& coeffs) {

		TypeVectorCoeffsFull cFull;

		int t = 0;
		for (int i = 0; i<nVec; i++) {
			if (i == offsetId) {
				continue;
			}
			cFull(i) = coeffs(t);
			t++;
		}
		cFull(offsetId) = 1 - coeffs.sum();
		return cFull;
	}

private :

	TypeMatrixM _M;
	TypeVectorb _b;

};

template<int nVec, typename Scalar, int referenceId>
class AffineSpace<nVec, Eigen::Dynamic, Scalar, referenceId> {

public:

	static_assert (nVec >= 2, "You need at least two vectors to span an affine subspace");

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, nVec> TypeMatrixA;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, nVec-1> TypeMatrixM;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> TypeVectorb;
	typedef Eigen::Matrix<Scalar, nVec-1, 1> TypeVectorCoeffs;
	typedef Eigen::Matrix<Scalar, nVec, 1> TypeVectorCoeffsFull;

	constexpr static int offsetId = (nVec + referenceId)%nVec; //compute the column in A which will be the offset of the affine space.

	explicit AffineSpace(TypeMatrixA const& A) {
		int nRows = A.rows();
		_b.resize(nRows);
		_M.resize(nRows, nVec-1);

		_b = A.col(offsetId);

		int t = 0;
		for (int i = 0; i<nVec; i++) {
			if (i == offsetId) {
				continue;
			}
			_M.col(t) = A.col(i) - _b;
			t++;
		}
	}

	TypeMatrixM& M() {
		return _M;
	}

	TypeMatrixM const& M() const{
		return _M;
	}

	TypeVectorb& b() {
		return _b;
	}

	TypeVectorb const& b() const{
		return _b;
	}

	static inline TypeVectorCoeffsFull fullCoeffs(TypeVectorCoeffs const& coeffs) {

		TypeVectorCoeffsFull cFull;

		int t = 0;
		for (int i = 0; i<nVec; i++) {
			if (i == offsetId) {
				continue;
			}
			cFull(i) = coeffs(t);
			t++;
		}
		cFull(offsetId) = 1 - coeffs.sum();
		return cFull;
	}

private :

	TypeMatrixM _M;
	TypeVectorb _b;
};

} // Optimization
} // StereoVision

#endif // LIBSTEVI_AFFINE_UTILS_H
