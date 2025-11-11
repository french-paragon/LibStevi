#ifndef STEREOVISION_ALIGNEMENT_H
#define STEREOVISION_ALIGNEMENT_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021-2022 Paragon<french.paragon@gmail.com>

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

#include "../geometry/core.h"
#include "../geometry/imagecoordinates.h"
#include "../geometry/rotations.h"
#include "../geometry/lensdistortion.h"
#include "../geometry/geometricexception.h"

#include "../optimization/l2optimization.h"

#include <MultidimArrays/MultidimIndexManipulators.h>

#include <utility>
#include <vector>

namespace StereoVision {
namespace Geometry {

template <typename Pos_T, int cols>
/*!
 * \brief projectPoints project points in homogeneous normalized image coordinates
 * \param pts the points to project, assumed to be given in the camera frame
 * \return the projected points coordinates.
 */
Eigen::Array<Pos_T, 2, cols> projectPoints(Eigen::Array<Pos_T, 3, cols> const& pts) {
	Eigen::Array<Pos_T, 2, cols> proj;
	proj.resize(2, pts.cols());

	proj.row(0) = pts.row(0)/pts.row(2);
	proj.row(1) = pts.row(1)/pts.row(2);

	return proj;
}
/*!
 * \brief projectPoints project points in homogeneous normalized image coordinates, given a scene to camera transform
 * \param pts the points to project, given in a scene frame
 * \param T the transformation from the scene frame to the camera frame
 * \return the projected points coordinates.
 */
template <typename Pos_T, int cols>
Eigen::Array<Pos_T, 2, cols> projectPoints(Eigen::Array<Pos_T, 3, cols> const& pts, AffineTransform<Pos_T> const& T) {
	Eigen::Array<Pos_T, 3, cols> transformedPts = T*pts;
	return projectPoints(transformedPts);
}
template <typename Pos_T, int cols>
Eigen::Array<Pos_T, 2, cols> projectPoints(Eigen::Array<Pos_T, 3, cols> const& pts,
													 Eigen::Matrix<Pos_T, 3, 3> const& R,
													 Eigen::Matrix<Pos_T, 3, 1>  const& t) {

	return projectPoints(pts, AffineTransform<Pos_T>(R,t));
}


template <typename Pos_T, int cols>
Eigen::Matrix<Pos_T, 2, cols> projectPoints(Eigen::Matrix<Pos_T, 3, cols> const& pts) {
	Eigen::Matrix<Pos_T, 2, cols> proj;
	proj.resize(2, pts.cols());

	proj.row(0) = (pts.row(0).array()/pts.row(2).array()).matrix();
	proj.row(1) = (pts.row(1).array()/pts.row(2).array()).matrix();

	return proj;
}
template <typename Pos_T, int cols>
Eigen::Matrix<Pos_T, 2, cols> projectPoints(Eigen::Matrix<Pos_T, 3, cols> const& pts, AffineTransform<Pos_T> const& T) {
	Eigen::Matrix<Pos_T, 3, cols> transformedPts = T*pts;
	return projectPoints(transformedPts);
}
template <typename Pos_T, int cols>
Eigen::Matrix<Pos_T, 2, cols> projectPoints(Eigen::Matrix<Pos_T, 3, cols> const& pts,
													 Eigen::Matrix<Pos_T, 3, 3> const& R,
													 Eigen::Matrix<Pos_T, 3, 1>  const& t) {

	return projectPoints(pts, AffineTransform<Pos_T>(R,t));
}

template <typename Pos_T>
Eigen::Matrix<Pos_T, 2,1> World2ImageCoordinates(Eigen::Matrix<Pos_T, 3,1> const& worldCoords,
												 AffineTransform<Pos_T> const& World2Cam,
												 Eigen::Matrix<Pos_T, 2,1> const& f,
												 Eigen::Matrix<Pos_T, 2,1> const& pp)
{

	Eigen::Matrix<Pos_T, 2,1> homogeneous = projectPoints(worldCoords, World2Cam);
	return Homogeneous2ImageCoordinates(homogeneous, f, pp);
}

template <typename Pos_T, typename K_T, typename T_T, typename B_T>
Eigen::Matrix<Pos_T, 2,1> World2DistortedImageCoordinates(Eigen::Matrix<Pos_T, 3,1> const& worldCoords,
														  AffineTransform<Pos_T> const& World2Cam,
														  Eigen::Matrix<Pos_T, 2,1> const& f,
														  Eigen::Matrix<Pos_T, 2,1> const& pp,
														  std::optional<Eigen::Matrix<K_T, 3, 1>> k123 = std::nullopt,
														  std::optional<Eigen::Matrix<T_T, 2, 1>> t12 = std::nullopt,
														  std::optional<Eigen::Matrix<B_T, 2, 1>>  B12 = std::nullopt)
{

	Eigen::Matrix<Pos_T, 2,1> homogeneous = projectPoints(worldCoords, World2Cam);
	return fullLensDistortionHomogeneousCoordinates(homogeneous, f, pp, k123, t12, B12);
}

template <typename Pos_T, typename K_T = float, typename T_T = float, typename B_T = float>
Eigen::Matrix<Pos_T, 2,1> World2DistortedImageCoordinates(Eigen::Matrix<Pos_T, 3,1> const& worldCoords,
														  AffineTransform<Pos_T> const& World2Cam,
														  Pos_T f,
														  Eigen::Matrix<Pos_T, 2,1> const& pp,
														  std::optional<Eigen::Matrix<K_T, 3, 1>> k123 = std::nullopt,
														  std::optional<Eigen::Matrix<T_T, 2, 1>> t12 = std::nullopt,
														  std::optional<Eigen::Matrix<B_T, 2, 1>>  B12 = std::nullopt)
{
	return World2DistortedImageCoordinates(worldCoords, World2Cam, Eigen::Matrix<Pos_T, 2,1>(f,f), pp, k123, t12, B12);
}

template <typename Pos_T, int nCols>
/*!
 * \brief reprojectPoints from the transformation from cam1 to cam2 and a set of points in homogeneous coordinates in both images, find the 3D coordinates of points in cam1 frame.
 * \param R the rotation part of the transform cam1 2 cam2
 * \param t the translation part of the transform cam1 2 cam2
 * \param pt_cam_1 points in cam1 homogeneous coordinates
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the points coordinates in cam1 frame
 */
Eigen::Array<Pos_T, 3, nCols> reprojectPoints(Eigen::Matrix<Pos_T, 3, 3> const& R,
											  Eigen::Matrix<Pos_T, 3, 1> const& t,
											  Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1,
											  Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {

	int nPts = pt_cam_1.cols();

	if (pt_cam_1.cols() != pt_cam_2.cols()) {
		throw GeometricException("Points arrays of different dimensions provided");
	}

	Eigen::Array<Pos_T, 3, nCols> reproj;

	reproj.resize(3,nPts);
	reproj.topRows(2) = pt_cam_1;
	reproj.bottomRows(1).setOnes();

	Eigen::Array<Pos_T, 1, nCols> x3_v1 = (t.x() - pt_cam_2.row(0)*t.z()) /
			(
				pt_cam_2.row(0)*(R(2,0)*pt_cam_1.row(0) + R(2,1)*pt_cam_1.row(1) + R(2,2)) -
				(R(0,0)*pt_cam_1.row(0) + R(0,1)*pt_cam_1.row(1) + R(0,2))
			);

	Eigen::Array<Pos_T, 1, nCols> x3_v2 = (t.y() - pt_cam_2.row(1)*t.z()) /
			(
				pt_cam_2.row(1)*(R(2,0)*pt_cam_1.row(0) + R(2,1)*pt_cam_1.row(1) + R(2,2)) -
				(R(1,0)*pt_cam_1.row(0) + R(1,1)*pt_cam_1.row(1) + R(1,2))
			);

	Eigen::Array<Pos_T, 1, nCols> x3 = (x3_v1 + x3_v2)/2.0;

	//make sure to get only the values of x3_v1 if x3_v2 is not finite and vice versa.
	x3 = (x3.isFinite()).select(x3,x3_v1);
	x3 = (x3.isFinite()).select(x3,x3_v2);

	reproj.row(0) *= x3;
	reproj.row(1) *= x3;
	reproj.row(2) *= x3;

	return reproj;

}

/*!
 * \brief reprojectPoints from the transformation from cam1 to cam2 and a set of points in homogeneous coordinates in both images, find the 3D coordinates of points in cam1 frame.
 * \param T the transform cam1 to cam2
 * \param pt_cam_1 points in cam1 homogeneous coordinates
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the points coordinates in cam1 frame
 */
template <typename Pos_T, int nCols>
Eigen::Array<Pos_T, 3, nCols> reprojectPoints(AffineTransform<Pos_T> const& T,
											  Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1,
											  Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {

	return reprojectPoints(T.R, T.t, pt_cam_1, pt_cam_2);
}

template <typename Pos_T, int nCols>
/*!
 * \brief reprojectPointsLstSqr is a more robust but more expensive reprojection routine to compute points reprojections (compared to reprojectPoints)
 * \param R the rotation part of the transform cam1 2 cam2
 * \param t the translation part of the transform cam1 2 cam2
 * \param pt_cam_1 points in cam1 homogeneous coordinates
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the points coordinates in cam1 frame
 */
Eigen::Array<Pos_T, 3, nCols> reprojectPointsLstSqr(Eigen::Matrix<Pos_T, 3, 3> const& R,
									  Eigen::Matrix<Pos_T, 3, 1> const& t,
									  Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1,
									  Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {

	typedef Eigen::Matrix<Pos_T, 3, 2> MatrixAtype;

	int nPts = pt_cam_1.cols();

	if (pt_cam_1.cols() != pt_cam_2.cols()) {
		throw GeometricException("Points arrays of different dimensions provided");
	}

	Eigen::Array<Pos_T, 3, nCols> reproj;

	reproj.resize(3,nPts);
	reproj.topRows(2) = pt_cam_1;
	reproj.bottomRows(1).setOnes();

	for (int i = 0; i < nPts; i++) {
		Eigen::Matrix<Pos_T, 3, 1> v2;
		v2.template block<2,1>(0,0) = pt_cam_2.col(i);
		v2[2] = 1;
		Eigen::Matrix<Pos_T, 3, 1> v2C1 = R.transpose()*v2;
		MatrixAtype A;
		A.col(0) = reproj.col(i);
		A.col(1) = -v2C1;

		Eigen::Matrix<Pos_T, 2, 2> invQxx = A.transpose()*A;

		auto svd = invQxx.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

		Eigen::Matrix<Pos_T, 2, 2> pseudoInverse = svd.matrixV() * (svd.singularValues().array().abs() > 1e-4).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().transpose();

		Eigen::Matrix<Pos_T, 2, 1> lambdas = pseudoInverse*A.transpose()*(-R.transpose()*t);

		Eigen::Matrix<Pos_T, 3, 1> est1 = A.col(0)*lambdas[0];
		Eigen::Matrix<Pos_T, 3, 1> est2 = -A.col(1)*lambdas[1] -R.transpose()*t;

		reproj.col(i) = (est1 + est2)/2.;
	}

	return reproj;

}
template <typename Pos_T, int nCols>
/*!
 * \brief reprojectPointsLstSqr is a more robust but more expensive reprojection routine to compute points reprojections (compared to reprojectPoints)
 * \param T the transform cam1 to cam2
 * \param pt_cam_1 points in cam1 homogeneous coordinates
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the points coordinates in cam1 frame
 */
Eigen::Array3Xf reprojectPointsLstSqr(AffineTransform<Pos_T> const& T,
									  Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1,
									  Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {
	return reprojectPointsLstSqr(T.R, T.t, pt_cam_1, pt_cam_2);
}

template <typename Pos_T, int nCols>
/*!
 * \brief estimateEssentialMatrix estimate the essential matrix between a pair of cameras
 * \param pt_cam_1 points in cam1 homogeneous coordinates (must be at least 8 points)
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the essential matrix.
 */
Eigen::Matrix<Pos_T, 3, 3> estimateEssentialMatrix(Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1, Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {

	typedef Eigen::Matrix<Pos_T, 9, nCols> MatrixFtype;

	int nPts = pt_cam_1.cols();

	if (nPts < 8 or pt_cam_1.cols() != pt_cam_2.cols()) {
		throw GeometricException("Points arrays of different dimensions provided");
	}

	MatrixFtype F;
	F.resize(9, nPts);

	F.row(0) = pt_cam_2.row(0)*pt_cam_1.row(0);
	F.row(1) = pt_cam_2.row(0)*pt_cam_1.row(1);
	F.row(2) = pt_cam_2.row(0);
	F.row(3) = pt_cam_2.row(1)*pt_cam_1.row(0);
	F.row(4) = pt_cam_2.row(1)*pt_cam_1.row(1);
	F.row(5) = pt_cam_2.row(1);
	F.row(6) = pt_cam_1.row(0);
	F.row(7) = pt_cam_1.row(1);
	F.row(8).setOnes();

	auto svd = F.jacobiSvd(Eigen::ComputeFullU);
	auto e = svd.matrixU().col(8);
	Eigen::Matrix<Pos_T, 3, 3> E;
	E << e[0], e[1], e[2],
		 e[3], e[4], e[5],
		 e[6], e[7], e[8];

	return E;
}

template <typename Pos_T, int nCols>
/*!
 * \brief estimatePerspectiveTransformMatrix estimate the perspective transform matrix between two sets of points
 * \param pt_cam_1 points from first coordinate system
 * \param pt_cam_2 same points in the second coordinate system
 * \return the perspective transform matrix.
 */
Eigen::Matrix<Pos_T, 3, 3> estimatePerspectiveTransformMatrix(Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1, Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {

    using ParamVecT = Eigen::Matrix<Pos_T, Eigen::Dynamic,1>;
    using VecbT = Eigen::Matrix<Pos_T, Eigen::Dynamic,1>;
    using MatrixAT = Eigen::Matrix<Pos_T, Eigen::Dynamic,Eigen::Dynamic>;

    int nPts = pt_cam_1.cols();

    if (pt_cam_1.cols() != pt_cam_2.cols()) {
        throw GeometricException("Points arrays of different dimensions provided");
    }

    int nParams = 9 + nPts-1; //9 parameters for the matrix, one scaling parameter for every point except the first where scaling is set to 1 to lift ambiguity
    int nObs = 3*nPts;

    MatrixAT A;
    A.resize(nObs, nParams);
    A.setConstant(0);
    VecbT b;
    b.resize(nObs);
    b.setConstant(0);

    for (int i = 0; i < nPts; i++) {

        A(3*i,0) = pt_cam_1(0,i);
        A(3*i,1) = pt_cam_1(1,i);
        A(3*i,2) = 1;

        A(3*i+1,3) = pt_cam_1(0,i);
        A(3*i+1,4) = pt_cam_1(1,i);
        A(3*i+1,5) = 1;

        A(3*i+2,6) = pt_cam_1(0,i);
        A(3*i+2,7) = pt_cam_1(1,i);
        A(3*i+2,8) = 1;

        if (i == 0) { //first point
            b[i] = pt_cam_2(0,i);
            b[i+1] = pt_cam_2(1,i);
            b[i+2] = 1;
        } else {
            //scaling parameters
            A(3*i,8+i) = -pt_cam_2(0,i);
            A(3*i+1,8+i) = -pt_cam_2(1,i);
            A(3*i+2,8+i) = -1;
        }
    }

    ParamVecT opt = Optimization::leastSquares(A,b);

    Eigen::Matrix<Pos_T, 3, 3> ret;

    ret(0,0) = opt[0];
    ret(0,1) = opt[1];
    ret(0,2) = opt[2];

    ret(1,0) = opt[3];
    ret(1,1) = opt[4];
    ret(1,2) = opt[5];

    ret(2,0) = opt[6];
    ret(2,1) = opt[7];
    ret(2,2) = opt[8];

    return ret;
}

template <typename Pos_T>
/*!
 * \brief essentialMatrix2Transforms extract a pair of possible transforms from an essential matrix
 * \param E the essential matrix
 * \return a pair of possible transforms that can be used in the selectTransform function.
 */
std::pair<AffineTransform<Pos_T>, AffineTransform<Pos_T>> essentialMatrix2Transforms(Eigen::Matrix<Pos_T, 3, 3> const& E) {

	auto svd = E.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix<Pos_T, 3, 3> U = svd.matrixU();
	Eigen::Matrix<Pos_T, 3, 3> V = svd.matrixV();

	//ensure both U and V are rotation matrices.
	if (U.determinant() < 0) {
		U = -U;
	}

	if (V.determinant() < 0) {
		V = -V;
	}

	Eigen::Matrix<Pos_T, 3, 3> W = Eigen::Matrix<Pos_T, 3, 3>::Zero();
	W(1,0) = -1;
	W(0,1) = 1;
	W(2,2) = 1;

	AffineTransform<Pos_T> T1;
	AffineTransform<Pos_T> T2;

	T1.R = U*W*V.transpose();
	T2.R = U*W.transpose()*V.transpose();

	W(2,2) = 0;

	Eigen::Matrix<Pos_T,3,3> tmp = U*W*U.transpose();
	T1.t = unskew(tmp);
	T2.t = -T1.t;

	return {T1, T2};

}

template <typename Pos_T, int nCols>
/*!
 * \brief selectTransform from a pair of possible transforms extracted from the essential matrix, extract the transform which puts all points in front of both cameras
 * \param T1 a possible transform
 * \param T2 another possible transform, with different rotation and translation
 * \param pt_cam_1 the points used to estimate E, in cam1 homogeneous coordinates
 * \param pt_cam_2 the points used to estimate E, in cam2 homogeneous coordinates
 * \return The transform which reproject all points in front of both cameras (might be T1, T2 or a mix of both).
 */
AffineTransform<Pos_T> selectTransform(AffineTransform<Pos_T> const& T1,
									   AffineTransform<Pos_T> const& T2,
									   Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1,
									   Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {

	int nPts = pt_cam_1.cols();

	AffineTransform<Pos_T> Rs[4];

	int count_good = 0;

	for (auto const& R_cand : {T1.R, T2.R}) {

		for (auto const& t_cand : {T1.t, T2.t}) {

			Eigen::Array<Pos_T, 3, nCols> reproj = reprojectPoints(R_cand, t_cand, pt_cam_1, pt_cam_2);

			if ((reproj.bottomRows(1) >= 0.).all()) {
				Eigen::Matrix<Pos_T,3,3> R = R_cand.transpose();
				Eigen::Matrix<Pos_T,3,1> t = -R_cand.transpose()*t_cand;
				reproj = reprojectPoints(R, t, pt_cam_2, pt_cam_1);

				if ((reproj.bottomRows(1) >= 0.).all()) {

					Rs[count_good].R = R_cand;
					Rs[count_good].t = t_cand;

					count_good++;
				}
			}

		}
	}

	if (count_good == 0) {
		throw GeometricException("No valid transforms has been found");
	}

	if (count_good > 1) {

		int selected = -1;
		float s_error = std::numeric_limits<float>::infinity();

		for (int i = 0; i < count_good; i++) {

			Eigen::Array<Pos_T, 3, nCols> reproj = reprojectPoints(Rs[i], pt_cam_1, pt_cam_2);
			Eigen::Array<Pos_T, 2, nCols> onOtherIm = projectPoints(reproj, Rs[i]);

			float c_error = (onOtherIm - pt_cam_2).matrix().norm()/nPts;

			Eigen::Matrix<Pos_T, 3, 3> R = Rs[i].R.transpose();
			Eigen::Matrix<Pos_T, 3, 1> t =  -Rs[i].R.transpose()*Rs[i].t;
			reproj = reprojectPoints(R, t, pt_cam_2, pt_cam_1);

			onOtherIm = projectPoints(reproj, R, t);

			c_error += (onOtherIm - pt_cam_1).matrix().norm()/nPts;

			if (c_error < s_error) {
				selected = i;
				s_error = c_error;
			}
		}

		if (selected < 0) {
			throw GeometricException("No valid transforms has been found");
		}

		return Rs[selected];

	}

	return Rs[0];

}

template <typename Pos_T, int nCols>
/*!
 * \brief essentialMatrix2Transform extract the transformation cam1 to cam2 from the estimated essential matrix
 * \param E the essential matrix
 * \param pt_cam_1 the points used to estimate E, in cam1 homogeneous coordinates
 * \param pt_cam_2 the points used to estimate E, in cam2 homogeneous coordinates
 * \return the transformation cam1 to cam2
 */
AffineTransform<Pos_T> essentialMatrix2Transform(Eigen::Matrix<Pos_T, 3, 3> const& E,
												 Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1,
												 Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {
	auto candidates = essentialMatrix2Transforms(E);
	return selectTransform(candidates.first, candidates.second, pt_cam_1, pt_cam_2);
}

template <typename Pos_T, int nCols>
/*!
 * \brief findTransform jointly build the essential matrix and extract the corresponding transform from it
 * \param pt_cam_1 points in cam1 homogeneous coordinates (must be at least 8 points)
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the transform from cam1 to cam2
 */
AffineTransform<float> findTransform(Eigen::Array<Pos_T, 2, nCols> const& pt_cam_1,
									 Eigen::Array<Pos_T, 2, nCols> const& pt_cam_2) {
	Eigen::Matrix<Pos_T,3,3> E = estimateEssentialMatrix(pt_cam_1, pt_cam_2);
	return essentialMatrix2Transform(E, pt_cam_1, pt_cam_2);
}



template<typename T>
struct p3pSolution {
    std::array<AffineTransform<T>, 4> solutions;
    int nValidSolutions;
};

template<typename T, int nRefineIteration = 5>
p3pSolution<T> p3p(Eigen::Array<T,3,3> const& pt_cam, Eigen::Array<T,3,3> const& pt_world);

class p3pInternals {
private:

    /*!
     * \brief solveCubic give one root of the cubic (up to numerical precision)
     * \param a the cube coefficient of the cubic
     * \param b the square coefficient of the cubic
     * \param c the linear coefficient of the cubic
     * \param d the constant coefficient of the cubic
     * \param x0 the initial solution to start from
     * \param tol the numerical tolerance of the solver
     * \param maxitem the maximal number of iterations
     * \return a root, or nan in case of error
     */
    template<typename T>
    inline static T solveCubic(T a, T b, T c, T d, T x0 = 0, T tol = 1e-6, int maxIter = 1000) {

        static_assert (std::is_floating_point_v<T>, "Function accept only a floating point number");

        T ret = x0;

        int i = 0;
        for (i = 0; i < maxIter; i++) {
            T fx = a*Math::iPow<3>(ret) + b*Math::iPow<2>(ret) + c*ret + d;
            T dfx = 3*a*Math::iPow<2>(ret) + 2*b*ret + c;
            ret = ret - fx/dfx;

            if (std::abs(fx) < tol) {
                break;
            }
        }

        if (i == maxIter) {
            return std::nan("");
        }

        return ret;
    }

    template<typename T>
	inline static Eigen::Matrix<T,3, 1> getEigenVectorKnowing0(Eigen::Matrix<T,9,1> const& flat, T lambda) {

        T c = c = lambda*lambda + flat[0]*flat[4] - lambda*(flat[0]+flat[4]) - flat[1]*flat[1];

        T a1 = (lambda*flat[2] + flat[1]*flat[5] - flat[2]*flat[4])/c;
        T a2 = (lambda*flat[5] + flat[1]*flat[2] - flat[0]*flat[5])/c;

		Eigen::Matrix<T,3,1> ret;
        ret << a1, a2, 1;
        ret.normalize();

        return ret;
    }

    template<typename T>
    inline static std::tuple<Eigen::Matrix<T,3,3>, T, T> EigenDecompositionKnowing0(Eigen::Matrix<T,3,3> const& mat) {

        using Mat = Eigen::Matrix<T,3,3>;
        using Vec = Eigen::Matrix<T,3,1>;

        Vec b3 = mat.row(1).cross(mat.row(2)); //0-eigenvector
        b3.normalize();

		Eigen::Matrix<T,9,1> flat;
        flat << mat(0,0), mat(0,1), mat(0,2),
                mat(1,0), mat(1,1), mat(1,2),
                mat(2,0), mat(2,1), mat(2,2);

        T p1 = -flat[0] - flat[4] - flat[8];
        T p0 = -flat[1]*flat[1] - flat[2]*flat[2] - flat[5]*flat[5] + flat[0]*(flat[4] + flat[8]) + flat[4]*flat[8];

        T delta = p1*p1 - 4*p0;
        T sqrtDelta = std::sqrt(delta);

        T sigma1;
        T sigma2;

        if (p1 < 0) {
            sigma1 = (-p1 + sqrtDelta)/2;
            sigma2 = (-p1 - sqrtDelta)/2;
        } else {
            sigma1 = 2*p0/(-p1 + sqrtDelta);
            sigma2 = 2*p0/(-p1 - sqrtDelta);
        }

        Vec b1 = getEigenVectorKnowing0(flat, sigma1);
        Vec b2 = getEigenVectorKnowing0(flat, sigma2);

        if (std::abs(sigma1) > std::abs(sigma2)) {
            Mat ret;
            ret.col(0) = b1;
            ret.col(1) = b2;
            ret.col(2) = b3;

            return std::make_tuple(ret, sigma1, sigma2);
        }

        Mat ret;
        ret.col(0) = b2;
        ret.col(1) = b1;
        ret.col(2) = b3;

        return std::make_tuple(ret, sigma2, sigma1);

    }

    template<typename T, int nRefineIteration>
	friend p3pSolution<T> p3p(Eigen::Array<T,3,3> const& pt_cam, Eigen::Array<T,3,3> const& pt_world);

};


/*!
 * \brief p3p compute a set of solutions to the p3p problem
 * \param pt_cam the homogeneous coordinates in the camera frame
 * \param pt_world the coordinates in the world or scene frame
 * \return a set of up to 4 solutions
 *
 * Code derived from the paper https://openaccess.thecvf.com/content_ECCV_2018/papers/Mikael_Persson_Lambda_Twist_An_ECCV_2018_paper.pdf
 */
template<typename T, int nRefineIteration>
p3pSolution<T> p3p(Eigen::Array<T,3,3> const& pt_cam, Eigen::Array<T,3,3> const& pt_world) {

    using Vec = Eigen::Matrix<T,3,1>;
    using Mat = Eigen::Matrix<T,3,3>;

    Vec x1 = pt_world.col(0);
    Vec x2 = pt_world.col(1);
    Vec x3 = pt_world.col(2);

    Vec y1 = pt_cam.col(0);
    Vec y2 = pt_cam.col(1);
    Vec y3 = pt_cam.col(2);

    y1.normalize();
    y2.normalize();
    y3.normalize();

    //Compute the aij and bij
    Vec diff12=x1-x2;
    Vec diff13=x1-x3;
    Vec diff23=x2-x3;
    Vec diff12xdiff23(diff12.cross(diff23));

    Mat X;
    X.col(0) = diff12;
    X.col(1) = diff23;
    X.col(2) = diff12xdiff23;

    Mat invX = X.inverse();

    T a12 = diff12.squaredNorm();
    T a13 = diff13.squaredNorm();
    T a23 = diff23.squaredNorm();

    T b12 = y1.dot(y2);
    T b13 = y1.dot(y3);
    T b23 = y2.dot(y3);

    //compute D1 and D2

    Mat M12;
    M12 << 1, -b12, 0,
          -b12, 1, 0,
           0, 0, 0;
    Mat M13;
    M13 << 1, 0, -b13,
           0, 0, 0,
          -b13, 0, 1;
    Mat M23;
    M23 << 0, 0, 0,
           0, 1, -b23,
           0, -b23, 1;

    Mat D1 = M12*a23 - M23*a12;
    Mat D2 = M13*a23 - M23*a13;

    Vec d11 = D1.col(0);
    Vec d12 = D1.col(1);
    Vec d13 = D1.col(2);

    Vec d21 = D2.col(0);
    Vec d22 = D2.col(1);
    Vec d23 = D2.col(2);

    //compute a real root to the cubic equation
    T c3 = D2.determinant();
    T c2 = d11.dot(d22.cross(d23)) + d12.dot(d23.cross(d21)) + d13.dot(d21.cross(d22));
    T c1 = d21.dot(d12.cross(d13)) + d22.dot(d13.cross(d11)) + d23.dot(d11.cross(d12));
    T c0 = D1.determinant();

    T gamma0 = p3pInternals::solveCubic<double>(c3, c2, c1, c0);

    Mat D0 = D1 + gamma0*D2;

    //Diagonalize D0
    Mat E;
    T sigma1;
    T sigma2;

    std::tie(E, sigma1, sigma2) = p3pInternals::EigenDecompositionKnowing0(D0);

    //compute possible s

    T s_p = std::sqrt(-sigma2/sigma1);
    T s_n = -s_p;

    //compute the tau_k
    p3pSolution<T> sol;
    sol.nValidSolutions = 0;

	Eigen::Matrix<T,9,1> flatE;
    flatE << E(0,0), E(0,1), E(0,2),
             E(1,0), E(1,1), E(1,2),
             E(2,0), E(2,1), E(2,2);


    for (int i = 0; i < 2; i++) {
        T s = (i == 0) ? s_p : s_n;

        T w0 = (flatE[3] - s*flatE[4])/(s*flatE[1] - flatE[0]);
        T w1 = (flatE[6] - s*flatE[7])/(s*flatE[1] - flatE[0]);

        T a = ((a13 - a12)*w1*w1 + 2*a12*b13*w1 - a12);
        T b = (2*a12*b13*w0 - 2*a13*b12*w1 - 2*w0*w1*(a12 - a13));
        T c = ((a13 - a12)*w0*w0 - 2*a13*b12*w0 + a13);

        T delta = b*b - 4*a*c;

        if (delta < 0) {
            continue;
        }

        std::array<T, 2> tks;
        int nTks;

        if (delta <= std::numeric_limits<T>::min()) {
            nTks = 1;
            tks[0] = -b/(2*a);
        } else {
            nTks = 2;
            T sqrtDelta = std::sqrt(delta);
            tks[0] = (-b+sqrtDelta)/(2*a);
            tks[1] = (-b-sqrtDelta)/(2*a);
        }

        for (int j = 0; j < nTks; j++) {
            T tk = tks[j];

            if (tk < 0 or !std::isfinite(tk)) {
                continue;
            }

            Vec Lambda;

            T l2sqr = a23/(tk*(tk-2*b23)+1);

            if (l2sqr < 0) {
                continue;
            }

            Lambda[1] = std::sqrt(l2sqr);
            Lambda[2] = tk*Lambda[1];

            Lambda[0] = w0*Lambda[1] + w1*Lambda[2];

            //refined using Gauss/Newtown
            for (int i = 0; i < nRefineIteration; i++) {

                Vec func(Lambda.transpose()*M12*Lambda - a12, Lambda.transpose()*M13*Lambda - a13, Lambda.transpose()*M23*Lambda - a23);

                Mat Jacobian;
                Jacobian.row(0) = 2*Lambda.transpose()*M12;
                Jacobian.row(1) = 2*Lambda.transpose()*M13;
                Jacobian.row(2) = 2*Lambda.transpose()*M23;

                Vec delta = Jacobian.fullPivHouseholderQr().solve(-func);

                if (!delta.array().isFinite().all()) {
                    break;
                }

                Lambda += delta;

            }

            if (Lambda[0] < 0) {
                continue;
            }

            if (Lambda[1] < 0) {
                continue;
            }

            if (Lambda[2] < 0) {
                continue;
            }

            // compute the rotation:
            Vec ry1 = y1*Lambda[0];
            Vec ry2 = y2*Lambda[1];
            Vec ry3 = y3*Lambda[2];

            Vec yd12 = ry1-ry2;
            Vec yd23 = ry2-ry3;
            Vec yd12xd23 = yd12.cross(yd23);

            Mat Y;
            Y << yd12(0), yd23(0), yd12xd23(0),
                 yd12(1), yd23(1), yd12xd23(1),
                 yd12(2), yd23(2), yd12xd23(2);


            Mat R = Y*invX;
            Vec t = (ry1 - R*x1);

            sol.solutions[sol.nValidSolutions] = AffineTransform<T>(R, t);
            sol.nValidSolutions++;
        }

    }

    return sol;
}

/*!
 * \brief p4p solve the pnp problem with the minimal number of non-ambiguous points (4)
 * \param pt_cam the homogeneous coordinates in the camera frame
 * \param pt_world the coordinates in the world or scene frame
 * \return a single solution
 */
template<typename T>
AffineTransform<T> p4p(Eigen::Array<T,2,4> const& pt_cam, Eigen::Array<T,3,4> const& pt_world) {

    //The function seem a bit unstable: some point configurations might lead to ambiguous solutions ->
    //TODO: test and allow the function to use more than a fourth point for computing the error.
    static_assert (std::is_floating_point_v<T>, "P4P function support only floating point numbers template type");

    using Mat = Eigen::Matrix<T,3,3>;
	using Vec = Eigen::Matrix<T,3,1>;

    Eigen::Array<T,3,3> homPtCam;
    homPtCam.template block<2,3>(0,0) = pt_cam.template block<2,3>(0,0);
    homPtCam.template block<1,3>(2,0) = Eigen::Array<T,1,3>::Constant(1);

	p3pSolution<T> p3pSolution = p3p<T>(homPtCam, pt_world.template block<3,3>(0,0));

    Vec imgPt;
    imgPt.template block<2,1>(0,0) = pt_cam.col(3);
    imgPt[2] = 1;
    Vec worldPt = pt_world.col(3);
    AffineTransform<T> P; // identity
    double err0=std::numeric_limits<double>::max();

    for(int i = 0; i < p3pSolution.nValidSolutions; i++) {

        AffineTransform<T> const& tmp = p3pSolution.solutions[i];

        Mat test = tmp.R*tmp.R.transpose();
        Mat closure = test - Mat::Identity();
        if (closure.norm() > ((std::is_same_v<T, float>) ? 1e-3 : 1e-6)) {
            continue;
        }

        Vec reproj = tmp*worldPt;

        if(reproj[2]<0) {
            continue;
        }

        reproj /= reproj[2];

        T err = (reproj-imgPt).squaredNorm();

        if (std::isnan(err)) {
            continue;
        }

        if (err < err0 ){
            P=tmp;
            err0 = err;
        }
    }

    return P;
}

/*!
 * \brief JacobianPointProjection gives the derivative of the projection of a point
 * \param currentWorld2Cam the current transform from world frame to camera frame
 * \param pt_world the point to project
 * \return the derivative of the projection of the point (in homogeneous coordinate) as a function of the six parameters of the shape preserving transform
 * (not including the scale).
 *
 * the first three columns are for the rx, ry, rz rotation parameters, the last three column for the tx, ty, tz translation parameters.
 */
template<typename T>
Eigen::Matrix<T, 2, 6> JacobianPointProjection(ShapePreservingTransform<T> const& currentWorld2Cam,
											   Eigen::Matrix<T,3,1> const& pt_world) {

    using Mat36 = Eigen::Matrix<T,3,6>;
    using Mat33 = Eigen::Matrix<T,3,3>;
    using Mat26 = Eigen::Matrix<T,2,6>;
    using Mat23 = Eigen::Matrix<T,2,3>;
    using Mat22 = Eigen::Matrix<T,2,2>;

	using Vec3 = Eigen::Matrix<T,3,1>;
	using Vec2 = Eigen::Matrix<T,2,1>;

    Mat33 diffRx = diffRodriguez(currentWorld2Cam.r, Axis::X);
    Mat33 diffRy = diffRodriguez(currentWorld2Cam.r, Axis::Y);
    Mat33 diffRz = diffRodriguez(currentWorld2Cam.r, Axis::Z);

    Mat36 jacobianFrameChange;

    jacobianFrameChange.col(0) = diffRx*pt_world;
    jacobianFrameChange.col(1) = diffRy*pt_world;
    jacobianFrameChange.col(2) = diffRz*pt_world;

    jacobianFrameChange.template block<3,3>(0,3) = Mat33::Identity();

    Vec3 pt_camera = currentWorld2Cam*pt_world;

    Mat23 jacobianHomenegous;
    jacobianHomenegous.template block<2,2>(0,0) = Mat22::Identity()/pt_camera.z();
    jacobianHomenegous.col(2) = -pt_camera.template block<2,1>(0,0)/Math::iPow<2>(pt_camera.z());

    Mat26 ret = jacobianHomenegous*jacobianFrameChange;
    return ret;

}

template<typename T>
AffineTransform<T> pnpRefine(Eigen::Array<T,2,Eigen::Dynamic> const& pt_cam,
                             Eigen::Array<T,3,Eigen::Dynamic> const& pt_coord,
                             ShapePreservingTransform<T> const& initialScene2Cam,
                             int nIterations = 500,
                             T incrThreshold = 1e-6) {


    using MatrixAT = Eigen::Matrix<T,Eigen::Dynamic,6>;

	using VecObsT = Eigen::Matrix<T,Eigen::Dynamic,1>;
	using VecParamT = Eigen::Matrix<T,6,1>;

    using MatrixQT = Eigen::Matrix<T,6,6>;

    using Mat33 = Eigen::Matrix<T,3,3>;

	using Vec3 = Eigen::Matrix<T,3,1>;

    int nPts = pt_cam.cols();
    int nObs = 2*nPts;

    AffineTransform<T> invalid(Mat33::Zero(), Vec3::Zero());

    if (pt_coord.cols() != nPts) {
        return invalid;
    }

    MatrixAT A;
    A.resize(nObs,6);

    VecObsT lBulle;
    lBulle.resize(nObs);

    VecObsT obs;
    obs.resize(nObs);

    for(int i = 0; i < nPts; i++) {
        obs.template block<2,1>(2*i,0) = pt_cam.col(i).matrix();
    }

    ShapePreservingTransform<T> current = initialScene2Cam;

    for(int i = 0; i < nIterations; i++) {

        Eigen::Array<T,2,Eigen::Dynamic> pt_currentHomogeneous = projectPoints(pt_coord, current.toAffineTransform());

        for(int i = 0; i < nPts; i++) {
            A.template block<2,6>(2*i,0) = JacobianPointProjection<T>(current, pt_coord.col(i).matrix());
            lBulle.template block<2,1>(2*i,0) = pt_currentHomogeneous.col(i).matrix();
        }

        VecObsT error = obs - lBulle;

        MatrixQT invQxx = A.transpose()*A;

        VecParamT deltaX = invQxx.fullPivHouseholderQr().solve(A.transpose()*(error));

        if (!deltaX.array().isFinite().all()) {
            //convergence error
            return invalid;
        }

        current.r += deltaX.template block<3,1>(0,0);
        current.t += deltaX.template block<3,1>(3,0);

        if (deltaX.norm() < incrThreshold) {
            break;
        }

    }

    return current.toAffineTransform();

}

/*!
 * \brief pnp compute the estimated target to cam transform from projected points
 * \param pt_cam 2D points in homogeneous normalized image coordinates
 * \param pt_coords the 3D points in target frame coordinates
 * \return the affine transform corresponding to the target to cam transform
 */
template<typename T, int nCols>
AffineTransform<T> pnp(Eigen::Array<T,2,nCols> const& pt_cam,
                       Eigen::Array<T,3,nCols> const& pt_coords) {


    using Mat33 = Eigen::Matrix<T,3,3>;
	using Vec3 = Eigen::Matrix<T,3,1>;

    //treat invalid cases

    AffineTransform<T> invalid(Mat33::Zero(), Vec3::Zero());

    if (pt_cam.cols() < 4 or pt_cam.cols() != pt_coords.cols()) {
        return invalid;
    }

    int nPts = pt_cam.cols();

    //points used to compute the
    Eigen::Array<T,2,4> base_pt_cam;
    Eigen::Array<T,3,4> base_pt_coords;

    //pick the points the most futher appart in the set

    int idx0 = 0;
    int idx1 = 1;

    T dist = (pt_cam.col(idx0).matrix() - pt_cam.col(idx1).matrix()).norm();

    for (int i = 0; i < nPts; i++) {
        for (int j = i+1; j < nPts; j++) {
            T cand_dist = (pt_cam.col(i).matrix() - pt_cam.col(j).matrix()).norm();

            if (cand_dist > dist) {
                dist = cand_dist;
                idx0 = i;
                idx1 = j;
            }
        }
    }

    int idx2 = 0;
    dist = 0;

    for (int i = 0; i < nPts; i++) {
        T c_dist0 = (pt_cam.col(i).matrix() - pt_cam.col(idx0).matrix()).norm();
        T c_dist1 = (pt_cam.col(i).matrix() - pt_cam.col(idx1).matrix()).norm();

        T c_dist = std::min(c_dist0, c_dist1);

        if (c_dist > dist) {
            idx2 = i;
        }
    }

    int idx3 = 0;
    dist = 0;

    for (int i = 0; i < nPts; i++) {
        T c_dist0 = (pt_cam.col(i).matrix() - pt_cam.col(idx0).matrix()).norm();
        T c_dist1 = (pt_cam.col(i).matrix() - pt_cam.col(idx1).matrix()).norm();
        T c_dist2 = (pt_cam.col(i).matrix() - pt_cam.col(idx2).matrix()).norm();

        T c_dist = std::min(c_dist0, std::min(c_dist1, c_dist2));

        if (c_dist > dist) {
            idx2 = i;
        }
    }

    base_pt_cam.col(0) = pt_cam.col(idx0);
    base_pt_cam.col(1) = pt_cam.col(idx1);
    base_pt_cam.col(2) = pt_cam.col(idx2);
    base_pt_cam.col(3) = pt_cam.col(idx3);

    base_pt_coords.col(0) = pt_coords.col(idx0);
    base_pt_coords.col(1) = pt_coords.col(idx1);
    base_pt_coords.col(2) = pt_coords.col(idx2);
    base_pt_coords.col(3) = pt_coords.col(idx3);

    AffineTransform<T> initial = p4p(base_pt_cam, base_pt_coords);
    ShapePreservingTransform<T> initialRigid = ShapePreservingTransform<T>(inverseRodriguezFormula(initial.R), initial.t, 1);

    return pnpRefine(pt_cam, pt_coords, initialRigid);
}

/*!
 * \brief pnp compute the estimated target to cam transform from projected points
 * \param pt_cam 2D points in homogeneous normalized image coordinates
 * \param idxs indices of the points the 2D landmarks correspond to
 * \param pt_coords matching 3D points in target frame coordinates
 * \return the affine transform corresponding to the target to cam transform
 */
template<typename T, int nCols, int nColsCoords>
AffineTransform<T> pnp(Eigen::Array<T,2,nCols> const& pt_cam,
                       std::vector<int> const& idxs,
                       Eigen::Array<T,3,nColsCoords> const& pt_coords) {


    using Mat33 = Eigen::Matrix<T,3,3>;
	using Vec3 = Eigen::Matrix<T,3,1>;

    //treat invalid cases

    AffineTransform<T> invalid(Mat33::Zero(), Vec3::Zero());

    if (pt_cam.cols() < 4 or pt_cam.cols() != idxs.size()) {
        return invalid;
    }

    Eigen::Array<T,3,nCols> pt_coords_new;

    if (nCols == Eigen::Dynamic) {
        pt_coords_new.resize(3, pt_cam.cols());
    }

    for(int i = 0; i < pt_cam.cols(); i++) {
        pt_coords_new.col(i) = pt_coords.col(idxs[i]);
    }

    return pnp(pt_cam, pt_coords_new);

}

template <typename Pos_T>
/*!
 * \brief The imageToImageReprojector class represent the coordinate on a point in a target image
 */
class ImageToImageReprojector {

public:

	ImageToImageReprojector(Eigen::Matrix<Pos_T, 2,1> const& source_f,
							Eigen::Matrix<Pos_T, 2,1> const& source_pp,
							Eigen::Matrix<Pos_T, 2,1> const& target_f,
							Eigen::Matrix<Pos_T, 2,1> const& target_pp,
							AffineTransform<Pos_T> const& sourceToTarget,
							ImageAnchors source_imageOrigin = ImageAnchors::TopLeft,
							ImageAnchors target_imageOrigin = ImageAnchors::TopLeft) :
		_source_f(source_f),
		_source_pp(source_pp),
		_target_f(target_f),
		_target_pp(target_pp),
		_sourceToTarget(sourceToTarget),
		_source_imageOrigin(source_imageOrigin),
		_target_imageOrigin(target_imageOrigin)
	{

	}

	ImageToImageReprojector(Eigen::Matrix<Pos_T, 2,1> const& source_f,
							 Eigen::Matrix<Pos_T, 2,1> const& source_pp,
							 Pos_T target_f,
							 Eigen::Matrix<Pos_T, 2,1> const& target_pp,
							 AffineTransform<Pos_T> const& sourceToTarget,
							 ImageAnchors source_imageOrigin = ImageAnchors::TopLeft,
							 ImageAnchors target_imageOrigin = ImageAnchors::TopLeft) :
		 ImageToImageReprojector(source_f,
								 source_pp,
								 Eigen::Matrix<Pos_T, 2,1>(target_f, target_f),
								 target_pp,
								 sourceToTarget,
								 source_imageOrigin,
								 target_imageOrigin)
	 {

	 }

	ImageToImageReprojector(Pos_T source_f,
							 Eigen::Matrix<Pos_T, 2,1> const& source_pp,
							 Eigen::Matrix<Pos_T, 2,1> const& target_f,
							 Eigen::Matrix<Pos_T, 2,1> const& target_pp,
							 AffineTransform<Pos_T> const& sourceToTarget,
							 ImageAnchors source_imageOrigin = ImageAnchors::TopLeft,
							 ImageAnchors target_imageOrigin = ImageAnchors::TopLeft) :
		 ImageToImageReprojector(Eigen::Matrix<Pos_T, 2,1>(source_f, source_f),
								 source_pp,
								 target_f,
								 target_pp,
								 sourceToTarget,
								 source_imageOrigin,
								 target_imageOrigin)
	 {

	 }

	ImageToImageReprojector(Pos_T source_f,
							 Eigen::Matrix<Pos_T, 2,1> const& source_pp,
							 Pos_T target_f,
							 Eigen::Matrix<Pos_T, 2,1> const& target_pp,
							 AffineTransform<Pos_T> const& sourceToTarget,
							 ImageAnchors source_imageOrigin = ImageAnchors::TopLeft,
							 ImageAnchors target_imageOrigin = ImageAnchors::TopLeft) :
		 ImageToImageReprojector(Eigen::Matrix<Pos_T, 2,1>(source_f, source_f),
								 source_pp,
								 Eigen::Matrix<Pos_T, 2,1>(target_f, target_f),
								 target_pp,
								 sourceToTarget,
								 source_imageOrigin,
								 target_imageOrigin)
	 {

	 }

	virtual Eigen::Matrix<Pos_T, 2,1> reprojected(Eigen::Matrix<Pos_T, 2,1> const& sourcePos, Pos_T depth) {
		Eigen::Matrix<Pos_T, 3,1> coordsSource;
		coordsSource.template block<2,1>(0,0) = Image2HomogeneousCoordinates(sourcePos, _source_f, _source_pp, _source_imageOrigin);
		coordsSource[2] = 1;

		coordsSource *= depth;


		Eigen::Matrix<Pos_T, 3,1> coordsTarget = _sourceToTarget*coordsSource;

		Eigen::Matrix<Pos_T, 2,1> proj = projectPoints(coordsTarget);

		return Homogeneous2ImageCoordinates(proj, _target_f, _target_pp, _target_imageOrigin);

	}

	Eigen::Matrix<Pos_T, 2,1> operator()(Eigen::Matrix<Pos_T, 2,1> const& sourcePos, Pos_T depth) {
		return reprojected(sourcePos, depth);
	}

	template<int inDims>
	Multidim::Array<Pos_T, inDims+1> reprojected(Multidim::Array<Pos_T, inDims+1> const& sourcePos, Multidim::Array<Pos_T, inDims> depth) {

		if (sourcePos.shape()[inDims] != 2) {
			return Multidim::Array<Pos_T, inDims+1>();
		}

		for (int i = 0; i < inDims; i++) {
			if (sourcePos.shape()[i] != depth.shape()[i]) {
				return Multidim::Array<Pos_T, inDims+1>();
			}
		}

		Multidim::Array<Pos_T, inDims+1> outPos(sourcePos.shape());

		Multidim::IndexConverter<inDims> idxConverter(depth.shape());

		for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
			std::array<int,inDims> idx = idxConverter.getIndexFromPseudoFlatId(i);

			std::array<int,inDims+1> idxPos;

			for (int j = 0; j < inDims; j++) {
				idxPos[j] = idx[j];
			}

			Eigen::Matrix<Pos_T, 2,1> pos;
			idxPos[inDims] = 0;
			pos[0] = sourcePos.valueUnchecked(idxPos);
			idxPos[inDims] = 1;
			pos[1] = sourcePos.valueUnchecked(idxPos);

			Eigen::Matrix<Pos_T, 2,1> oPos = reprojected(pos, depth.atUnchecked(idx));


			idxPos[inDims] = 0;
			outPos.atUnchecked(idxPos) = oPos[0];
			idxPos[inDims] = 1;
			outPos.atUnchecked(idxPos) = oPos[1];

		}

		return outPos;
	}

	Multidim::Array<Pos_T, 3> reprojectMap(Multidim::Array<Pos_T, 2> depth) {

		std::array<int,2> s = depth.shape();
		std::array<int,3> shp;

		shp[0] = s[0];
		shp[1] = s[1];
		shp[2] = 2;

		Multidim::Array<Pos_T, 3> outPos(shp);

		for (int i = 0; i < s[0]; i++) {
			for (int j = 0; j < s[1]; j++) {

				Eigen::Matrix<Pos_T, 2,1> pos;
				pos[0] = j;
				pos[1] = i;

				Eigen::Matrix<Pos_T, 2,1> oPos = reprojected(pos, depth.atUnchecked(i,j));

				outPos.atUnchecked(i,j,0) = oPos[0];
				outPos.atUnchecked(i,j,1) = oPos[1];
			}
		}

		return outPos;
	}

private:
	Eigen::Matrix<Pos_T, 2,1> _source_f;
	Eigen::Matrix<Pos_T, 2,1> _source_pp;
	Eigen::Matrix<Pos_T, 2,1> _target_f;
	Eigen::Matrix<Pos_T, 2,1> _target_pp;
	AffineTransform<Pos_T> _sourceToTarget;
	ImageAnchors _source_imageOrigin;
	ImageAnchors _target_imageOrigin;
};



} //namespace Geometry
} // namespace StereoVision

#endif // STEREOVISIONAPP_ALIGNEMENT_H
