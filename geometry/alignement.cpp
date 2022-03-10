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

#include "alignement.h"

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include <pnp/pnp_ransac.h>

#include "geometricexception.h"

namespace StereoVision {
namespace Geometry {
/*!
 * \brief projectPoints project points in homogeneous normalized image coordinates
 * \param pts the points to project, assumed to be given in the camera frame
 * \return the projected points coordinates.
 */
Eigen::Array2Xf projectPoints(Eigen::Array3Xf const& pts) {

	Eigen::Array2Xf proj;
	proj.resize(2, pts.cols());

	proj.row(0) = pts.row(0)/pts.row(2);
	proj.row(1) = pts.row(1)/pts.row(2);

	return proj;

}
/*!
 * \brief projectPoints project points in homogeneous normalized image coordinates
 * \param pts the points to project
 * \param T the transform from the points frame to the camera frame
 * \return the projected points coordinates.
 */
Eigen::Array2Xf projectPoints(Eigen::Array3Xf const& pts, AffineTransform const& T) {
	Eigen::Array3Xf transformedPts = T*pts;
	return projectPoints(transformedPts);
}
/*!
 * \brief projectPoints project points in homogeneous normalized image coordinates
 * \param pts the points to project
 * \param R the rotation part of the transform from the points frame to the camera frame
 * \param t the translation part of the transform from the points frame to the camera frame
 * \return the projected points coordinates.
 */
Eigen::Array2Xf projectPoints(Eigen::Array3Xf const& pts, Eigen::Matrix3f const& R, Eigen::Vector3f const& t) {
	Eigen::Array3Xf transformedPts = AffineTransform(R,t)*pts;
	return projectPoints(transformedPts);
}

Eigen::Array2Xd projectPointsD(Eigen::Array3Xd const& pts) {

	Eigen::Array2Xd proj;
	proj.resize(2, pts.cols());

	proj.row(0) = pts.row(0)/pts.row(2);
	proj.row(1) = pts.row(1)/pts.row(2);

	return proj;
}

Eigen::Array2Xd projectPointsD(Eigen::Array3Xd const& pts, Eigen::Matrix3d const& R, Eigen::Vector3d const& t) {

	Eigen::Array3Xd transformedPts = pts;

	for (int i = 0; i < pts.cols(); i++) {
		transformedPts.col(i) = R*pts.col(i).matrix() + t;
	}

	return projectPointsD(transformedPts);
}

/*!
 * \brief reprojectPoints from the transformation from cam1 to cam2 and a set of points in homogeneous coordinates in both images, find the 3D coordinates of points in cam1 frame.
 * \param T the transform cam1 to cam2
 * \param pt_cam_1 points in cam1 homogeneous coordinates
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the points coordinates in cam1 frame
 */
Eigen::Array3Xf reprojectPoints(AffineTransform const& T,
								Eigen::Array2Xf const& pt_cam_1,
								Eigen::Array2Xf const& pt_cam_2) {
	return reprojectPoints(T.R, T.t, pt_cam_1, pt_cam_2);
}

/*!
 * \brief reprojectPoints from the transformation from cam1 to cam2 and a set of points in homogeneous coordinates in both images, find the 3D coordinates of points in cam1 frame.
 * \param R the rotation part of the transform cam1 2 cam2
 * \param t the translation part of the transform cam1 2 cam2
 * \param pt_cam_1 points in cam1 homogeneous coordinates
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the points coordinates in cam1 frame
 */
Eigen::Array3Xf reprojectPoints(Eigen::Matrix3f const& R,
								 Eigen::Vector3f const& t,
								 Eigen::Array2Xf const& pt_cam_1,
								 Eigen::Array2Xf const& pt_cam_2) {

	int nPts = pt_cam_1.cols();

	if (pt_cam_1.cols() != pt_cam_2.cols()) {
		throw GeometricException("Points arrays of different dimensions provided");
	}

	Eigen::Array3Xf reproj;

	reproj.resize(3,nPts);
	reproj.topRows(2) = pt_cam_1;
	reproj.bottomRows(1).setOnes();

	Eigen::ArrayXf x3_v1 = (t.x() - pt_cam_2.row(0)*t.z()) /
			(
				pt_cam_2.row(0)*(R(2,0)*pt_cam_1.row(0) + R(2,1)*pt_cam_1.row(1) + R(2,2)) -
				(R(0,0)*pt_cam_1.row(0) + R(0,1)*pt_cam_1.row(1) + R(0,2))
			);

	Eigen::ArrayXf x3_v2 = (t.y() - pt_cam_2.row(1)*t.z()) /
			(
				pt_cam_2.row(1)*(R(2,0)*pt_cam_1.row(0) + R(2,1)*pt_cam_1.row(1) + R(2,2)) -
				(R(1,0)*pt_cam_1.row(0) + R(1,1)*pt_cam_1.row(1) + R(1,2))
			);

	Eigen::ArrayXf x3 = (x3_v1 + x3_v2)/2.0;

	//make sure to get only the values of x3_v1 if x3_v2 is not finite and vice versa.
	x3 = (x3.isFinite()).select(x3,x3_v1);
	x3 = (x3.isFinite()).select(x3,x3_v2);

	reproj.row(0) *= x3;
	reproj.row(1) *= x3;
	reproj.row(2) *= x3;

	return reproj;

}

/*!
 * \brief reprojectPointsLstSqr is a more robust but more expensive reprojection routine to compute points reprojections (compared to reprojectPoints)
 * \param T the transform cam1 to cam2
 * \param pt_cam_1 points in cam1 homogeneous coordinates
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the points coordinates in cam1 frame
 */
Eigen::Array3Xf reprojectPointsLstSqr(AffineTransform const& T,
									  Eigen::Array2Xf const& pt_cam_1,
									  Eigen::Array2Xf const& pt_cam_2) {
	return reprojectPointsLstSqr(T.R, T.t, pt_cam_1, pt_cam_2);
}
/*!
 * \brief reprojectPointsLstSqr is a more robust but more expensive reprojection routine to compute points reprojections (compared to reprojectPoints)
 * \param R the rotation part of the transform cam1 2 cam2
 * \param t the translation part of the transform cam1 2 cam2
 * \param pt_cam_1 points in cam1 homogeneous coordinates
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the points coordinates in cam1 frame
 */
Eigen::Array3Xf reprojectPointsLstSqr(Eigen::Matrix3f const& R,
									  Eigen::Vector3f const& t,
									  Eigen::Array2Xf const& pt_cam_1,
									  Eigen::Array2Xf const& pt_cam_2) {

	typedef Eigen::Matrix<float, 3, 2> MatrixAtype;

	int nPts = pt_cam_1.cols();

	if (pt_cam_1.cols() != pt_cam_2.cols()) {
		throw GeometricException("Points arrays of different dimensions provided");
	}

	Eigen::Array3Xf reproj;

	reproj.resize(3,nPts);
	reproj.topRows(2) = pt_cam_1;
	reproj.bottomRows(1).setOnes();

	for (int i = 0; i < nPts; i++) {
		Eigen::Vector3f v2;
		v2.block<2,1>(0,0) = pt_cam_2.col(i);
		v2[2] = 1;
		Eigen::Vector3f v2C1 = R.transpose()*v2;
		MatrixAtype A;
		A.col(0) = reproj.col(i);
		A.col(1) = -v2C1;

		Eigen::Matrix2f invQxx = A.transpose()*A;

		auto svd = invQxx.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);

		Eigen::Matrix2f pseudoInverse = svd.matrixV() * (svd.singularValues().array().abs() > 1e-4).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().transpose();

		Eigen::Vector2f lambdas = pseudoInverse*A.transpose()*(-R.transpose()*t);

		Eigen::Vector3f est1 = A.col(0)*lambdas[0];
		Eigen::Vector3f est2 = -A.col(1)*lambdas[1] -R.transpose()*t;

		reproj.col(i) = (est1 + est2)/2.;
	}

	return reproj;

}

/*!
 * \brief estimateEssentialMatrix estimate the essential matrix between a pair of cameras
 * \param pt_cam_1 points in cam1 homogeneous coordinates (must be at least 8 points)
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the essential matrix.
 */
Eigen::Matrix3f estimateEssentialMatrix(const Eigen::Array2Xf &pt_cam_1, const Eigen::Array2Xf &pt_cam_2) {

	typedef Eigen::Matrix<float, 9, Eigen::Dynamic> MatrixFtype;

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
	Eigen::Matrix3f E;
	E << e[0], e[1], e[2],
		 e[3], e[4], e[5],
		 e[6], e[7], e[8];

	return E;

}

/*!
 * \brief essentialMatrix2Transforms extract a pair of possible transforms from an essential matrix
 * \param E the essential matrix
 * \return a pair of possible transforms that can be used in the selectTransform function.
 */
std::pair<AffineTransform, AffineTransform> essentialMatrix2Transforms(Eigen::Matrix3f const& E) {

	auto svd = E.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix3f U = svd.matrixU();
	Eigen::Matrix3f V = svd.matrixV();

	//ensure both U and V are rotation matrices.
	if (U.determinant() < 0) {
		U = -U;
	}

	if (V.determinant() < 0) {
		V = -V;
	}

	Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
	W(1,0) = -1;
	W(0,1) = 1;
	W(2,2) = 1;

	AffineTransform T1;
	AffineTransform T2;

	T1.R = U*W*V.transpose();
	T2.R = U*W.transpose()*V.transpose();

	W(2,2) = 0;

	T1.t = unskew(U*W*U.transpose());
	T2.t = -T1.t;

	return {T1, T2};
}

/*!
 * \brief essentialMatrix2Transform extract the transformation cam1 to cam2 from the estimated essential matrix
 * \param E the essential matrix
 * \param pt_cam_1 the points used to estimate E, in cam1 homogeneous coordinates
 * \param pt_cam_2 the points used to estimate E, in cam2 homogeneous coordinates
 * \return the transformation cam1 to cam2
 */
AffineTransform essentialMatrix2Transform(Eigen::Matrix3f const& E,
										  Eigen::Array2Xf const& pt_cam_1,
										  Eigen::Array2Xf const& pt_cam_2) {
	auto candidates = essentialMatrix2Transforms(E);
	return selectTransform(candidates.first, candidates.second, pt_cam_1, pt_cam_2);
}

/*!
 * \brief selectTransform from a pair of possible transforms extracted from the essential matrix, extract the transform which puts all points in front of both cameras
 * \param T1 a possible transform
 * \param T2 another possible transform, with different rotation and translation
 * \param pt_cam_1 the points used to estimate E, in cam1 homogeneous coordinates
 * \param pt_cam_2 the points used to estimate E, in cam2 homogeneous coordinates
 * \return The transform which reproject all points in front of both cameras (might be T1, T2 or a mix of both).
 */
AffineTransform selectTransform(AffineTransform const& T1,
								AffineTransform const& T2,
								const Eigen::Array2Xf &pt_cam_1,
								const Eigen::Array2Xf &pt_cam_2) {

	int nPts = pt_cam_1.cols();

	AffineTransform Rs[4];

	int count_good = 0;

	for (auto const& R_cand : {T1.R, T2.R}) {

		for (auto const& t_cand : {T1.t, T2.t}) {

			Eigen::Array3Xf reproj = reprojectPoints(R_cand, t_cand, pt_cam_1, pt_cam_2);

			if ((reproj.bottomRows(1) >= 0.).all()) {
				reproj = reprojectPoints(R_cand.transpose(), -R_cand.transpose()*t_cand, pt_cam_2, pt_cam_1);

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

			Eigen::Array3Xf reproj = reprojectPoints(Rs[i], pt_cam_1, pt_cam_2);
			Eigen::Array2Xf onOtherIm = projectPoints(reproj, Rs[i]);

			float c_error = (onOtherIm - pt_cam_2).matrix().norm()/nPts;

			reproj = reprojectPoints(Rs[i].R.transpose(), -Rs[i].R.transpose()*Rs[i].t, pt_cam_2, pt_cam_1);
			onOtherIm = projectPoints(reproj, Rs[i].R.transpose(), -Rs[i].R.transpose()*Rs[i].t);

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

/*!
 * \brief findTransform jointly build the essential matrix and extract the corresponding transform from it
 * \param pt_cam_1 points in cam1 homogeneous coordinates (must be at least 8 points)
 * \param pt_cam_2 same points in cam2 homogeneous coordinates
 * \return the transform from cam1 to cam2
 */
AffineTransform findTransform(Eigen::Array2Xf const& pt_cam_1,
							  Eigen::Array2Xf const& pt_cam_2) {

	Eigen::Matrix3f E = estimateEssentialMatrix(pt_cam_1, pt_cam_2);
	return essentialMatrix2Transform(E, pt_cam_1, pt_cam_2);
}

/*!
 * \brief pnp compute the estimated target to cam transform from projected points
 * \param pt_cam 2D points in homogeneous normalized image coordinates
 * \param pt_coords matching 3D points in target frame coordinates
 * \return the affine transform corresponding to the target to cam transform
 */
AffineTransform pnp(Eigen::Array2Xf const& pt_cam, Eigen::Array3Xf const& pt_coords) {

	std::vector<int> idxs(pt_cam.cols());

	for(int i = 0; i < pt_cam.cols(); i++) {
		idxs[i] = i;
	}

	return pnp(pt_cam, idxs, pt_coords);

}

/*!
 * \brief pnp compute the estimated target to cam transform from projected points
 * \param pt_cam 2D points in homogeneous normalized image coordinates
 * \param idxs indices of the points the 2D landmarks correspond to
 * \param pt_coords the 3D points in target frame coordinates
 * \return the affine transform corresponding to the target to cam transform
 */
AffineTransform pnp(Eigen::Array2Xf const& pt_cam, std::vector<int> const& idxs, Eigen::Array3Xf const& pt_coords) {

	std::vector<cvl::Vector3D> xs;
	std::vector<cvl::Vector2D> yns;

	xs.reserve(idxs.size());
	yns.reserve(idxs.size());

	for (size_t i = 0; i < idxs.size(); i++) {
		cvl::Vector3D pt;
		pt.at<0>() = pt_coords.col(idxs[i]).x();
		pt.at<1>() = pt_coords.col(idxs[i]).y();
		pt.at<2>() = pt_coords.col(idxs[i]).z();

		cvl::Vector2D im_pt;

		im_pt.at<0>() = pt_cam.col(i).x();
		im_pt.at<1>() = pt_cam.col(i).y();

		if (!std::isfinite(pt.at<0>()) or
				!std::isfinite(pt.at<1>()) or
				!std::isfinite(pt.at<2>()) or
				!std::isfinite(im_pt.at<0>()) or
				!std::isfinite(im_pt.at<1>())) {
			continue;
		}

		xs.push_back(pt);
		yns.push_back(im_pt);
	}

	cvl::PoseD pose = cvl::pnp_ransac(xs, yns);

	cvl::Matrix3d R = pose.rotation();
	cvl::Vector3d t = pose.translation();

	AffineTransform T;
	T.R << R(0,0), R(0,1), R(0,2),
			R(1,0), R(1,1), R(1,2),
			R(2,0), R(2,1), R(2,2);
	T.t << t.at<0>(), t.at<1>(), t.at<2>();

	return T;

}

} // namespace Geometry
} // namespace StereoVision
