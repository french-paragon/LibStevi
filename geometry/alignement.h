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
#include "../geometry/lensdistortion.h"
#include "../geometry/geometricexception.h"

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

AffineTransform<float> pnp(Eigen::Array2Xf const& pt_cam, Eigen::Array3Xf const& pt_coords);
AffineTransform<float> pnp(Eigen::Array2Xf const& pt_cam, std::vector<int> const& idxs, Eigen::Array3Xf const& pt_coords);

AffineTransform<double> pnp(Eigen::Array2Xd const& pt_cam, Eigen::Array3Xd const& pt_coords);
AffineTransform<double> pnp(Eigen::Array2Xd const& pt_cam, std::vector<int> const& idxs, Eigen::Array3Xd const& pt_coords);



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
