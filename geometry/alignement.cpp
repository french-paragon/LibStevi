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

namespace StereoVision {
namespace Geometry {


/*!
 * \brief pnp compute the estimated target to cam transform from projected points
 * \param pt_cam 2D points in homogeneous normalized image coordinates
 * \param pt_coords matching 3D points in target frame coordinates
 * \return the affine transform corresponding to the target to cam transform
 */
AffineTransform<float> pnp(Eigen::Array2Xf const& pt_cam, Eigen::Array3Xf const& pt_coords) {

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
AffineTransform<float> pnp(Eigen::Array2Xf const& pt_cam, std::vector<int> const& idxs, Eigen::Array3Xf const& pt_coords) {

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

	AffineTransform<float> T;
	T.R << R(0,0), R(0,1), R(0,2),
			R(1,0), R(1,1), R(1,2),
			R(2,0), R(2,1), R(2,2);
	T.t << t.at<0>(), t.at<1>(), t.at<2>();

	return T;
}



AffineTransform<double> pnp(Eigen::Array2Xd const& pt_cam, Eigen::Array3Xd const& pt_coords) {
	std::vector<int> idxs(pt_cam.cols());

	for(int i = 0; i < pt_cam.cols(); i++) {
		idxs[i] = i;
	}

	return pnp(pt_cam, idxs, pt_coords);
}
AffineTransform<double> pnp(Eigen::Array2Xd const& pt_cam, std::vector<int> const& idxs, Eigen::Array3Xd const& pt_coords) {

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

	AffineTransform<double> T;
	T.R << R(0,0), R(0,1), R(0,2),
			R(1,0), R(1,1), R(1,2),
			R(2,0), R(2,1), R(2,2);
	T.t << t.at<0>(), t.at<1>(), t.at<2>();

	return T;

}

} // namespace Geometry
} // namespace StereoVision
