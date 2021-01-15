#ifndef STEREOVISIONAPP_ALIGNEMENT_H
#define STEREOVISIONAPP_ALIGNEMENT_H

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

#include "geometry/core.h"

#include <utility>
#include <vector>

namespace StereoVision {
namespace Geometry {

Eigen::Array2Xf projectPoints(Eigen::Array3Xf const& pts);
Eigen::Array2Xf projectPoints(Eigen::Array3Xf const& pts, AffineTransform const& T);
Eigen::Array2Xf projectPoints(Eigen::Array3Xf const& pts, Eigen::Matrix3f const& R, Eigen::Vector3f const& t);

Eigen::Array2Xd projectPointsD(Eigen::Array3Xd const& pts);
Eigen::Array2Xd projectPointsD(Eigen::Array3Xd const& pts, Eigen::Matrix3d const& R, Eigen::Vector3d const& t);

Eigen::Array3Xf reprojectPoints(Eigen::Matrix3f const& R,
								 Eigen::Vector3f const& t,
								 Eigen::Array2Xf const& pt_cam_1,
								 Eigen::Array2Xf const& pt_cam_2);
Eigen::Array3Xf reprojectPoints(AffineTransform const& T,
								 Eigen::Array2Xf const& pt_cam_1,
								 Eigen::Array2Xf const& pt_cam_2);

Eigen::Matrix3f estimateEssentialMatrix(Eigen::Array2Xf const& pt_cam_1, Eigen::Array2Xf const& pt_cam_2);

std::pair<AffineTransform, AffineTransform> essentialMatrix2Transforms(Eigen::Matrix3f const& E);

AffineTransform essentialMatrix2Transform(Eigen::Matrix3f const& E,
										  Eigen::Array2Xf const& pt_cam_1,
										  Eigen::Array2Xf const& pt_cam_2);

AffineTransform selectTransform(AffineTransform const& T1,
								AffineTransform const& T2,
								Eigen::Array2Xf const& pt_cam_1,
								Eigen::Array2Xf const& pt_cam_2);

AffineTransform findTransform(Eigen::Array2Xf const& pt_cam_1,
							  Eigen::Array2Xf const& pt_cam_2);

AffineTransform pnp(Eigen::Array2Xf const& pt_cam, Eigen::Array3Xf const& pt_coords);
AffineTransform pnp(Eigen::Array2Xf const& pt_cam, std::vector<int> const& idxs, Eigen::Array3Xf const& pt_coords);

} //namespace Geometry
} // namespace StereoVision

#endif // STEREOVISIONAPP_ALIGNEMENT_H
