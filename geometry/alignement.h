#ifndef STEREOVISIONAPP_ALIGNEMENT_H
#define STEREOVISIONAPP_ALIGNEMENT_H

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
