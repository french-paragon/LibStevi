#ifndef POINTCLOUDALIGNMENT_H
#define POINTCLOUDALIGNMENT_H

#include "geometry/rotations.h"

#include <vector>


namespace  StereoVision {
namespace Geometry {

AffineTransform estimateAffineMap(Eigen::VectorXf const& obs,
								  Eigen::Matrix3Xf const& pts,
								  std::vector<int> const& idxs,
								  std::vector<Axis> const& coordinate);

AffineTransform estimateQuasiShapePreservingMap(Eigen::VectorXf const& obs,
												Eigen::Matrix3Xf const& pts,
												std::vector<int> const& idxs,
												std::vector<Axis> const& coordinate,
												float damping = 2e-1,
												IterativeTermination * status = nullptr,
												float incrLimit = 1e-4,
												int iterationLimit = 500,
												bool verbose = false);

ShapePreservingTransform affine2ShapePreservingMap(AffineTransform const & initial);

AffineTransform estimateShapePreservingMap(Eigen::VectorXf const& obs,
										   Eigen::Matrix3Xf const& pts,
										   std::vector<int> const& idxs,
										   std::vector<Axis> const& coordinate,
										   IterativeTermination * status,
										   int n_steps = 50,
										   float incrLimit = 1e-8,
										   float damping = 5e-1,
										   float dampingScale = 1e-1);

} // namespace Geometry
}; //namespace StereoVision

#endif // POINTCLOUDALIGNMENT_H
