#ifndef POINTCLOUDALIGNMENT_H
#define POINTCLOUDALIGNMENT_H

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

#include "../geometry/rotations.h"

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
