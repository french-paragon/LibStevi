#ifndef STEREOVISION_POINTCLOUDALIGNMENT_H
#define STEREOVISION_POINTCLOUDALIGNMENT_H

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

#include "../geometry/rotations.h"

#include <vector>
#include <optional>


namespace  StereoVision {
namespace Geometry {

AffineTransform<float> estimateAffineMap(Eigen::VectorXf const& obs,
								  Eigen::Matrix3Xf const& pts,
								  std::vector<int> const& idxs,
								  std::vector<Axis> const& coordinate);

AffineTransform<float> estimateQuasiShapePreservingMap(Eigen::VectorXf const& obs,
												Eigen::Matrix3Xf const& pts,
												std::vector<int> const& idxs,
												std::vector<Axis> const& coordinate,
												float damping = 2e-1,
												IterativeTermination * status = nullptr,
												float incrLimit = 1e-4,
												int iterationLimit = 500,
												bool verbose = false);

AffineTransform<float> estimateQuasiRigidMap(Eigen::VectorXf const& obs,
									  Eigen::Matrix3Xf const& pts,
									  std::vector<int> const& idxs,
									  std::vector<Axis> const& coordinate,
									  float damping = 2e-1,
									  IterativeTermination * status = nullptr,
									  float incrLimit = 1e-4,
									  int iterationLimit = 500,
									  bool verbose = false);

ShapePreservingTransform<float> affine2ShapePreservingMap(AffineTransform<float> const & initial);

ShapePreservingTransform<float> estimateTranslationMap(Eigen::VectorXf const& obs,
												Eigen::Matrix3Xf const& pts,
												std::vector<int> const& idxs,
												std::vector<Axis> const& coordinate,
												float *residual,
												bool verbose);

ShapePreservingTransform<float> estimateScaleMap(Eigen::VectorXf const& obs,
										  Eigen::Matrix3Xf const& pts,
										  std::vector<int> const& idxs,
										  std::vector<Axis> const& coordinate,
										  float *residual,
										  bool verbose);

ShapePreservingTransform<float> estimateRotationMap(Eigen::VectorXf const& obs,
											 Eigen::Matrix3Xf const& pts,
											 std::vector<int> const& idxs,
											 std::vector<Axis> const& coordinate,
											 float *residual,
											 IterativeTermination * status,
											 bool verbose,
											 int n_steps = 50,
											 float incrLimit = 1e-8);

ShapePreservingTransform<float> estimateShapePreservingMap(Eigen::VectorXf const& obs,
													Eigen::Matrix3Xf const& pts,
													std::vector<int> const& idxs,
													std::vector<Axis> const& coordinate,
													IterativeTermination * status,
													int n_steps = 50,
													float incrLimit = 1e-8,
													float damping = 5e-1,
													float dampingScale = 1e-1);


//! \brief this function gives a coarse estimate of a shape preserving map, if at least three points have all their coordinates observed.
std::optional<ShapePreservingTransform<float>> initShapePreservingMapEstimate(Eigen::VectorXf const& obs,
																	   Eigen::Matrix3Xf const& pts,
																	   std::vector<int> const& idxs,
																	   std::vector<Axis> const& coordinate);

} // namespace Geometry
}; //namespace StereoVision

#endif // POINTCLOUDALIGNMENT_H
