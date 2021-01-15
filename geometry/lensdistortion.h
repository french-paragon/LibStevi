#ifndef STEREOVISIONAPP_LENSDISTORTION_H
#define STEREOVISIONAPP_LENSDISTORTION_H

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

#include "../geometry/core.h"

namespace StereoVision {
namespace Geometry {

Eigen::Vector2f radialDistortion(Eigen::Vector2f pos, Eigen::Vector3f k123);
Eigen::Vector2d radialDistortionD(Eigen::Vector2d pos, Eigen::Vector3d k123);

Eigen::Vector2f tangentialDistortion(Eigen::Vector2f pos, Eigen::Vector2f t12);
Eigen::Vector2d tangentialDistortionD(Eigen::Vector2d pos, Eigen::Vector2d t12);

Eigen::Vector2f invertRadialDistorstion(Eigen::Vector2f pos,
										Eigen::Vector3f k123,
										int iters = 5);
Eigen::Vector2f invertTangentialDistorstion(Eigen::Vector2f pos,
											Eigen::Vector2f t12,
											int iters = 5);
Eigen::Vector2f invertRadialTangentialDistorstion(Eigen::Vector2f pos,
												  Eigen::Vector3f k123,
												  Eigen::Vector2f t12,
												  int iters = 5);

Eigen::Vector2f skewDistortion(Eigen::Vector2f pos, Eigen::Vector2f B12, float f, Eigen::Vector2f pp);
Eigen::Vector2d skewDistortionD(Eigen::Vector2d pos, Eigen::Vector2d B12, double f, Eigen::Vector2d pp);

Eigen::Vector2f inverseSkewDistortion(Eigen::Vector2f pos,
									  Eigen::Vector2f B12,
									  float f,
									  Eigen::Vector2f pp);

} // namespace Geometry
} // namespace StereoVision

#endif // STEREOVISIONAPP_LENSDISTORTION_H
