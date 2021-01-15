#ifndef STEREOVISION_LENSDISTORTIONSMAP_H
#define STEREOVISION_LENSDISTORTIONSMAP_H

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

#include "../stevi_global.h"

#include <eigen3/Eigen/Core>

namespace StereoVision {
namespace Interpolation {

ImageArray computeLensDistortionMap(int height,
									int width,
									float f,
									Eigen::Vector2f pp,
									Eigen::Vector3f k123,
									Eigen::Vector2f t12,
									Eigen::Vector2f B12);

} // namespace Interpolation
} // namespace StereoVisionApp

#endif // STEREOVISIONAPP_LENSDISTORTIONSMAP_H
