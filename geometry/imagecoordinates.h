#ifndef IMAGECOORDINATES_H
#define IMAGECOORDINATES_H

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

namespace StereoVision {
namespace Geometry {

enum class ImageAnchors : char {
	TopLeft,
	TopRight,
	BottomLeft,
	BottomRight
};

Eigen::Vector2f Image2HomogeneousCoordinates(Eigen::Vector2f const& pt,
											 float fx,
											 float fy,
											 Eigen::Vector2f const& pp,
											 ImageAnchors imageOrigin);

Eigen::Vector2f Image2HomogeneousCoordinates(Eigen::Vector2f const& pt,
											 float f,
											 Eigen::Vector2f const& pp,
											 ImageAnchors imageOrigin);

Eigen::Array2Xf Image2HomogeneousCoordinates(Eigen::Array2Xf const& pt,
											 float fx,
											 float fy,
											 Eigen::Vector2f const& pp,
											 ImageAnchors imageOrigin);

Eigen::Array2Xf Image2HomogeneousCoordinates(Eigen::Array2Xf const& pt,
											 float f,
											 Eigen::Vector2f const& pp,
											 ImageAnchors imageOrigin);

} // namespace Geometry
} // namespace StereoVision

#endif // IMAGECOORDINATES_H
