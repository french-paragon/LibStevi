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

#include "lensdistortionsmap.h"

#include "geometry/lensdistortion.h"

namespace StereoVision {
namespace Interpolation {

ImageArray computeLensDistortionMap(int height,
									int width,
									float f,
									Eigen::Vector2f pp,
									Eigen::Vector3f k123,
									Eigen::Vector2f t12,
									Eigen::Vector2f B12) {

	ImageArray out(height, width, 2);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			float unDist_im_x = j;
			float unDist_im_y = i;

			Eigen::Vector2f fPos(unDist_im_x, unDist_im_y);
			Eigen::Vector2f homogeneous = (fPos - pp)/f;

			Eigen::Vector2f dr = Geometry::radialDistortion(homogeneous, k123);
			Eigen::Vector2f dt = Geometry::tangentialDistortion(homogeneous, t12);

			homogeneous += dr + dt;

			Eigen::Vector2f dPos = Geometry::skewDistortion(homogeneous, B12, f, pp);

			out.at<Multidim::AccessCheck::Nocheck>(i, j, 0) = dPos.y();
			out.at<Multidim::AccessCheck::Nocheck>(i, j, 1) = dPos.x();
		}
	}

	return out;
}

} // namespace Interpolation
} // namespace StereoVision
