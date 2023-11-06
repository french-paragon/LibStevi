#ifndef PIXELSLINES_H
#define PIXELSLINES_H
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2023  Paragon<french.paragon@gmail.com>

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

#include <Eigen/Core>

namespace StereoVision {
namespace ImageProcessing {

/*!
 * \brief listPixPointsOnLine gives the coordinate of the pixels crossed by a line between two points
 * \param pt1 the starting point of the line
 * \param pt2 the ending point of the line
 * \return the point on the line the closest to the center of the pixel for each pixel crossed by the line.
 */
template<typename T>
Eigen::Array<T, 2, Eigen::Dynamic> listPixPointsOnLine(Eigen::Matrix<T,2,1> const& pt1,
                                                       Eigen::Matrix<T,2,1> const& pt2,
                                                       bool discret = false) {

    if (pt1.array().isNaN().any() or pt2.array().isNaN().any()) {
        return Eigen::Array<T, 2, Eigen::Dynamic>();
    }

    static_assert (std::is_floating_point_v<T>, "Function requires a floating point value");

    int px1_idx_x = static_cast<int>((pt1.x() < pt2.x()) ? std::floor(pt1[0]) : std::ceil(pt1[0]));
    int px2_idx_x = static_cast<int>((pt2.x() < pt1.x()) ? std::floor(pt2[0]) : std::ceil(pt2[0]));

    int px1_idx_y = static_cast<int>((pt1.y() < pt2.y()) ? std::floor(pt1[1]) : std::ceil(pt1[1]));
    int px2_idx_y = static_cast<int>((pt2.y() < pt1.y()) ? std::floor(pt2[1]) : std::ceil(pt2[1]));

    int nPixX = std::abs(px2_idx_x - px1_idx_x)+1;
    int nPixY = std::abs(px2_idx_y - px1_idx_y)+1;

    int x_incr = (px2_idx_x - px1_idx_x >= 0) ? 1 : -1;
    int y_incr = (px2_idx_y - px1_idx_y >= 0) ? 1 : -1;

    int nPixels = nPixX + nPixY - 1;

    Eigen::Array<T, 2, Eigen::Dynamic> ret;
    ret.resize(2, nPixels);

    Eigen::Matrix<T,2,1> o(x_incr*px1_idx_x, y_incr*px1_idx_y);
    Eigen::Matrix<T,2,1> v0(x_incr*pt1[0], y_incr*pt1[1]);
    Eigen::Matrix<T,2,1> v1(x_incr*pt2[0], y_incr*pt2[1]);

    Eigen::Matrix<T,2,1> v = v1 - v0;

    int dx = 0;
    int dy = 0;

    T vSquared = v.dot(v);

    for (int i = 0; i < nPixels; i++) {

        Eigen::Matrix<T,2,1> x(o[0]+dx, o[1]+dy);

        //intersect the line y = -x + c passing through the current pixel. This line yield a fractional approximation garanteed to be within the pixel (for antialiasing).
        T c = x[0] + x[1];
        T t = (c - v0[0] - v0[1])/(v[0] + v[1]);

        Eigen::Matrix<T,2,1> nearest = v0 + t*v;

        if (discret) {
            ret(0,i) = x_incr*x[0];
            ret(1,i) = y_incr*x[1];
        } else {
            ret(0,i) = x_incr*nearest[0];
            ret(1,i) = y_incr*nearest[1];
        }

        T rdx = (std::ceil(nearest[0]+0.5) - (nearest[0]+0.5))*v[1];
        T rdy = (std::ceil(nearest[1]+0.5) - (nearest[1]+0.5))*v[0];

        if (rdx > rdy) {
            dy++;
        } else {
            dx++;
        }
    }

    return ret;

}

} // namespace ImageProcessing
} // namespace StereoVision

#endif // PIXELSLINES_H
