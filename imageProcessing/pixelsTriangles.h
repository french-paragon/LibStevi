#ifndef PIXELSTRIANGLES_H
#define PIXELSTRIANGLES_H
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024  Paragon<french.paragon@gmail.com>

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

#include <vector>

#include <Eigen/Core>

namespace StereoVision {
namespace ImageProcessing {

template<typename T>
struct WeigthedPixCoord {
    Eigen::Matrix<int,2,1> pixCoord;
    T weight;
};

/*!
 * \brief listPixPointsInTriangle gives a list of pixels
 *
 * \param pt1 first point of the triangle
 * \param pt2 second point of the triangle
 * \param pt3 third point of the triangle
 *
 * \tparam ensureNearFullPix ensure that pixels closer than sqrt(0.5) from the centroid of the triangle get a weight of 100% if true
 * \tparam T the compute type (floating point expected)
 *
 * \return a list of pixels coordinates, with relative weights for anti-aliasing
 */
template<bool ensureNearFullPix = true, typename T>
std::vector<WeigthedPixCoord<T>> listPixPointsInTriangle(Eigen::Matrix<T,2,1> const& pt1,
                                                      Eigen::Matrix<T,2,1> const& pt2,
                                                      Eigen::Matrix<T,2,1> const& pt3) {

    constexpr T antiAliasingDist = 0.7; //almost sqrt(0.5)

    T minX = std::min(pt1.x(), pt2.x());
    T maxX = std::max(pt1.x(), pt2.x());

    T minY = std::min(pt1.y(), pt2.y());
    T maxY = std::max(pt1.y(), pt2.y());

    minX = std::min(minX, pt3.x());
    maxX = std::max(maxX, pt3.x());

    minY = std::min(minY, pt3.y());
    maxY = std::max(maxY, pt3.y());

    Eigen::Matrix<T,2,1> centroid = (pt1 + pt2 + pt3)/3;

    Eigen::Matrix<T,2,1> v12 = pt2-pt1;
    Eigen::Matrix<T,2,1> v21 = -v12;

    Eigen::Matrix<T,2,1> v13 = pt3-pt1;
    Eigen::Matrix<T,2,1> v31 = -v13;

    Eigen::Matrix<T,2,1> v23 = pt3-pt2;
    Eigen::Matrix<T,2,1> v32 = -v23;

    Eigen::Matrix<T,2,1> aprxdir23 = (v12 + v13);
    Eigen::Matrix<T,2,1> aprxdir31 = (v23 + v21);
    Eigen::Matrix<T,2,1> aprxdir12 = (v31 + v32);

    Eigen::Matrix<T,2,1> dir23 = Eigen::Matrix<T,2,1>(v23.y(), -v23.x());
    Eigen::Matrix<T,2,1> dir31 = Eigen::Matrix<T,2,1>(v31.y(), -v31.x());
    Eigen::Matrix<T,2,1> dir12 = Eigen::Matrix<T,2,1>(v12.y(), -v12.x());

    if (dir23.dot(aprxdir23) < 0) {
        dir23 = -dir23;
    }

    if (dir31.dot(aprxdir31) < 0) {
        dir31 = -dir31;
    }

    if (dir12.dot(aprxdir12) < 0) {
        dir12 = -dir12;
    }

    dir23.normalize();
    dir31.normalize();
    dir12.normalize();

    Eigen::Matrix<T,2,1> origin23 = (pt2 + pt3)/2;
    Eigen::Matrix<T,2,1> origin31 = (pt3 + pt1)/2;
    Eigen::Matrix<T,2,1> origin12 = (pt1 + pt2)/2;

    int iMinX = std::floor(minX);
    int iMaxX = std::ceil(maxX);

    int iMinY = std::floor(minY);
    int iMaxY = std::ceil(maxY);

    std::vector<WeigthedPixCoord<T>> ret;
    ret.reserve((iMaxX - iMinX)*(iMaxY-iMinY)/2 + 3*std::max(iMaxX-iMinX,iMaxY-iMinY) + 3);

    for (int i = iMinX; i <= iMaxX; i++) {
        for (int j = iMinY; j <= iMaxY; j++) {

            Eigen::Matrix<T,2,1> coord = Eigen::Matrix<T,2,1>(i, j);

            T pos23 = dir23.dot(coord - origin23)/dir23.dot(dir23);
            T pos31 = dir31.dot(coord - origin31)/dir31.dot(dir31);
            T pos12 = dir12.dot(coord - origin12)/dir12.dot(dir12);

            if (ensureNearFullPix) {
                if ((coord - centroid).norm() < sqrt(0.5)) {
                    ret.push_back({Eigen::Matrix<int,2,1>(i,j), 1});
                    continue;
                }
            }

            if (pos23 > antiAliasingDist) {
                continue;
            }

            if (pos31 > antiAliasingDist) {
                continue;
            }

            if (pos12 > antiAliasingDist) {
                continue;
            }

            T coeff23 = (pos23 < antiAliasingDist) ? 1 : 0.5 - pos23/(2*antiAliasingDist);
            T coeff31 = (pos31 < antiAliasingDist) ? 1 : 0.5 - pos31/(2*antiAliasingDist);
            T coeff12 = (pos12 < antiAliasingDist) ? 1 : 0.5 - pos12/(2*antiAliasingDist);

            T coeffFull = coeff23*coeff31*coeff12;

            ret.push_back({Eigen::Matrix<int,2,1>(i,j), coeffFull});

        }
    }

    return ret;

}

}
}


#endif // PIXELSTRIANGLES_H
