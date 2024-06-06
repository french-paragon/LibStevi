#ifndef POSESMATH_H
#define POSESMATH_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024 Paragon<french.paragon@gmail.com>

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
#include <Eigen/Geometry>

#include "./rotations.h"

namespace StereoVision {
namespace Geometry {

/*!
 * \brief computeRotationSpeed compute the rotation speed from two successive orientations (expressed as axis angles)
 * \param r1 the first orientation
 * \param r2 the second orientation
 * \param dt the time delta between the two orientation
 * \return the angular speed required to move from r1 to r2 in dt time at constant speed.
 */
template<typename T, typename tT>
Eigen::Matrix<T,3,1> computeRotationSpeed(Eigen::Matrix<T,3,1> const& r1, Eigen::Matrix<T,3,1> const& r2, tT dt) {

    Eigen::Quaternion<T> q1 = axisAngleToQuaternion(r1);
    Eigen::Quaternion<T> q2 = axisAngleToQuaternion(r2);

    //x*q1 = q2 -> x = q2 * q1^-1
    Eigen::Quaternion<T> qx = q2*q1.inverse();

    Eigen::Matrix<T,3,1> rx = quaternionToAxisAngle(qx);

    return rx*dt;

}

} // namespace Geometry
} // namespace StereoVision

#endif // POSESMATH_H
