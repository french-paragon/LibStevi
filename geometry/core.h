#ifndef STEREOVISION_CORE_H
#define STEREOVISION_CORE_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021-2023 Paragon<french.paragon@gmail.com>

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

#include <eigen3/Eigen/Core>

namespace StereoVision {
namespace Geometry {

template<typename T = float>
Eigen::Matrix<T,3,3> skew(Eigen::Matrix<T,3,1> const& v) {
    Eigen::Matrix<T,3,3> r;
    r << T(0), -v.z(), v.y(),
         v.z(), T(0), -v.x(),
         -v.y(), v.x(), T(0);

    return r;
}

template<typename T = float>
Eigen::Matrix<T,3,1> unskew(Eigen::Matrix<T,3,3> const& m) {
    return Eigen::Matrix<T,3,1>(m(2,1), -m(2,0), m(1,0));
}

enum class Axis : char {
    X = 0,
    Y = 1,
    Z = 2
};

inline Eigen::Vector3f pathFromDiff(Axis dir) {

    Eigen::Vector3f ret = Eigen::Vector3f::Zero();
    ret[static_cast<int>(dir)] = 1;
    return ret;
}


enum class IterativeTermination : char {
    Error,
    Converged,
    MaxStepReached
};

Eigen::Vector3f pathFromDiff(Axis dir);

template<typename T = float>
class AffineTransform
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        AffineTransform(Eigen::Matrix<T,3,3> R, Eigen::Matrix<T,3,1> t):
            t(t),
            R(R) {

        }
        AffineTransform():
            t(Eigen::Matrix<T,3,1>::Zero()),
            R(Eigen::Matrix<T,3,3>::Identity()) {

        }

        Eigen::Matrix<T,3,1> operator*(Eigen::Matrix<T,3,1> const& pt) const {
            return R*pt + t;
        }

        template <int nCols>
        std::enable_if_t<nCols!=1, Eigen::Matrix<T,3,nCols>> operator*(Eigen::Matrix<T,3,nCols> const& pts) const {
            return applyOnto<nCols>(pts.array()).matrix();
        }

        template <int nCols>
        std::enable_if_t<nCols!=1, Eigen::Matrix<T,3,nCols>> operator*(Eigen::Array<T,3,nCols> const& pts) const {
            return applyOnto<nCols>(pts);
        }
        AffineTransform<T> operator*(AffineTransform<T> const& other) const {
            return AffineTransform<T>(R*other.R, R*other.t + t);
        }

    inline bool isFinite() const {
        return t.array().isFinite().all() and R.array().isFinite().all();
    }

        Eigen::Matrix<T,3,1> t;
        Eigen::Matrix<T,3,3> R;

protected:

        template <int nCols>
        Eigen::Array<T,3,nCols> applyOnto(Eigen::Array<T,3,nCols> const& pts) const {

            Eigen::Array<T,3,nCols> transformedPts;
            transformedPts.resize(3, pts.cols());

            for (int i = 0; i < transformedPts.cols(); i++) {
                transformedPts.col(i) = R*(pts.col(i).matrix()) + t;
            }

            return transformedPts;
        }
};

} // namespace Geometry
} // namespace StereoVision

#endif // CORE_H
