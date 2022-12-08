#ifndef STEREOVISION_CORE_H
#define STEREOVISION_CORE_H

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

#include <eigen3/Eigen/Core>

namespace StereoVision {
namespace Geometry {

Eigen::Matrix3f skew(Eigen::Vector3f const& v);
Eigen::Vector3f unskew(Eigen::Matrix3f const& m);

Eigen::Matrix3d skewD(Eigen::Vector3d const& v);
Eigen::Vector3d unskewD(Eigen::Matrix3d const& m);

enum class Axis : char {
	X,
	Y,
	Z
};


enum class IterativeTermination : char {
	Error,
	Converged,
	MaxStepReached
};

Eigen::Vector3f pathFromDiff(Axis dir);

class AffineTransform
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	AffineTransform(Eigen::Matrix3f R, Eigen::Vector3f t);
	AffineTransform();

	Eigen::Vector3f operator*(Eigen::Vector3f const& pt) const;
        Eigen::Matrix3Xf operator*(Eigen::Matrix3Xf const& pts) const;
        Eigen::Array3Xf operator*(Eigen::Array3Xf const& pts) const;
        AffineTransform operator*(AffineTransform const& other) const;

	inline bool isFinite() const {
		return t.array().isFinite().all() and R.array().isFinite().all();
	}

	Eigen::Vector3f t;
	Eigen::Matrix3f R;

protected:

	Eigen::Array3Xf applyOnto(Eigen::Array3Xf const& pts) const;
};

} // namespace Geometry
} // namespace StereoVision

#endif // CORE_H
