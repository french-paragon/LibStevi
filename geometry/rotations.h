#ifndef STEREOVISION_ROTATIONS_H
#define STEREOVISION_ROTATIONS_H

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

class ShapePreservingTransform
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	ShapePreservingTransform(Eigen::Vector3f r, Eigen::Vector3f t, float s);
	ShapePreservingTransform();

	Eigen::Vector3f operator*(Eigen::Vector3f const& pt) const;
	Eigen::Matrix3Xf operator*(Eigen::Matrix3Xf const& pts) const;
	Eigen::Array3Xf operator*(Eigen::Array3Xf const& pts) const;

	AffineTransform toAffineTransform() const;
	ShapePreservingTransform inverse() const;

	Eigen::Vector3f t;
	Eigen::Vector3f r;
	float s;

protected:

	Eigen::Array3Xf applyOnto(Eigen::Array3Xf const& pts) const;
};

Eigen::Matrix3f rodriguezFormula(Eigen::Vector3f const& r);
Eigen::Vector3f inverseRodriguezFormula(Eigen::Matrix3f const& R);

Eigen::Matrix3f diffRodriguezLieAlgebra(Eigen::Vector3f const& r);
Eigen::Matrix3f diffRodriguez(Eigen::Vector3f const& r, Axis direction);

Eigen::Matrix3d rodriguezFormulaD(Eigen::Vector3d const& r);
Eigen::Vector3d inverseRodriguezFormulaD(Eigen::Matrix3d const& R);

Eigen::Matrix3d diffRodriguezLieAlgebraD(Eigen::Vector3d const& r);
Eigen::Matrix3d diffRodriguezD(Eigen::Vector3d const& r, Axis direction);

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> eulerRadXYZToRotation(Scalar eulerX,
												  Scalar eulerY,
												  Scalar eulerZ) {

	return (Eigen::AngleAxis<Scalar>(eulerX, Eigen::Matrix<Scalar,3,1>::UnitX())*
			Eigen::AngleAxis<Scalar>(eulerY, Eigen::Matrix<Scalar,3,1>::UnitY())*
			Eigen::AngleAxis<Scalar>(eulerZ, Eigen::Matrix<Scalar,3,1>::UnitZ())).toRotationMatrix();

}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> eulerDegXYZToRotation(Scalar eulerX,
												  Scalar eulerY,
												  Scalar eulerZ) {
	return eulerRadXYZToRotation(eulerX/180*M_PI, eulerY/180*M_PI, eulerZ/180*M_PI);
}

} // namespace Geometry
} //namespace StereoVision

#endif // ROTATIONS_H
