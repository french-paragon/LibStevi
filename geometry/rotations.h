#ifndef ROTATIONS_H
#define ROTATIONS_H

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

} // namespace Geometry
} //namespace StereoVision

#endif // ROTATIONS_H
