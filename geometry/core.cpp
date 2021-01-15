#include "core.h"

namespace StereoVision {
namespace Geometry {

Eigen::Matrix3f skew(Eigen::Vector3f const& v) {
	Eigen::Matrix3f r;
	r << 0, -v.z(), v.y(),
		 v.z(), 0, -v.x(),
		 -v.y(), v.x(), 0;

	return r;
}
Eigen::Vector3f unskew(Eigen::Matrix3f const& m) {
	return Eigen::Vector3f(m(2,1), -m(2,0), m(1,0));
}

Eigen::Matrix3d skewD(Eigen::Vector3d const& v) {
	Eigen::Matrix3d r;
	r << 0, -v.z(), v.y(),
		 v.z(), 0, -v.x(),
		 -v.y(), v.x(), 0;

	return r;
}
Eigen::Vector3d unskewD(Eigen::Matrix3d const& m) {
	return Eigen::Vector3d(m(2,1), -m(2,0), m(1,0));
}

Eigen::Vector3f pathFromDiff(Axis dir) {
	switch (dir) {
	case Axis::X:
		return Eigen::Vector3f(1.,0,0);
	case Axis::Y:
		return Eigen::Vector3f(0,1.,0);
	case Axis::Z:
		return Eigen::Vector3f(0,0,1.);
	}
}

AffineTransform::AffineTransform(Eigen::Matrix3f R, Eigen::Vector3f t) :
	t(t),
	R(R)
{

}
AffineTransform::AffineTransform() :
	t(Eigen::Vector3f::Zero()),
	R(Eigen::Matrix3f::Identity())
{

}

Eigen::Vector3f AffineTransform::operator*(Eigen::Vector3f const& pt) const {
	return R*pt + t;
}
Eigen::Matrix3Xf AffineTransform::operator*(Eigen::Matrix3Xf const& pts) const {
	return applyOnto(pts.array()).matrix();
}
Eigen::Array3Xf AffineTransform::operator*(Eigen::Array3Xf const& pts) const {
	return applyOnto(pts);
}

Eigen::Array3Xf AffineTransform::applyOnto(Eigen::Array3Xf const& pts) const {

	Eigen::Array3Xf transformedPts;
	transformedPts.resize(3, pts.cols());

	for (int i = 0; i < transformedPts.cols(); i++) {
		transformedPts.col(i) = R*(pts.col(i).matrix()) + t;
	}

	return transformedPts;
}

} // namespace Geometry
} // namespace StereoVision
