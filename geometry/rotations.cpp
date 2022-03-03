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

#include "rotations.h"

namespace StereoVision {
namespace Geometry {

ShapePreservingTransform::ShapePreservingTransform(Eigen::Vector3f r, Eigen::Vector3f t, float s) :
	t(t), r(r), s(s)
{

}

ShapePreservingTransform::ShapePreservingTransform() :
	t(Eigen::Vector3f::Zero()), r(Eigen::Vector3f::Zero()), s(1.)
{

}

Eigen::Vector3f ShapePreservingTransform::operator*(Eigen::Vector3f const& pt) const {
	return s*rodriguezFormula(r)*pt + t;
}
Eigen::Matrix3Xf ShapePreservingTransform::operator*(Eigen::Matrix3Xf const& pts) const {
	return applyOnto(pts.array()).matrix();
}
Eigen::Array3Xf ShapePreservingTransform::operator*(Eigen::Array3Xf const& pts) const {
	return applyOnto(pts);
}
ShapePreservingTransform ShapePreservingTransform::operator*(ShapePreservingTransform const& other) const {
	//s*rodriguezFormula(r)*(other.s*rodriguezFormula(other.r)*pt + other.t) + t;
	//s*other.s*rodriguezFormula(r)*rodriguezFormula(other.r)*pt + s*rodriguezFormula(r)*other.t + t

	Eigen::Matrix3f R = rodriguezFormula(r);
	Eigen::Matrix3f Rc = R*rodriguezFormula(other.r);
	return ShapePreservingTransform(inverseRodriguezFormula(Rc), s*R*other.t + t, s*other.s);
}

AffineTransform ShapePreservingTransform::toAffineTransform() const {
	return AffineTransform(s*rodriguezFormula(r), t);
}

ShapePreservingTransform ShapePreservingTransform::inverse() const {
	return ShapePreservingTransform(-r, - rodriguezFormula(r).transpose()*t/s, 1/s);
}

Eigen::Array3Xf ShapePreservingTransform::applyOnto(Eigen::Array3Xf const& pts) const {

	Eigen::Array3Xf transformedPts;
	transformedPts.resize(3, pts.cols());

	for (int i = 0; i < transformedPts.cols(); i++) {
		transformedPts.col(i) = s*rodriguezFormula(r)*(pts.col(i).matrix()) + t;
	}

	return transformedPts;
}

Eigen::Matrix3f rodriguezFormula(Eigen::Vector3f const& r)
{
	float theta = r.norm();
	Eigen::Matrix3f m = skew(r);

	Eigen::Matrix3f R;

	if (theta > 1e-6) {
		R = Eigen::Matrix3f::Identity() + sin(theta)/theta*m + (1 - cos(theta))/(theta*theta)*m*m;
	} else {
		R = Eigen::Matrix3f::Identity() + m + 0.5*m*m;
	}

	return R;
}

Eigen::Vector3f inverseRodriguezFormula(Eigen::Matrix3f const& R) {

	float d =  0.5*(R(0,0) + R(1,1) + R(2,2) - 1);
	Eigen::Vector3f omega;

	Eigen::Vector3f dR = unskew(R - R.transpose());

	float nDr = dR.norm();

	if (d>0.999)
	{
	  omega=0.5*dR;
	}
	else if (nDr < 1e-3) {
		float theta = acos(d);
		Eigen::Vector3f d = R.diagonal();
		omega = theta*(d - Eigen::Vector3f::Ones()*d.minCoeff())/(1 - d.minCoeff());
	}
	else
	{
	  float theta = acos(d);
	  omega = theta/(2*sqrt(1-d*d))*dR;
	}

	return omega;
}

Eigen::Matrix3f diffRodriguezLieAlgebra(Eigen::Vector3f const& r)
{
	float theta = r.norm();
	Eigen::Matrix3f m = skew(r);

	Eigen::Matrix3f dR;

	float a;
	float b;
	float c;

	if (theta > 1e-6) {
		a = sin(theta)/theta;
		b = (1 - cos(theta))/(theta*theta);
		c = (1 - a)/(theta*theta);
	} else {
		a = 1;
		b = 1./2.;
		c = 1./6.;
	}

	dR = a*Eigen::Matrix3f::Identity() + b*m + c*(r*r.transpose());

	return dR;

}
Eigen::Matrix3f diffRodriguez(Eigen::Vector3f const& r, Axis direction) {
	float theta = r.norm();

	int i = (direction == Axis::X) ? 0 : ((direction == Axis::Y) ? 1 : 2);

	Eigen::Vector3f eye = Eigen::Vector3f::Zero();
	eye[i] = 1;

	if (theta < 1e-6) {
		return skew(eye);
	}

	Eigen::Matrix3f cross = skew(r);
	Eigen::Matrix3f R = rodriguezFormula(r);
	return (r[i]*cross + skew(cross*(eye - R.col(i))))/(theta*theta) * R;

}

Eigen::Matrix3d rodriguezFormulaD(Eigen::Vector3d const& r)
{
	double theta = r.norm();
	Eigen::Matrix3d m = skewD(r);

	Eigen::Matrix3d R;

	if (theta > 1e-6) {
		R = Eigen::Matrix3d::Identity() + sin(theta)/theta*m + (1 - cos(theta))/(theta*theta)*m*m;
	} else {
		R = Eigen::Matrix3d::Identity() + m + 0.5*m*m;
	}

	return R;
}
Eigen::Vector3d inverseRodriguezFormulaD(Eigen::Matrix3d const& R) {

	double d =  0.5*(R(0,0) + R(1,1) + R(2,2) - 1);
	Eigen::Vector3d omega;

	Eigen::Vector3d dR = unskewD(R - R.transpose());

	double nDr = dR.norm();

	if (d>0.999999)
	{
	  omega=0.5*dR;
	}
	else if (nDr < 1e-6) {
		float theta = acos(d);
		Eigen::Vector3d d = R.diagonal();
		omega = theta*(d - Eigen::Vector3d::Ones()*d.minCoeff())/(1 - d.minCoeff());
	}
	else
	{
	  double theta = acos(d);
	  omega = theta/(2*sqrt(1-d*d))*dR;
	}

	return omega;
}

Eigen::Matrix3d diffRodriguezLieAlgebraD(Eigen::Vector3d const& r) {

	float theta = r.norm();
	Eigen::Matrix3d m = skewD(r);

	Eigen::Matrix3d dR;

	double a;
	double b;
	double c;

	if (theta > 1e-6) {
		a = sin(theta)/theta;
		b = (1 - cos(theta))/(theta*theta);
		c = (1 - a)/(theta*theta);
	} else {
		a = 1;
		b = 1./2.;
		c = 1./6.;
	}

	dR = a*Eigen::Matrix3d::Identity() + b*m + c*(r*r.transpose());

	return dR;
}

Eigen::Matrix3d diffRodriguezD(Eigen::Vector3d const& r, Axis direction) {

	double theta = r.norm();

	int i = (direction == Axis::X) ? 0 : ((direction == Axis::Y) ? 1 : 2);

	Eigen::Vector3d eye = Eigen::Vector3d::Zero();
	eye[i] = 1;

	if (theta < 1e-8) {
		return skewD(eye);
	}

	Eigen::Matrix3d cross = skewD(r);
	Eigen::Matrix3d R = rodriguezFormulaD(r);
	return (r[i]*cross + skewD(cross*(eye - R.col(i))))/(theta*theta) * R;
}

} // namespace Geometry
} // namespace StereoVision
