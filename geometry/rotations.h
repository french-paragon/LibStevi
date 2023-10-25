#ifndef STEREOVISION_ROTATIONS_H
#define STEREOVISION_ROTATIONS_H

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

#include "../geometry/core.h"
#include <Eigen/Geometry>

namespace StereoVision {
namespace Geometry {

template<typename T>
Eigen::Matrix<T,3,3> rodriguezFormula(Eigen::Matrix<T,3,1> const& r) {
    T theta = r.norm();
    Eigen::Matrix<T,3,3> m = skew(r);

    Eigen::Matrix<T,3,3> R;

    if (theta > 1e-6) {
        R = Eigen::Matrix<T,3,3>::Identity() + sin(theta)/theta*m + (T(1) - cos(theta))/(theta*theta)*m*m;
    } else {
        R = Eigen::Matrix<T,3,3>::Identity() + m + 0.5*m*m;
    }

    return R;
}

template<typename T>
Eigen::Matrix<T,3,1> angleAxisRotate(Eigen::Matrix<T,3,1> const& r, Eigen::Matrix<T,3,1> const& v) {
    T theta = r.norm();

    Eigen::Matrix<T,3,1> rxv = r.cross(v);

    if (theta < 1e-6) {
        return v + rxv + 0.5*r.cross(rxv);
    }

    Eigen::Matrix<T,3,1> rotated = v + sin(theta)/theta*rxv + (T(1) - cos(theta))/(theta*theta)*r.cross(rxv);
    return rotated;
}

template<typename T>
Eigen::Matrix<T,3,1> inverseRodriguezFormula(Eigen::Matrix<T,3,3> const& R) {

    T d =  0.5*(R(0,0) + R(1,1) + R(2,2) - T(1));
    Eigen::Matrix<T,3,1> omega;

    Eigen::Matrix<T,3,1> dR = unskew<T>(R - R.transpose());

    T nDr = dR.norm();

    if (d>0.999)
    {
      omega=0.5*dR;
    }
    else if (nDr < 1e-3) {
        T theta = acos(d);
        Eigen::Matrix<T,3,1> d = R.diagonal();
        omega = theta*(d - Eigen::Matrix<T,3,1>::Ones()*d.minCoeff())/(T(1) - d.minCoeff());
    }
    else
    {
      T theta = acos(d);
      omega = theta/(T(2)*sqrt(T(1)-d*d))*dR;
    }

    return omega;
}

template<typename T>
Eigen::Matrix<T,3,3> diffRodriguezLieAlgebra(Eigen::Matrix<T,3,1> const& r) {

    float theta = r.norm();
    Eigen::Matrix<T,3,3> m = skew(r);

    Eigen::Matrix<T,3,3> dR;

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

    dR = a*Eigen::Matrix<T,3,3>::Identity() + b*m + c*(r*r.transpose());

    return dR;
}

template<typename T>
Eigen::Matrix<T,3,3> diffRodriguez(Eigen::Matrix<T,3,1> const& r, Axis direction) {

    float theta = r.norm();

    int i = (direction == Axis::X) ? 0 : ((direction == Axis::Y) ? 1 : 2);

    Eigen::Matrix<T,3,1> eye = Eigen::Matrix<T,3,1>::Zero();
    eye[i] = 1;

    if (theta < 1e-6) {
        return skew(eye);
    }

    Eigen::Matrix<T,3,3> cross = skew(r);
    Eigen::Matrix<T,3,3> R = rodriguezFormula(r);
    return (r[i]*cross + skew<T>(cross*(eye - R.col(i))))/(theta*theta) * R;
}

template<typename T>
class ShapePreservingTransform
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ShapePreservingTransform(Eigen::Matrix<T,3,1> r, Eigen::Matrix<T,3,1> t, T s) :
        t(t), r(r), s(s)
    {

    }
    ShapePreservingTransform() :
        t(Eigen::Matrix<T,3,1>::Zero()), r(Eigen::Matrix<T,3,1>::Zero()), s(1.)
    {

    }

    Eigen::Matrix<T,3,1> operator*(Eigen::Matrix<T,3,1> const& pt) const {
        return s*rodriguezFormula(r)*pt + t;
    }
    Eigen::Matrix<T,3,Eigen::Dynamic> operator*(Eigen::Matrix<T,3,Eigen::Dynamic> const& pts) const {
        return applyOnto(pts.array()).matrix();
    }
    Eigen::Array<T,3,Eigen::Dynamic> operator*(Eigen::Array<T,3,Eigen::Dynamic> const& pts) const {
        return applyOnto(pts);
    }

    ShapePreservingTransform<T> operator*(ShapePreservingTransform<T> const& other) const {
        Eigen::Matrix<T,3,3> R = rodriguezFormula(r);
        Eigen::Matrix<T,3,3> Rc = R*rodriguezFormula(other.r);
        return ShapePreservingTransform(inverseRodriguezFormula(Rc), s*R*other.t + t, s*other.s);
    }

    AffineTransform<T> toAffineTransform() const {
        return AffineTransform<T>(s*rodriguezFormula(r), t);
    }
    ShapePreservingTransform<T> inverse() const {
        return ShapePreservingTransform<T>(-r, - rodriguezFormula(r).transpose()*t/s, 1/s);
    }

    inline bool isFinite() const {
        return t.array().isFinite().all() and r.array().isFinite().all() and std::isfinite(s);
    }

    Eigen::Matrix<T,3,1> t;
    Eigen::Matrix<T,3,1> r;
    T s;

protected:

    Eigen::Array<T,3,Eigen::Dynamic> applyOnto(Eigen::Array<T,3,Eigen::Dynamic> const& pts) const {
        Eigen::Array3Xf transformedPts;
        transformedPts.resize(3, pts.cols());

        for (int i = 0; i < transformedPts.cols(); i++) {
            transformedPts.col(i) = s*rodriguezFormula(r)*(pts.col(i).matrix()) + t;
        }

        return transformedPts;
    }
};



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
    return eulerRadXYZToRotation(Scalar(eulerX/180*M_PI), Scalar(eulerY/180*M_PI), Scalar(eulerZ/180*M_PI));
}

} // namespace Geometry
} //namespace StereoVision

#endif // ROTATIONS_H
