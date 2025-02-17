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

#include "../utils/types_manipulations.h"

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

    T trace = R(0,0) + R(1,1) + R(2,2);
    T d =  0.5*(trace - T(1));

    if (d < T(-1)) { //necessary for numerical stability.
        d = T(-1);
    }

    Eigen::Matrix<T,3,1> omega;

    Eigen::Matrix<T,3,1> dR = unskew<T>(R - R.transpose());

    T nDr = dR.norm();

    double d_threshold = (TypesManipulations::typeExceedFloat32Precision<T>()) ? 0.999999 : 0.999;
    double nDr_threshold = (TypesManipulations::typeExceedFloat32Precision<T>()) ? 1e-6 : 1e-3;

    if (d>0.999)
    {
      omega=0.5*dR;
    }
    else if (nDr < 1e-3) {
        T theta = acos(d);
        Eigen::Matrix<T,3,3> S = R + R.transpose() + (T(1) - trace)*Eigen::Matrix<T,3,3>::Identity();
        Eigen::Matrix<T,3,1> n;

        for (int i = 0; i < 3; i++) {
            n[i] = sqrt(std::max<T>(S(i,i)/(T(3) - trace), T(0))); //compute the values, up to sign
        }

        if (n[0] > n[1] and n[0] > n[2]) {
            n[1] = (S(0,1)/(T(3) - trace))/n[0];
            n[2] = (S(0,2)/(T(3) - trace))/n[0];
        }

        if (n[1] > n[0] and n[1] > n[2]) {
            n[0] = (S(1,0)/(T(3) - trace))/n[1];
            n[2] = (S(1,2)/(T(3) - trace))/n[1];
        }

        if (n[2] > n[0] and n[2] > n[1]) {
            n[0] = (S(2,0)/(T(3) - trace))/n[2];
            n[1] = (S(2,1)/(T(3) - trace))/n[2];
        }

        omega = theta*n;
    }
    else
    {
      T theta = acos(d);
      omega = theta/(T(2)*sqrt(T(1)-d*d))*dR;
    }

    return omega;
}

/*!
 * \brief diffRodriguezLieAlgebra gives the right Jacobian of SO(3) for a given axis angle vector
 * \param r the rotation axis
 * \return J_{so(3)}(r), the right Jacobian of SO(3)
 *
 * J_{so(3)}(r) relate additional increments in so(3) and multiplicative increments in SO(3) such that:
 * Exp(r + dr) = Exp(r)Exp(J_{so(3)}(r)dr) and Log(Exp(r)Exp(dr)) = r + J_{so(3)}(r)^-1 dr  for small dr
 */
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
        b = -(1 - cos(theta))/(theta*theta);
        c = (1 - a)/(theta*theta);
    } else {
        a = 1;
        b = -1./2.;
        c = 1./6.;
    }

    dR = Eigen::Matrix<T,3,3>::Identity() + b*m + c*(m*m);

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

/*!
 * \brief diffAngleAxisRotate compute the derivative of the angle axis rotate function with respect to the rotation axis.
 * \param r the rotation axis
 * \param v the vector being rotated
 * \param direction the entry in r the derivative is taken with respect to.
 * \return the partial derivative of the function
 */
template<typename T>
Eigen::Matrix<T,3,1> diffAngleAxisRotate(Eigen::Matrix<T,3,1> const& r, Eigen::Matrix<T,3,1> const& v, Axis direction) {

    int axis = static_cast<int>(direction);

    Eigen::Matrix<T,3,1> diffR = pathFromDiff(direction).cast<T>();

    T theta = r.norm();
    T diffTheta = r[axis]/theta;

    Eigen::Matrix<T,3,1> rxv = r.cross(v);
    Eigen::Matrix<T,3,1> diffRxv = diffR.cross(v);


    Eigen::Matrix<T,3,1> diffRcrossRxv = r.cross(diffRxv) + diffR.cross(rxv);

    if (theta < 1e-3) {
        return -r[axis]/3 * rxv + (1-theta*theta/6)*diffRxv +
               -r[axis]/12 * rxv + (0.5 - theta*theta/24)*diffRcrossRxv;
    } //Taylor approximation for small values of theta

    //expansion
    Eigen::Matrix<T,3,1> diffRotated = (cos(theta)/theta - sin(theta)/(theta*theta))*diffTheta*rxv + sin(theta)/theta*diffRxv +
            ((sin(theta))/(theta*theta) - 2*(T(1) - cos(theta))/(theta*theta*theta))*diffTheta*r.cross(rxv) + (T(1) - cos(theta))/(theta*theta)*diffRcrossRxv;
    return diffRotated;

}

template<typename T>
Eigen::Quaternion<T> axisAngleToQuaternion(Eigen::Matrix<T, 3,1> const& axisAngle) {
    T normSquared = axisAngle[0]*axisAngle[0] + axisAngle[1]*axisAngle[1] + axisAngle[2]*axisAngle[2];
    T norm = sqrt(normSquared);

    if (norm < 1e-6) {

        T scale = T(2)/sqrt(normSquared + T(1));

        return Eigen::Quaternion<T>(scale, axisAngle[0]*scale*T(0.5), axisAngle[1]*scale*T(0.5), axisAngle[2]*scale*T(0.5));
    }

    Eigen::Matrix<T, 3,1> normalized = axisAngle/norm;

    return Eigen::Quaternion<T>(Eigen::AngleAxis<T>(norm, normalized));
}


/*!
 * \brief diffAxisAngleToQuaternion compute the jacobian of computing a quaternion from an axis angle
 * \param axisAngle the axis angle
 * \return the Jacobian.
 */
template<typename T>
Eigen::Matrix<T,4,3> diffAxisAngleToQuaternion(Eigen::Matrix<T, 3,1> const& axisAngle) {

    T normSquared = axisAngle[0]*axisAngle[0] + axisAngle[1]*axisAngle[1] + axisAngle[2]*axisAngle[2];
    T norm = sqrt(normSquared);

    Eigen::Matrix<T,4,3> ret;

    if (norm < 1e-5) {

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                ret(j,i) = axisAngle[j] * axisAngle[i] / 24;

                if (i == j) {
                    ret(j,i) += (0.5 - norm*norm/48);
                }
            }

            ret(3,i) = -(0.5 - norm*norm/48) * axisAngle[i] / 2;
        }

        return ret;

    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ret(j,i) = axisAngle[j] * (cos(norm/2)/(2*norm) - sin(norm/2)/(norm*norm)) * axisAngle[i]/norm;

            if (i == j) {
                ret(j,i) += sin(norm/2)/norm;
            }
        }

        ret(3,i) = -sin(norm/2) * axisAngle[i]/norm / 2;
    }

    return ret;
}

template<typename T>
Eigen::Matrix<T, 3,1> quaternionToAxisAngle(Eigen::Quaternion<T> const& quaternion) {
    Eigen::AngleAxis<T> angleAxis(quaternion);
    return angleAxis.axis()*angleAxis.angle();
}


template<typename T>
class RigidBodyTransform
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RigidBodyTransform(Eigen::Matrix<T,3,1> r, Eigen::Matrix<T,3,1> t) :
        t(t), r(r)
    {

    }
    RigidBodyTransform(Eigen::Matrix<T,6,1> p) :
        t(p.template block<3,1>(3,0)), r(p.template block<3,1>(0,0))
    {

    }
    RigidBodyTransform() :
        t(Eigen::Matrix<T,3,1>::Zero()), r(Eigen::Matrix<T,3,1>::Zero())
    {

    }

    /*!
     * \brief RigidBodyTransform constructor from an AffineTransform
     * \param affine the affine transform the RigidBodyTransform is constructed from
     *
     * The RigidBodyTransform is constructed by using inverseRodriguezFormula on the affine transform matrix to get an axis angle.
     */
    RigidBodyTransform(AffineTransform<T> const& affine) :
        t(affine.t), r(inverseRodriguezFormula(affine.R))
    {

    }

    Eigen::Matrix<T,3,1> operator*(Eigen::Matrix<T,3,1> const& pt) const {
        return angleAxisRotate(r,pt) + t;
    }

    template <int nCols>
    std::enable_if_t<nCols!=1, Eigen::Matrix<T,3,nCols>> operator*(Eigen::Matrix<T,3,nCols> const& pts) const {
        return applyOnto<nCols>(pts.array()).matrix();
    }

    template <int nCols>
    Eigen::Array<T,3,Eigen::Dynamic> operator*(Eigen::Array<T,3,Eigen::Dynamic> const& pts) const {
        return applyOnto<nCols>(pts);
    }

    /*!
     * \brief operator * compose two RigidBodyTransform
     * \param other the other RigidBodyTransform to compose with
     * \return the composed transform (equivalent to applying other, followed by this)
     */
    RigidBodyTransform<T> operator*(RigidBodyTransform<T> const& other) const {
        Eigen::Matrix<T,3,3> R = rodriguezFormula(r);
        Eigen::Matrix<T,3,3> Rc = R*rodriguezFormula(other.r);
        return RigidBodyTransform(inverseRodriguezFormula(Rc), R*other.t + t);
    }

    /*!
     * \brief operator * scale the transform with a scalar coefficient
     * \param scale the scaling scalar coefficient
     * \return the scaled transform
     *
     * This function is usefull when using RigidBodyTransforms for interpolation in se(3)
     */
    RigidBodyTransform<T> operator*(T const& scale) const {
        return RigidBodyTransform(scale*r, scale*t);
    }

    /*!
     * \brief operator + add two RigidBodyTransforms together, considering them as vectors in se(3).
     * \param other the RigidBodyTransform to add
     * \return the addition of two RigidBodyTransform
     *
     * This function is usefull when using RigidBodyTransforms for interpolation in se(3)
     */
    RigidBodyTransform<T> operator+(RigidBodyTransform<T> const& other) {
        return RigidBodyTransform(other.r + r, other.t + t);
    }

    AffineTransform<T> toAffineTransform() const {
        return AffineTransform<T>(rodriguezFormula(r), t);
    }
    RigidBodyTransform<T> inverse() const {
        return RigidBodyTransform<T>(-r, -angleAxisRotate<T>(-r, t));
    }

    template <typename Tc>
    RigidBodyTransform<Tc> cast() const {
        return RigidBodyTransform<Tc>(r.template cast<Tc>(), t.template cast<Tc>());
    }

    /*!
     * \brief Jacobian compute the jacobian of the function Gamma*v, where Gamma is the shape preserving transform, with respect to the r, t and s parameters
     * \param v the vector the jacobian should be evaluated at
     * \return the jacobian
     */
    Eigen::Matrix<T,3,6> Jacobian(Eigen::Matrix<T,3,1> const& v) {
        Eigen::Matrix<T,3,6> J;

        J.template block<3,1>(0,0) = diffAngleAxisRotate(r, v, Axis::X);
        J.template block<3,1>(0,1) = diffAngleAxisRotate(r, v, Axis::Y);
        J.template block<3,1>(0,2) = diffAngleAxisRotate(r, v, Axis::Z);

        J.template block<3,3>(0,3) = Eigen::Matrix<T,3,3>::Identity();

        return J;
    }

    inline bool isFinite() const {
        return t.array().isFinite().all() and r.array().isFinite().all();
    }

    Eigen::Matrix<T,3,1> t;
    Eigen::Matrix<T,3,1> r;

protected:

    template <int nCols>
    Eigen::Array<T,3,nCols> applyOnto(Eigen::Array<T,3,nCols> const& pts) const {
        Eigen::Array3Xf transformedPts;
        transformedPts.resize(3, pts.cols());

        for (int i = 0; i < transformedPts.cols(); i++) {
            transformedPts.col(i) = angleAxisRotate<T>(r, pts.col(i).matrix()) + t;
        }

        return transformedPts;
    }
};

template<typename T>
RigidBodyTransform<T> operator*(T scale, RigidBodyTransform<T> transform) {
    return transform.operator*(scale);
}

/*!
 * \brief interpolateRigidBodyTransformOnManifold interpolate between two RigidBodyTransform on the manifold
 * \param w1 weight for t1
 * \param t1 transformation 1
 * \param w2 weight for t2
 * \param t2 transformation 2
 * \return the linear interpolation between t1 and t2, using weights w1 and w2 and done on the manifold (se(3)).
 */
template<typename T>
RigidBodyTransform<T> interpolateRigidBodyTransformOnManifold(
        T w1,
        RigidBodyTransform<T> const& t1,
        T w2,
        RigidBodyTransform<T> const& t2) {

    RigidBodyTransform<T> transform = t2*t1.inverse();

    T w = w2 / (w1 + w2);

    return (w*transform)*t1;

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
    ShapePreservingTransform(Eigen::Matrix<T,7,1> p) :
        t(p.template block<3,1>(3,0)), r(p.template block<3,1>(0,0)), s(p[6])
    {

    }
    ShapePreservingTransform() :
        t(Eigen::Matrix<T,3,1>::Zero()), r(Eigen::Matrix<T,3,1>::Zero()), s(1.)
    {

    }

    Eigen::Matrix<T,3,1> operator*(Eigen::Matrix<T,3,1> const& pt) const {
        return s*angleAxisRotate(r,pt) + t;
    }

    template <int nCols>
    std::enable_if_t<nCols!=1, Eigen::Matrix<T,3,nCols>> operator*(Eigen::Matrix<T,3,nCols> const& pts) const {
        return applyOnto<nCols>(pts.array()).matrix();
    }

    template <int nCols>
    Eigen::Array<T,3,Eigen::Dynamic> operator*(Eigen::Array<T,3,Eigen::Dynamic> const& pts) const {
        return applyOnto<nCols>(pts);
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
        return ShapePreservingTransform<T>(-r, -angleAxisRotate<T>(-r, t/s), 1/s);
    }

    template <typename Tc>
    ShapePreservingTransform<Tc> cast() const {
        return ShapePreservingTransform<Tc>(r.template cast<Tc>(), t.template cast<Tc>(), static_cast<Tc>(s));
    }

    /*!
     * \brief Jacobian compute the jacobian of the function Gamma*v, where Gamma is the shape preserving transform, with respect to the r, t and s parameters
     * \param v the vector the jacobian should be evaluated at
     * \return the jacobian
     */
    Eigen::Matrix<T,3,7> Jacobian(Eigen::Matrix<T,3,1> const& v) {
        Eigen::Matrix<T,3,7> J;

        J.template block<3,1>(0,0) = diffAngleAxisRotate(r, v, Axis::X);
        J.template block<3,1>(0,1) = diffAngleAxisRotate(r, v, Axis::Y);
        J.template block<3,1>(0,2) = diffAngleAxisRotate(r, v, Axis::Z);

        J.template block<3,3>(0,0) *= s;

        J.template block<3,3>(0,3) = Eigen::Matrix<T,3,3>::Identity();

        J.template block<3,1>(0,6) = angleAxisRotate(r,v);

        return J;
    }

    inline bool isFinite() const {
        return t.array().isFinite().all() and r.array().isFinite().all() and std::isfinite(s);
    }

    Eigen::Matrix<T,3,1> t;
    Eigen::Matrix<T,3,1> r;
    T s;

protected:

    template <int nCols>
    Eigen::Array<T,3,nCols> applyOnto(Eigen::Array<T,3,nCols> const& pts) const {
        Eigen::Array3Xf transformedPts;
        transformedPts.resize(3, pts.cols());

        for (int i = 0; i < transformedPts.cols(); i++) {
            transformedPts.col(i) = s*angleAxisRotate<T>(r, pts.col(i).matrix()) + t;
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

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> rMat2eulerRadxyz(Eigen::Matrix<Scalar, 3, 3> const& RMat) {
    // assuming RMat = RMatx RMaty RMatz; Solution is:
    //        [      cy cz            - cy sz         sy    ]
    //        [                                             ]
    //        [ cx sz + cz sx sy  cx cz - sx sy sz  - cy sx ]
    //        [                                             ]
    //        [ sx sz - cx cz sy  cx sy sz + cz sx   cx cy  ]

    double sinY = RMat(0,2);
    double cosY = sqrt(1 - sinY*sinY);

    double sinZ = -RMat(0,1)/cosY;
    double sinX = -RMat(1,2)/cosY;

    if (!std::isfinite(sinZ) or ! std::isfinite(sinX)) {
        //we have RMat =
        //        [       0               0         +/-1]
        //        [                                     ]
        //        [ cx sz +/- cz sx  cx cz +/- sx sz   0]
        //        [                                     ]
        //        [ sx sz +/- cx cz  +/- cx sz + cz sx 0]

        if (sinY > 0) {
            sinY = 1;
        } else {
            sinY = -1;
        }

        double cosZ = RMat(1,1); //valid, if we assume X = 0, which we can

        //acceptable, as any
        sinX = 0;
        sinZ = sqrt(1 - cosZ*cosZ);
    }

    double X = std::asin(sinX);
    double Y = std::asin(sinY);
    double Z = std::asin(sinZ);

    return Eigen::Matrix<Scalar, 3, 1>(X,Y,Z);
}

} // namespace Geometry
} //namespace StereoVision

#endif // ROTATIONS_H
