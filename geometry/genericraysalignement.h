#ifndef GENERICRAYSALIGNEMENT_H
#define GENERICRAYSALIGNEMENT_H
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

#include <Eigen/Core>

#include "../geometry/rotations.h"
#include "../utils/iterative_numerical_algorithm_output.h"
#include "../optimization/huber_kernel.h"
#include "../optimization/gaussnewtownproblem.h"
#include <optional>

namespace StereoVision {
namespace Geometry {

/*!
 * \brief The RayInfos struct store the origin and direction of a ray.
 */
template<typename T>
struct RayInfos {

    inline RayInfos(Eigen::Matrix<T,3,1> const& origin, Eigen::Matrix<T,3,1> const& direction) :
        localSystemRayOrigin(origin),
        localSystemRayDirection(direction)
    {

    }

    Eigen::Matrix<T,3,1> localSystemRayOrigin;
    Eigen::Matrix<T,3,1> localSystemRayDirection;
};

/*!
 * \brief The RayPairInfos struct store the information about a pairs of rays and the rigid body transform between their frame of reference
 */
template<typename T>
struct RayPairInfos {

    Eigen::Matrix<T,3,1> v1;
    Eigen::Matrix<T,3,1> v2;
    Eigen::Matrix<T,3,3> R1_to_2;
    Eigen::Matrix<T,3,1> t;
};

/*!
 * \brief alignRaysSets aligns two rays sets
 * \param raySet1 the first ray set
 * \param raySet2 the second ray set
 * \param initialSolution the initial solution
 * \param maxIter the maximum number of iteration
 * \param tol the relative tolerance of the increment.
 * \return the estimated transform from set 1 to set 2
 */
template<typename T>
IterativeNumericalAlgorithmOutput<RigidBodyTransform<T>> alignRaysSets(std::vector<RayInfos<T>> const& raySet1,
                                                   std::vector<RayInfos<T>> const& raySet2,
                                                   RigidBodyTransform<T> const& initialSolution = RigidBodyTransform<T>(Eigen::Matrix<T,3,1>::Zero(), Eigen::Matrix<T,3,1>::Zero()),
                                                   int maxIter = 50,
                                                   T tol = 1e-2) {

    using MatrixAType = Eigen::Matrix<T, Eigen::Dynamic, 6>;
    using MatrixbType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using MatrixxType = Eigen::Matrix<T, 6, 1>;

    if (raySet1.size() != raySet2.size()) {
        return IterativeNumericalAlgorithmOutput<RigidBodyTransform<T>>(RigidBodyTransform<T>(), ConvergenceType::Failed);
    }

    int nObs = raySet1.size();

    //Problem we are trying to solve is:
    // Minimize Sum_i || raySet2[i].localSystemRayDirection * (R*raySet1[i].localSystemRayOrigin + t) x R*raySet1[i].localSystemRayDirection ||
    // under the constraint that R is a rotation matrix (we enforce this using rodriguez representation).

    RigidBodyTransform<T> sol = initialSolution;
    ConvergenceType conv = ConvergenceType::MaxIterReached;

    for (int i = 0; i < maxIter; i++) {

        MatrixbType res;
        res.resize(nObs,1);

        MatrixAType A;
        A.resize(nObs,6);

        for (int i = 0; i < nObs; i++) {

            Eigen::Matrix<T,3,1> const& pi = raySet1[i].localSystemRayOrigin;
            Eigen::Matrix<T,3,1> const& vi = raySet1[i].localSystemRayDirection;

            Eigen::Matrix<T,3,1> const& pj = raySet2[i].localSystemRayOrigin;
            Eigen::Matrix<T,3,1> const& vj = raySet2[i].localSystemRayDirection;

            Eigen::Matrix<T,3,1> Rpi = angleAxisRotate(sol.r, pi);
            Eigen::Matrix<T,3,1> Rvi = angleAxisRotate(sol.r, vi);

            res[i] = vj.dot((Rpi + sol.t - pj).cross(Rvi));

            A(i,0) = vj.dot(diffAngleAxisRotate(sol.r, pi, Axis::X).cross(Rvi) + (Rpi + sol.t - pj).cross(diffAngleAxisRotate(sol.r, vi, Axis::X)));
            A(i,1) = vj.dot(diffAngleAxisRotate(sol.r, pi, Axis::Y).cross(Rvi) + (Rpi + sol.t - pj).cross(diffAngleAxisRotate(sol.r, vi, Axis::Y)));
            A(i,2) = vj.dot(diffAngleAxisRotate(sol.r, pi, Axis::Z).cross(Rvi) + (Rpi + sol.t - pj).cross(diffAngleAxisRotate(sol.r, vi, Axis::Z)));
            A(i,3) = vj.dot(Eigen::Matrix<T,3,1>::Ones().cross(Rvi));
            A(i,4) = vj.dot(Eigen::Matrix<T,3,1>::Ones().cross(Rvi));
            A(i,5) = vj.dot(Eigen::Matrix<T,3,1>::Ones().cross(Rvi));

        }

        MatrixxType dx = A.colPivHouseholderQr().solve(res);

        sol.r -= dx.template block<3,1>(0,0);
        sol.t -= dx.template block<3,1>(3,0);

        if (dx.norm() < tol) {
            conv = ConvergenceType::Converged;
            break;
        }

    }

    return IterativeNumericalAlgorithmOutput<RigidBodyTransform<T>>(sol, conv);

}

/*!
 * \brief relaxedAlignRaysSets try to align two ray sets using a generic lineat transformation
 * \param raySet1 the first ray set
 * \param raySet2 the second ray set
 * \return the best Affine transforms mapping rayset1 to rayset2 (in case the problem is underdetermined, estimate the lowest frobenius norm transformation delta from identity).
 *
 * The relaxed linear approximation has 18 parameters (9 for the rotation matrix and 9 for a combination of translation and rotation).
 *
 * The model aim at solving the epipolar constraint <vj, (Rpi + t - pj) x Rvi> = <vj, R(pi x vi) + txRvi - pj x Rvi> = <vj, R(pi x vi) + Mvi - pj x Rvi> = 0,
 * where pi, vi are the position and direction of the ray in the first ray set frame of reference,
 * pj, vj are the position and direction of the ray in the second ray set frame of reference,
 * R is the matrix of the affine transform (assumed to be a rotation, which allow to simplify Rpi x Rvi = R(pi x vi)),
 * t is the translation of the afine transform,
 * M is the matrix representing the linear transform (with respect to v) txRv.
 *
 * The relaxed version of the problem consider the 9 coefficient of R and M as unknown x and solve the resulting linear equation Ax = 0.
 * t is latter extracted from M (in the least square sense), once R is estimated
 * The solution is taken as the singular vector of A with the lowest singular value, scaled such that det(R) = 1.
 * In case multiple singular values are 0, one is selected arbitrarily (TODO: look if we can find the one closest to a rotation matrix).
 */
template<typename T>
std::optional<AffineTransform<T>> relaxedAlignRaysSets(std::vector<RayInfos<T>> const& raySet1,
                                                       std::vector<RayInfos<T>> const& raySet2) {

    constexpr int nParams = 9+9; //9 parameters for rotation matrix, 9 for translation and rotation combination

    using MatrixAType = Eigen::Matrix<T, Eigen::Dynamic, nParams>;
    using VectorbType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using VectorxType = Eigen::Matrix<T, nParams, 1>;

    if (raySet1.size() != raySet2.size()) {
        return std::nullopt;
    }

    int nObs = raySet1.size();

    //Problem we are trying to solve is:
    // Minimize Sum_i || raySet2[i].localSystemRayDirection * (R*raySet1[i].localSystemRayOrigin + t) x R*raySet1[i].localSystemRayDirection ||
    // under the constraint that R is a rotation matrix (we enforce this using rodriguez representation).

    MatrixAType A;
    A.resize(nObs, nParams);

    for (int i = 0; i < nObs; i++) {

        Eigen::Matrix<T,3,1> const& pi = raySet1[i].localSystemRayOrigin;
        Eigen::Matrix<T,3,1> const& vi = raySet1[i].localSystemRayDirection;

        Eigen::Matrix<T,3,1> const& pj = raySet2[i].localSystemRayOrigin;
        Eigen::Matrix<T,3,1> const& vj = raySet2[i].localSystemRayDirection;

        //Eigen::Matrix<T,3,1> Rpi = R*pi;
        //Eigen::Matrix<T,3,1> Rvi = R*vi;

        Eigen::Matrix<T,3,1> pixvi = pi.cross(vi);

        A(i,0) = vj[0]*pixvi[0] - vj[1]*pj[2]*vi[0] + vj[2]*pj[1]*vi[0]; //R(0,0)
        A(i,1) = vj[0]*pixvi[1] - vj[1]*pj[2]*vi[1] + vj[2]*pj[1]*vi[1]; //R(0,1)
        A(i,2) = vj[0]*pixvi[2] - vj[1]*pj[2]*vi[2] + vj[2]*pj[1]*vi[2]; //R(0,2)
        A(i,3) = vj[1]*pixvi[0] - vj[2]*pj[0]*vi[0] + vj[0]*pj[2]*vi[0]; //R(1,0)
        A(i,4) = vj[1]*pixvi[1] - vj[2]*pj[0]*vi[1] + vj[0]*pj[2]*vi[1]; //R(1,1)
        A(i,5) = vj[1]*pixvi[2] - vj[2]*pj[0]*vi[2] + vj[0]*pj[2]*vi[2]; //R(1,2)
        A(i,6) = vj[2]*pixvi[0] - vj[0]*pj[1]*vi[0] + vj[1]*pj[0]*vi[0]; //R(2,0)
        A(i,7) = vj[2]*pixvi[1] - vj[0]*pj[1]*vi[1] + vj[1]*pj[0]*vi[1]; //R(2,1)
        A(i,8) = vj[2]*pixvi[2] - vj[0]*pj[1]*vi[2] + vj[1]*pj[0]*vi[2]; //R(2,2)

        A(i,9) = vj[0]*vi[0]; //M(0,0)
        A(i,10) = vj[0]*vi[1]; //M(0,1)
        A(i,11) = vj[0]*vi[2]; //M(0,2)
        A(i,12) = vj[1]*vi[0]; //M(0,0)
        A(i,13) = vj[1]*vi[1]; //M(0,1)
        A(i,14) = vj[1]*vi[2]; //M(0,2)
        A(i,15) = vj[2]*vi[0]; //M(0,0)
        A(i,16) = vj[2]*vi[1]; //M(0,1)
        A(i,17) = vj[2]*vi[2]; //M(0,2)

    }

    Eigen::JacobiSVD<MatrixAType> svd(A, Eigen::ComputeFullV);

    int m = svd.matrixV().cols();

    VectorbType singularValues = svd.singularValues();
    VectorxType minSingularVector = svd.matrixV().col(m-1); //last singular vector is the one associated with the smallest singular value.

    constexpr T opTol = 1e-7;
    int delta = 1;

    //Remove the cases where R = 0 are posible
    while (minSingularVector.template block<9,1>(0,0).norm() < opTol and delta < m) {
        delta += 1;
        minSingularVector = svd.matrixV().col(m-delta);
    }

    if (minSingularVector.template block<9,1>(0,0).norm() < opTol) {
        return std::nullopt; //problem is impossible to solve
    }

    Eigen::Matrix<T,3,3> unscaledR;
    Eigen::Matrix<T,3,3> unscaledM;

    unscaledR(0,0) = minSingularVector[0];
    unscaledR(0,1) = minSingularVector[1];
    unscaledR(0,2) = minSingularVector[2];
    unscaledR(1,0) = minSingularVector[3];
    unscaledR(1,1) = minSingularVector[4];
    unscaledR(1,2) = minSingularVector[5];
    unscaledR(2,0) = minSingularVector[6];
    unscaledR(2,1) = minSingularVector[7];
    unscaledR(2,2) = minSingularVector[8];

    unscaledM(0,0) = minSingularVector[9];
    unscaledM(0,1) = minSingularVector[10];
    unscaledM(0,2) = minSingularVector[11];
    unscaledM(1,0) = minSingularVector[12];
    unscaledM(1,1) = minSingularVector[13];
    unscaledM(1,2) = minSingularVector[14];
    unscaledM(2,0) = minSingularVector[15];
    unscaledM(2,1) = minSingularVector[16];
    unscaledM(2,2) = minSingularVector[17];

    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> unscaledRsvd(unscaledR);

    T invScale = unscaledRsvd.singularValues().mean();
    T scale = 1;

    if (std::isfinite(1/invScale)) {
        scale /= invScale;
    }

    Eigen::Matrix<T,3,3> R = scale*unscaledR;

    Eigen::Matrix<T,9,3> tObsMat;
    Eigen::Matrix<T,9,1> MVec;

    //M(0,0)
    MVec[0] = unscaledM(0,0);
    tObsMat(0,0) = 0;
    tObsMat(0,1) = unscaledR(2,0);
    tObsMat(0,2) = -unscaledR(1,0);
    //M(0,1)
    MVec[1] = unscaledM(0,1);
    tObsMat(1,0) = 0;
    tObsMat(1,1) = unscaledR(2,1);
    tObsMat(1,2) = -unscaledR(1,1);
    //M(0,2)
    MVec[2] = unscaledM(0,2);
    tObsMat(2,0) = 0;
    tObsMat(2,1) = unscaledR(2,2);
    tObsMat(2,2) = -unscaledR(1,2);
    //M(1,0)
    MVec[3] = unscaledM(1,0);
    tObsMat(3,0) = -unscaledR(2,0);
    tObsMat(3,1) = 0;
    tObsMat(3,2) = unscaledR(0,0);
    //M(1,1)
    MVec[4] = unscaledM(1,1);
    tObsMat(4,0) = -unscaledR(2,1);
    tObsMat(4,1) = 0;
    tObsMat(4,2) = unscaledR(0,1);
    //M(1,2)
    MVec[5] = unscaledM(1,2);
    tObsMat(5,0) = -unscaledR(2,2);
    tObsMat(5,1) = 0;
    tObsMat(5,2) = unscaledR(0,2);
    //M(2,0)
    MVec[6] = unscaledM(2,0);
    tObsMat(6,0) = unscaledR(1,0);
    tObsMat(6,1) = -unscaledR(0,0);
    tObsMat(6,2) = 0;
    //M(2,1)
    MVec[7] = unscaledM(2,1);
    tObsMat(7,0) = unscaledR(1,1);
    tObsMat(7,1) = -unscaledR(0,1);
    tObsMat(7,2) = 0;
    //M(2,2)
    MVec[8] = unscaledM(2,2);
    tObsMat(8,0) = unscaledR(1,2);
    tObsMat(8,1) = -unscaledR(0,2);
    tObsMat(8,2) = 0;

    Eigen::Matrix<T,3,1> t = tObsMat.colPivHouseholderQr().solve(MVec);

    return AffineTransform<T>(R,t);

}


/*!
 * \brief The AxisRaysSetsAligner class represent a GaussNewtown method to try to axis align two ray sets using a fully parametrized transformation
 *
 * The model aim at solving the epipolar constraint <Rv2 x R'Rv1, t> = 0,
 * where v2 and v1 are the directions of the rays in the second, respectively first frame of reference and R'v + t is the rigid transform from frame 1 to frame 2.
 */
template<typename T>
class AxisRaysSetsAligner : public Optimization::GaussNewtownProblem<T, 3> {

public:

    static constexpr int NParams = 3;

    /*!
     * \brief AxisRaysSetsAligner compute a boresight from a set of raysPairsInfos
     * \param raysPairInfos a reference to the rays pairs (note that AxisRaysSetsAligner keep a reference to it, so its lifetime shall not end before the AxisRaysSetsAligner)
     */
    AxisRaysSetsAligner(std::vector<RayPairInfos<T>> const& raysPairInfos,
                        Optimization::GaussNewtownKernel<T>* kernel = nullptr) :
        Optimization::GaussNewtownProblem<T, 3>(kernel),
        _raysPairInfos(raysPairInfos)
    {

    }

    virtual typename Optimization::GaussNewtownProblem<T, 3>::VectorbType computeResiduals
    (typename Optimization::GaussNewtownProblem<T, 3>::VectorxType const& x) const override {

        int nObs = _raysPairInfos.size();

        typename Optimization::GaussNewtownProblem<T, 3>::VectorbType b;
        b.resize(nObs);

        for (int i = 0; i < nObs; i++) {
            Eigen::Matrix<T,3,1> t = _raysPairInfos[i].t;
            Eigen::Matrix<T,3,1> r1 = _raysPairInfos[i].R1_to_2*StereoVision::Geometry::angleAxisRotate(x, _raysPairInfos[i].v1);
            Eigen::Matrix<T,3,1> r2 = StereoVision::Geometry::angleAxisRotate(x, _raysPairInfos[i].v2);
            b[i] = t.dot(r1.cross(r2));
        }

        return b;
    }

    virtual typename Optimization::GaussNewtownProblem<T, 3>::MatrixAType computeJacobian
    (typename Optimization::GaussNewtownProblem<T, 3>::VectorxType const& x) const override {

        int nObs = _raysPairInfos.size();

        typename Optimization::GaussNewtownProblem<T, 3>::MatrixAType A;
        A.resize(nObs, NParams);

        for (int i = 0; i < nObs; i++) {
            Eigen::Matrix<T,3,1> t = _raysPairInfos[i].t;
            Eigen::Matrix<T,3,1> r1 = _raysPairInfos[i].R1_to_2*StereoVision::Geometry::angleAxisRotate(x, _raysPairInfos[i].v1);
            Eigen::Matrix<T,3,1> r2 = StereoVision::Geometry::angleAxisRotate(x, _raysPairInfos[i].v2);

            std::array<StereoVision::Geometry::Axis, 3> axis =
            {StereoVision::Geometry::Axis::X,
             StereoVision::Geometry::Axis::Y,
             StereoVision::Geometry::Axis::Z};

            for (StereoVision::Geometry::Axis ax : axis) {

                Eigen::Matrix<T,3,1> dr1 = _raysPairInfos[i].R1_to_2*StereoVision::Geometry::diffAngleAxisRotate(x, _raysPairInfos[i].v1, ax);
                Eigen::Matrix<T,3,1> dr2 = StereoVision::Geometry::diffAngleAxisRotate(x, _raysPairInfos[i].v2, ax);

                A(i,static_cast<Eigen::Index>(ax)) = t.dot(dr1.cross(r2) + r1.cross(dr2));
            }

        }

        return A;

    }

protected:
    std::vector<RayPairInfos<T>> const& _raysPairInfos;

};

/*!
 * \brief relaxedAxisAlignRaysSets try to axis align two ray sets using a generic linear transformation
 * \param raysPairInfos the pairs of rays
 * \return the best boresight to align the rays in set 1 and 2.
 *
 * The relaxed linear approximation has 34 degrees of freedom (54 for the rotation matrix and its square coefficient, -20 constraints)
 *
 * The model aim at solving the epipolar constraint <Rv2 x R'Rv1, t> = 0,
 * where v2 and v1 are the directions of the rays in the second, respectively first frame of reference and R'v + t is the rigid transform from frame 1 to frame 2.
 */
template<typename T>
std::optional<Eigen::Matrix<T, 3, 3>> relaxedAxisAlignRaysSets(std::vector<RayPairInfos<T>> const& raysPairInfos) {

    constexpr int nParams = 54;
    constexpr int nConstraints = 21; //20 are linearily independant

    using MatrixAType = Eigen::Matrix<T, Eigen::Dynamic, nParams>;
    using VectorbType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using VectorxType = Eigen::Matrix<T, nParams, 1>;

    if (raysPairInfos.empty()) {
        return std::nullopt;
    }

    int nObs = raysPairInfos.size();
    int nEqs = nObs + nConstraints; //we have 20 constraints
    constexpr int nR = 9;

    Eigen::Matrix<Eigen::Index, 3, 3> rIds;
    rIds << 0, 1, 2, 3, 4, 5, 6, 7, 8; //index of the variables representing the entries in the rotation matrix

    Eigen::Matrix<Eigen::Index, nR, nR> rSquaredIds;

    int c = 0;

    for (int i = 0; i < nR; i++) {
        for (int j = i; j < nR; j++) {
            rSquaredIds(i,j) = nR + c;
            rSquaredIds(j,i) = nR + c;
            c++;
        }
    }

    MatrixAType A;
    A.resize(nEqs, nParams);

    VectorbType b;
    b.setConstant(nEqs, 0);

    std::array<Eigen::Index, 3> nId = {1,2,0};
    std::array<Eigen::Index, 3> pId = {2,0,1};

    int e = 0; //equation line

    //Constaints
    A.template block<nConstraints, nParams>(0,0).setConstant(0);

    // dot products of R
    // norms

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(e,rSquaredIds(rIds(i,j),rIds(i,j))) = 1;
            b[e] = 1;
        }
        e++;
    }

    //perp constraints

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(e,rSquaredIds(rIds(i,j),rIds(nId[i],j))) = 1;
        }
        e++;
    }

    // dot products of Rt
    // norms

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(e,rSquaredIds(rIds(j,i),rIds(j,i))) = 1;
            b[e] = 1;
        }
        e++;
    }

    //perp constraints

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(e,rSquaredIds(rIds(j,i),rIds(j,nId[i]))) = 1;
        }
        e++;
    }

    //cross product constraints of R

    for (int i = 0; i < 3; i++) { //iterate the three vectors pairs
        for (int j = 0; j < 3; j++) {
            A(e,rIds(i,j)) = -1;
            A(e,rSquaredIds(rIds(pId[i],pId[j]), rIds(nId[i],nId[j]))) = 1;
            A(e,rSquaredIds(rIds(pId[i],nId[j]), rIds(nId[i],pId[j]))) = -1;
            e++;
        }
    }

    for (RayPairInfos<T> const& pair : raysPairInfos) {

        A.template block<1,9>(e,0).setConstant(0);

        //To simplify the computation we represent the computation of the cross product as
        // (R[0,0] vLHS_0 + ... + R[2,2] vLHS_9) x R'(R[0,0] vRHS_0 + ... + R[2,2] vRHS_9)
        Eigen::Matrix<T, 3, 9> coeffsRHS; //represent the coefficient for the 9 rotation matrix entry for the 3 entries of the RHS of the cross product
        coeffsRHS.setConstant(0);
        Eigen::Matrix<T, 3, 9> coeffsLHS; //represent the coefficient for the 9 rotation matrix entry for the 3 entries of the LHS of the cross product
        coeffsLHS.setConstant(0);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                coeffsRHS(i,rIds(i,j)) = pair.v1[j];
                coeffsLHS(i,rIds(i,j)) = pair.v2[j];
            }
        }

        Eigen::Matrix<T, 3, 9> coeffsRotatedRHS = pair.R1_to_2*coeffsRHS;

        for (int i1 = 0; i1 < 3; i1++) {
            for (int j1 = 0; j1 < 3; j1++) {

                for (int i2 = 0; i2 < 3; i2++) {
                    for (int j2 = 0; j2 < 3; j2++) {

                        Eigen::Index id1 = rIds(i1,j1);
                        Eigen::Index id2 = rIds(i2,j2);

                        Eigen::Index pId = rSquaredIds(id1, id2);

                        if (id1 == id2) {

                            A(e, pId) = pair.t.dot(coeffsLHS.col(id1).cross(coeffsRotatedRHS.col(id2)));

                        } else {

                            A(e, pId) = pair.t.dot(coeffsLHS.col(id1).cross(coeffsRotatedRHS.col(id2)));
                            A(e, pId) += pair.t.dot(coeffsLHS.col(id2).cross(coeffsRotatedRHS.col(id1)));
                        }

                    }
                }
            }
        }

        e++;
    }

    if (!A.array().isFinite().all()) {
        return std::nullopt;
    }

    VectorxType sol = A.colPivHouseholderQr().solve(b);

    Eigen::Matrix<T,3,3> ret;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ret(i,j) = sol[rIds(i,j)];
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> unscaledRsvd(ret);

    T invScale = unscaledRsvd.singularValues().mean();
    T scale = 1;

    if (std::isfinite(1/invScale)) {
        scale /= invScale;
    }

    return scale*ret;

}

/*!
 * \brief robustRelaxedAxisAlignRaysSets try to axis align two ray sets using a generic linear transformation and huber loss to limit the impact of outliers
 * \param raysPairInfos the pairs of rays
 * \return the best boresight to align the rays in set 1 and 2.
 *
 * this function aims at solving the same problem as relaxedAxisAlignRaysSets, but based on the Huber loss instead of the quadratic loss.
 */
template<typename T>
std::optional<Eigen::Matrix<T, 3, 3>> robustRelaxedAxisAlignRaysSets(std::vector<RayPairInfos<T>> const& raysPairInfos,
                                                                     int maxIter = 100,
                                                                     T tol = 1e-4) {

    constexpr int nParams = 54;
    constexpr int nConstraints = 21; //20 are linearily independant

    using MatrixAType = Eigen::Matrix<T, Eigen::Dynamic, nParams>;
    using VectorbType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using VectorxType = Eigen::Matrix<T, nParams, 1>;

    if (raysPairInfos.empty()) {
        return std::nullopt;
    }

    int nObs = raysPairInfos.size();
    int nEqs = nObs + nConstraints; //we have 20 constraints
    constexpr int nR = 9;

    Eigen::Matrix<Eigen::Index, 3, 3> rIds;
    rIds << 0, 1, 2, 3, 4, 5, 6, 7, 8; //index of the variables representing the entries in the rotation matrix

    Eigen::Matrix<Eigen::Index, nR, nR> rSquaredIds;

    int c = 0;

    for (int i = 0; i < nR; i++) {
        for (int j = i; j < nR; j++) {
            rSquaredIds(i,j) = nR + c;
            rSquaredIds(j,i) = nR + c;
            c++;
        }
    }

    MatrixAType A;
    A.resize(nEqs, nParams);

    VectorbType b;
    b.setConstant(nEqs, 0);

    std::array<Eigen::Index, 3> nId = {1,2,0};
    std::array<Eigen::Index, 3> pId = {2,0,1};

    int e = 0; //equation line

    //Constaints
    A.template block<nConstraints, nParams>(0,0).setConstant(0);

    // dot products of R
    // norms

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(e,rSquaredIds(rIds(i,j),rIds(i,j))) = 1;
            b[e] = 1;
        }
        e++;
    }

    //perp constraints

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(e,rSquaredIds(rIds(i,j),rIds(nId[i],j))) = 1;
        }
        e++;
    }

    // dot products of Rt
    // norms

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(e,rSquaredIds(rIds(j,i),rIds(j,i))) = 1;
            b[e] = 1;
        }
        e++;
    }

    //perp constraints

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(e,rSquaredIds(rIds(j,i),rIds(j,nId[i]))) = 1;
        }
        e++;
    }

    //cross product constraints of R

    for (int i = 0; i < 3; i++) { //iterate the three vectors pairs
        for (int j = 0; j < 3; j++) {
            A(e,rIds(i,j)) = -1;
            A(e,rSquaredIds(rIds(pId[i],pId[j]), rIds(nId[i],nId[j]))) = 1;
            A(e,rSquaredIds(rIds(pId[i],nId[j]), rIds(nId[i],pId[j]))) = -1;
            e++;
        }
    }

    for (RayPairInfos<T> const& pair : raysPairInfos) {

        A.template block<1,9>(e,0).setConstant(0);

        //To simplify the computation we represent the computation of the cross product as
        // (R[0,0] vLHS_0 + ... + R[2,2] vLHS_9) x R'(R[0,0] vRHS_0 + ... + R[2,2] vRHS_9)
        Eigen::Matrix<T, 3, 9> coeffsRHS; //represent the coefficient for the 9 rotation matrix entry for the 3 entries of the RHS of the cross product
        coeffsRHS.setConstant(0);
        Eigen::Matrix<T, 3, 9> coeffsLHS; //represent the coefficient for the 9 rotation matrix entry for the 3 entries of the LHS of the cross product
        coeffsLHS.setConstant(0);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                coeffsRHS(i,rIds(i,j)) = pair.v1[j];
                coeffsLHS(i,rIds(i,j)) = pair.v2[j];
            }
        }

        Eigen::Matrix<T, 3, 9> coeffsRotatedRHS = pair.R1_to_2*coeffsRHS;

        for (int i1 = 0; i1 < 3; i1++) {
            for (int j1 = 0; j1 < 3; j1++) {

                for (int i2 = 0; i2 < 3; i2++) {
                    for (int j2 = 0; j2 < 3; j2++) {

                        Eigen::Index id1 = rIds(i1,j1);
                        Eigen::Index id2 = rIds(i2,j2);

                        Eigen::Index pId = rSquaredIds(id1, id2);

                        if (id1 == id2) {

                            A(e, pId) = pair.t.dot(coeffsLHS.col(id1).cross(coeffsRotatedRHS.col(id2)));

                        } else {

                            A(e, pId) = pair.t.dot(coeffsLHS.col(id1).cross(coeffsRotatedRHS.col(id2)));
                            A(e, pId) += pair.t.dot(coeffsLHS.col(id2).cross(coeffsRotatedRHS.col(id1)));
                        }

                    }
                }
            }
        }

        e++;
    }

    if (!A.array().isFinite().all()) {
        return std::nullopt;
    }

    VectorxType sol = A.colPivHouseholderQr().solve(b);

    for (int i = 0; i < maxIter; i++) {
        VectorbType currentVal = A*sol;

        MatrixAType Aupdtd = A;

        //apply the Huber loss to the observations, but not the constraints
        for (int j = nConstraints; j < nEqs; j++) {
            Aupdtd.row(j) *= Optimization::diffSqrtHuberLoss(currentVal[j]);
            currentVal[j] = Optimization::sqrtHuberLoss(currentVal[j]);
        }

        VectorxType dx = Aupdtd.colPivHouseholderQr().solve(b - currentVal);

        sol += dx;

        T delta = dx.norm()/nParams;

        if (delta < tol) {
            break;
        }
    }

    Eigen::Matrix<T,3,3> ret;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ret(i,j) = sol[rIds(i,j)];
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> unscaledRsvd(ret);

    T invScale = unscaledRsvd.singularValues().mean();
    T scale = 1;

    if (std::isfinite(1/invScale)) {
        scale /= invScale;
    }

    return scale*ret;

}


} // namespace Geometry
} // namespace StereoVision

#endif // GENERICRAYSALIGNEMENT_H
