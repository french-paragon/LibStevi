/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024  Paragon<french.paragon@gmail.com>

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

#include "geometry/alignement.h"
#include "geometry/rotations.h"
#include "geometry/geometricexception.h"
#include "geometry/genericraysalignement.h"

#include <random>
#include <iostream>
#include <Eigen/SVD>
#include <Eigen/Geometry>

using namespace StereoVision;
using namespace StereoVision::Geometry;

struct GenericRayAlignementProblem {
    Eigen::Vector3d boresight;
    RigidBodyTransform<double> frame1toframe2;
    std::vector<RayInfos<double>> frame1Rays;
    std::vector<RayInfos<double>> frame2Rays;
};

/*!
 * \brief generateRayAlignmentProblemPushBroom simulate the rays matches for two push broom lines
 * \param nRays the number of rays
 * \param frame1toframe2 the transform from line1 to line2.
 * \param pathRadius the distance (both positive and negative) that the line is going to have.
 * \param rayPosStd the standard deviation of a point in the line.
 * \param rayOrientStd the standard deviation of the position of the camera.
 * \param seed the random seed to use.
 * \param boresight the boresight of the camera.
 * \return the generated GenericRayAlignementProblem
 */
GenericRayAlignementProblem generateRayAlignmentProblemPushBroom(int nRays,
                                                                 RigidBodyTransform<double> const& frame1toframe2,
                                                                 double pathRadius = 1,
                                                                 double rayPosStd = 0.2,
                                                                 double rayOrientStd = 0.01,
                                                                 int seed = -1,
                                                                 Eigen::Vector3d const& boresight = Eigen::Vector3d::Zero()) {

    using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    std::random_device rd;
    int re_seed = seed;
    if (seed < 0) {
        re_seed = rd();
    }
    std::default_random_engine gen(re_seed);

    std::normal_distribution<double> normal(0,1);
    std::uniform_real_distribution<double> angleDist(-M_PI_4, M_PI_4);

    std::vector<double> rayPosProp(nRays);

    for (int i = 0; i < nRays; i++) {
        rayPosProp[i] = -pathRadius + 2*pathRadius*(i+1)/nRays;
    }

    GenericRayAlignementProblem ret;
    ret.boresight = boresight;
    ret.frame1toframe2 = frame1toframe2;

    ret.frame1Rays.reserve(nRays);
    ret.frame2Rays.reserve(nRays);

    Vec noiseX;
    Vec noiseY;
    Vec noiseZ;

    Vec noise2X;
    Vec noise2Y;
    Vec noise2Z;

    Mat transform;

    noiseX.resize(nRays);
    noiseY.resize(nRays);
    noiseZ.resize(nRays);

    noise2X.resize(nRays);
    noise2Y.resize(nRays);
    noise2Z.resize(nRays);

    transform.resize(nRays, nRays);

    for (int i = 0; i < nRays; i++) {
        noiseX[i] = normal(gen);
        noiseY[i] = normal(gen);
        noiseZ[i] = normal(gen);

        noise2X[i] = normal(gen);
        noise2Y[i] = normal(gen);
        noise2Z[i] = normal(gen);

        for (int j = 0; j < nRays; j++) {
            transform(i,j) = (i==j) ? rayPosStd : 0;
        }
    }

    noiseX = transform*noiseX;
    noiseY = transform*noiseY;
    noiseZ = transform*noiseZ;

    noise2X = transform*noise2X;
    noise2Y = transform*noise2Y;
    noise2Z = transform*noise2Z;

    for (int i = 0; i < nRays; i++) {

        Eigen::Vector3d f1pos(rayPosProp[i]*pathRadius + noiseX[i], noiseY[i], noiseZ[i]);
        double angle = angleDist(gen);
        Eigen::Vector3d f1ray(0, sin(angle), cos(angle));

        Eigen::Vector3d randomRot(rayOrientStd*normal(gen), rayOrientStd*normal(gen), rayOrientStd*normal(gen));

        f1ray = StereoVision::Geometry::angleAxisRotate(randomRot, f1ray);

        ret.frame1Rays.emplace_back(f1pos, f1ray);

        Eigen::Vector3d f2pos(rayPosProp[i]*pathRadius + noise2X[i], noise2Y[i], noise2Z[i]);

        angle = angleDist(gen);
        double dist12 = (ret.frame1toframe2*f1pos - f2pos).norm();
        double distProj = std::abs(normal(gen))*dist12;

        Eigen::Vector3d f2posinf1 = ret.frame1toframe2.inverse()*f2pos;

        Eigen::Vector3d ray = f1pos + distProj*f1ray - f2posinf1;
        ray.normalize();

        Eigen::Vector3d f2ray = angleAxisRotate(ret.frame1toframe2.r, ray);
        f2ray.normalize();

        ret.frame2Rays.emplace_back(f2pos, f2ray);

    }

    return ret;

}

int main(int argc, char** argv) {

    std::cout << "Testing convergence for generic rays intersection function" << "\n" << std::endl;

    constexpr int seed1 = 69;
    constexpr int seed2 = 42;
    constexpr int nRays = 300;
    constexpr int maxIter = 1000;
    constexpr double tolerance = 1e-5;
    float pathRadius = 10;
    float rayPosStd = 3;
    float rayOrientStd = 0.6;

    RigidBodyTransform<double> frame1toframe2(Eigen::Vector3d(0.042, -0.021, M_PI + 0.012), Eigen::Vector3d(-0.24, 5.42, 0.32));

    GenericRayAlignementProblem problem = generateRayAlignmentProblemPushBroom(nRays, frame1toframe2, pathRadius, rayPosStd, rayOrientStd, seed1);

    int nObs = problem.frame1Rays.size();

    struct noiseDefinition {
        double posStd;
        double rotStd;
    };

    std::vector<noiseDefinition> noise2check = {noiseDefinition{0,0}, noiseDefinition{0.1,0.01}, noiseDefinition{0.5,0.05}, noiseDefinition{1,0.1}, noiseDefinition{5,0.5}};

    for (noiseDefinition & noise : noise2check) {

        std::random_device rd;
        int re_seed = seed2;
        if (seed2 < 0) {
            re_seed = rd();
        }
        std::default_random_engine gen(re_seed);

        std::normal_distribution<double> normal(0,1);

        Eigen::Vector3d randPosDelta(normal(gen), normal(gen), normal(gen));
        Eigen::Vector3d randRotDelta(normal(gen), normal(gen), normal(gen));
        randPosDelta *= noise.posStd;
        randRotDelta *= noise.rotStd;

        RigidBodyTransform<double> initial = frame1toframe2;
        initial.r += randRotDelta;
        initial.t += randPosDelta;

        std::cout << "Sigma_t = " << noise.posStd << " Sigma_r = " << noise.rotStd << " ";

        auto solution = alignRaysSets(problem.frame1Rays, problem.frame2Rays, initial, maxIter, tolerance);

        if (solution.convergence() == ConvergenceType::Failed) {
            std::cout << "Failed to reach any solution" << std::endl;
        }

        std::string convergence_status = solution.convergenceStr();

        double rotDelta = (solution.value().r - frame1toframe2.r).norm();
        double posDelta = (solution.value().t - frame1toframe2.t).norm();

        Eigen::VectorXd res;
        res.resize(nObs);

        for (int i = 0; i < nObs; i++) {

            Eigen::Vector3d const& pi = problem.frame1Rays[i].localSystemRayOrigin;
            Eigen::Vector3d const& vi = problem.frame1Rays[i].localSystemRayDirection;

            Eigen::Vector3d const& pj = problem.frame2Rays[i].localSystemRayOrigin;
            Eigen::Vector3d const& vj = problem.frame2Rays[i].localSystemRayDirection;

            Eigen::Vector3d Rpi = angleAxisRotate(solution.value().r, pi);
            Eigen::Vector3d Rvi = angleAxisRotate(solution.value().r, vi);

            res[i] = vj.dot((Rpi + solution.value().t - pj).cross(Rvi));
        }

        std::cout << "RotDelta = " << rotDelta << " PosDelta = " << posDelta << " Convergence Status: " << convergence_status << " MSE = " << res.squaredNorm()/res.size() << std::endl;

    }

    std::vector<noiseDefinition> measurementNoise2check = {noiseDefinition{0,0}, noiseDefinition{0.01,0.001}, noiseDefinition{0.1,0.01}};


    for (noiseDefinition & noise : measurementNoise2check) {

        std::cout << "\nRelaxed solution: " << "Sigma_t = " << noise.posStd << " Sigma_r = " << noise.rotStd << " ";

        auto rays1 = problem.frame1Rays;
        auto rays2 = problem.frame2Rays;

        std::random_device rd;
        int re_seed = seed2;
        if (seed2 < 0) {
            re_seed = rd();
        }
        std::default_random_engine gen(re_seed);

        std::normal_distribution<double> normal(0,1);

        for (auto & ray : rays1) {
            ray.localSystemRayOrigin += Eigen::Vector3d(noise.posStd*normal(gen), noise.posStd*normal(gen), noise.posStd*normal(gen));

            Eigen::Vector3d smallRot(noise.rotStd*normal(gen), noise.rotStd*normal(gen), noise.rotStd*normal(gen));
            ray.localSystemRayDirection = angleAxisRotate(smallRot, ray.localSystemRayDirection);
        }

        for (auto & ray : rays2) {

            ray.localSystemRayOrigin += Eigen::Vector3d(noise.posStd*normal(gen), noise.posStd*normal(gen), noise.posStd*normal(gen));

            Eigen::Vector3d smallRot(noise.rotStd*normal(gen), noise.rotStd*normal(gen), noise.rotStd*normal(gen));
            ray.localSystemRayDirection = angleAxisRotate(smallRot, ray.localSystemRayDirection);

        }

        auto relaxedSolution = relaxedAlignRaysSets(rays1, rays2);
        Eigen::Matrix3d Rtrue = rodriguezFormula(frame1toframe2.r);

        if (!relaxedSolution.has_value()) {
            std::cout << "Failed to get a relaxed solution" << std::endl;
        }

        Eigen::VectorXd res;
        res.resize(nObs);

        for (int i = 0; i < nObs; i++) {

            Eigen::Vector3d const& pi = problem.frame1Rays[i].localSystemRayOrigin;
            Eigen::Vector3d const& vi = problem.frame1Rays[i].localSystemRayDirection;

            Eigen::Vector3d const& pj = problem.frame2Rays[i].localSystemRayOrigin;
            Eigen::Vector3d const& vj = problem.frame2Rays[i].localSystemRayDirection;

            Eigen::Vector3d Rpi = relaxedSolution.value().R*pi;
            Eigen::Vector3d Rvi = relaxedSolution.value().R*vi;

            res[i] = vj.dot((Rpi + relaxedSolution.value().t - pj).cross(Rvi));
        }

        double Rdet = relaxedSolution.value().R.determinant();
        Eigen::Matrix3d Rcheck = relaxedSolution.value().R.transpose() * relaxedSolution.value().R;

        Eigen::Matrix3d Rdelta = relaxedSolution.value().R.transpose() * Rtrue;
        double rotDelta = inverseRodriguezFormula(Rdelta).norm();
        double posDelta = (relaxedSolution.value().t - frame1toframe2.t).norm();

        std::cout << "RotDeltaRelaxed = " << rotDelta << " PosDeltaRelaxed = " << posDelta <<
                     " RotDet = " << Rdet <<
                     " RCheck: [" << Rcheck(0,0) << " " << Rcheck(1,0) << " " << Rcheck(2,0) << " " <<
                     Rcheck(0,1) << " " << Rcheck(1,1) << " " << Rcheck(2,1) << " " <<
                     Rcheck(0,2) << " " << Rcheck(1,2) << " " << Rcheck(2,2) << "] " <<
                     "MSERelaxed = " << res.squaredNorm()/res.size() << std::endl;

    }

    std::cout << std::endl;

    return 0;
}
