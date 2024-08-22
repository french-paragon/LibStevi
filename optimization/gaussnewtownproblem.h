#ifndef GAUSSNEWTOWNPROBLEM_H
#define GAUSSNEWTOWNPROBLEM_H

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

#include <optional>

#include "../utils/iterative_numerical_algorithm_output.h"
#include "./huber_kernel.h"

#include <Eigen/Core>
#include <Eigen/QR>

namespace StereoVision {
namespace Optimization {

/*!
 * \brief The GaussNewtownKernel class implements a kernel to apply to a GaussNewtownProblem
 *
 * The kernel can for example be used to implement the Huber loss instead of the standard L2 loss for the residuals.
 */
template <typename T>
class GaussNewtownKernel {

public:
    virtual T kernel(T val) const = 0;
    virtual T diffKernel(T val) const = 0;
};


template <typename T>
class GaussNewtownHuberKernel : public GaussNewtownKernel<T> {

public:
    GaussNewtownHuberKernel(T threshold) :
        _threshold(threshold)
    {

    }

    virtual T kernel(T val) const override {
        return sqrtHuberLoss(val, _threshold);
    }

    virtual T diffKernel(T val) const override {
        return diffSqrtHuberLoss(val, _threshold);
    }

protected:

    T _threshold;
};

/*!
 * \brief The GaussNewtownProblem class represent a Gauss Newtown problem
 *
 * This class automatize the implementation of gauss newtown problems.
 *
 * It also make it easy to access the hystory of parameters values
 *
 * The algorithm solve a series of functions of the form f(x) = 0
 * in the least square sense. If the problems is partially or
 * totally underdetermined, a solution is still returned.
 *
 * it is as easy as re-implementing the functions computeResiduals and computeJacobian;
 */
template <typename T, int nParams>
class GaussNewtownProblem {
    static_assert (nParams > 0, "Number of parameters has to be greather than 0.");

public:
    /*!
     * \brief GaussNewtownProblem constructor
     * \param kernel the kernel to use, or nullptr for standard L2. The problem takes ownership of the kernel
     */
    GaussNewtownProblem(GaussNewtownKernel<T>* kernel = nullptr) :
        _nIterations(0),
        _convergenceType(ConvergenceType::Unknown),
        _kernel(kernel)
    {

    }

    ~GaussNewtownProblem() {
        if (_kernel != nullptr) {
            delete _kernel;
        }
    }

    typedef Eigen::Matrix<T, Eigen::Dynamic, nParams> MatrixAType; //derivative
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorbType; //solution
    typedef Eigen::Matrix<T, nParams, 1> VectorxType; //parameters

    virtual VectorbType computeResiduals(VectorxType const& parameters) const = 0;
    virtual MatrixAType computeJacobian(VectorxType const& parameters) const = 0;

    inline bool isOptimized() const {
        return _nIterations > 0;
    }

    inline int nIterations() const {
        return _nIterations;
    }

    inline ConvergenceType convergenceType() const {
        return _convergenceType;
    }

    inline VectorxType solution() const {
        return _solution;
    }

    inline VectorbType residuals() const {
        return _residuals;
    }

    /*!
     * \brief solutionHistory return the history of solutions during optimization
     * \return the history of solutions, first entry is initial guess, last entry is the same as solution()
     */
    inline std::vector<VectorxType> solutionHistory() const {
        return _solution_history;
    }

    ConvergenceType run(int maxIter, T tol, VectorxType const& initialGuess = VectorxType::Zero()) {

        _convergenceType = ConvergenceType::MaxIterReached;

        _solution_history.reserve(maxIter+1);

        _solution = initialGuess;
        _solution_history.push_back(_solution);

        for (_nIterations = 0; _nIterations < maxIter; _nIterations++) {

            _residuals = computeResiduals(_solution);
            MatrixAType A = computeJacobian(_solution);

            if (A.rows() != _residuals.rows()) {
                _convergenceType = ConvergenceType::Failed;
                break;
            }

            if (_kernel != nullptr) {
                for (int i = 0; i < A.rows(); i++) {
                    for (int j = 0; j < nParams; j++) {
                        A(i,j) = _kernel->diffKernel(_residuals[i])*A(i,j);
                    }
                }
                for (int i = 0; i < A.rows(); i++) {
                        _residuals[i] = _kernel->kernel(_residuals[i]);
                }
            }

            //compute the increment for the parameters
            VectorxType dx = A.colPivHouseholderQr().solve(-_residuals); //b = 0

            _solution += dx;
            _solution_history.push_back(_solution);

            T delta = dx.norm()/nParams;

            if (delta < tol) {
                _convergenceType = ConvergenceType::Converged;
                _nIterations++;
                break;
            }

        }

        return _convergenceType;

    }

protected:

    int _nIterations;
    ConvergenceType _convergenceType;

    VectorxType _solution;
    VectorbType _residuals;

    GaussNewtownKernel<T>* _kernel;

    std::vector<VectorxType> _solution_history;

};

}
}

#endif // GAUSSNEWTOWNPROBLEM_H
