#ifndef BFGS_H
#define BFGS_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2026  Paragon<french.paragon@gmail.com>

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

#include "../utils/iterative_numerical_algorithm_output.h"

#include <Eigen/Core>
#include <Eigen/QR>

namespace StereoVision {
namespace Optimization {

/*!
 * \brief The lBFGSMinimizationProblem class represent a minimization problem to solve using the lBFGS algorithm
 *
 * This class automatize the implementation of solving minimization problems using lBFGS.
 *
 * It also make it easy to access the hystory of parameters values and optimization function value.
 *
 * The function (and gradient computation) are provided by the template parameter Func;
 */
template <typename T, typename Func, int M, int nParams>
class lBFGSMinimizationProblem : public Func {
public:
    template<typename ... FuncConstrT>
    lBFGSMinimizationProblem(FuncConstrT const& ... funcsArgs) :
        Func(funcsArgs...),
        _nIterations(0),
        _convergenceType(ConvergenceType::Unknown)
    {

    }

    typedef Eigen::Matrix<T, nParams, 1> VectorxType; //parameters
    using CallBackFunc = std::function<void(lBFGSMinimizationProblem<T,Func,M,nParams> const&)>;

    inline T objective(VectorxType const& parameters) const {
        return Func::objective(parameters);
    }
    inline VectorxType gradient(VectorxType const& parameters) const {
        return Func::gradient(parameters);
    }
    inline VectorxType initialDiagonal(VectorxType const& parameters) const {
        return Func::initialDiagonal(parameters);
    }

    inline bool isOptimized() const {
        return _nIterations > 0;
    }

    inline int nIterations() const {
        return _nIterations;
    }

    inline ConvergenceType convergenceType() const {
        return _convergenceType;
    }

    inline VectorxType const& solution() const {
        return _solution;
    }

    /*!
     * \brief objectiveHistory return the history of the objective function value during optimization
     * \return the history of objective function values, first entry is initial guess, last entry is the one corresponding to solution()
     */
    inline std::vector<T> const& objectiveHistory() const {
        return _objective_history;
    }
    inline T currentObjectiveValue() const {
        return _objective_history.back();
    }
    inline VectorxType currentGradientValue() const {
        return _gradient;
    }
    inline VectorxType previousSolutionDelta() const {
        return _solutionDelta_history.back();
    }

    /*!
     * \brief solutionHistory return the history of solutions during optimization
     * \return the history of solutions, first entry is initial guess, last entry is the same as solution()
     */
    inline std::vector<VectorxType> const& solutionHistory() const {
        return _solution_history;
    }


    ConvergenceType run(int maxIter, T tol,
                        VectorxType const& initialGuess = VectorxType::Zero(),
                        CallBackFunc iterationCallback = CallBackFunc()) {

        _convergenceType = ConvergenceType::MaxIterReached;

        _objective_history.reserve(maxIter+1);
        _solution_history.reserve(maxIter+1);
        _solutionDelta_history.reserve(maxIter+1);
        _solutionDelta_history.reserve(maxIter+1);

        _solution = initialGuess;
        _solution_history.push_back(_solution);
        T currentObjective = objective(_solution);
        _objective_history.push_back(currentObjective);

        VectorxType& g = _gradient;
        g = gradient(_solution);

        std::vector<VectorxType>& s = _solutionDelta_history;
        std::vector<VectorxType>& y = _gradientDelta_history;

        T previousSearchScale = 0.5; //in case line search requires a scale, twice this variable will be selected. This help keeping track of expected search scales.

        for (_nIterations = 0; _nIterations < maxIter; _nIterations++) {

            if (g.rows() != _solution.rows()) {
                _convergenceType = ConvergenceType::Failed;
                break;
            }

            //find direction

            VectorxType q = g;
            VectorxType z;

            if constexpr (nParams != 1) {

                int estIdx = std::max(0,_nIterations-M);
                int nAlphas = std::max(1,_nIterations-estIdx);

                std::vector<T> alphas(nAlphas);

                for (int i = _nIterations-1; i >= estIdx; i--) {
                    T alpha = s[i].dot(q)/(s[i].dot(y[i]));
                    alphas[i-estIdx] = alpha;
                    q = q - alpha*y[i];
                }

                T gamma = 1;
                if (_nIterations > 0) {
                    int gIdx = _nIterations - std::min(_nIterations,M);
                    gamma = s[gIdx].dot(y[gIdx])/(y[gIdx].dot(y[gIdx]));
                }

                VectorxType initialDiag = initialDiagonal(_solution);
                z = gamma*q.array()*initialDiag.array();

                for(int i = estIdx; i < _nIterations; i++) {
                    T beta = y[i].dot(z)/(s[i].dot(y[i]));
                    z += s[i]*(alphas[i-estIdx] - beta);
                }

                z = -z; //orient for minimizatio
            } else {
                z = -q; //if only a single parameter optimize the direction finding
            }

            T zMax = z.template lpNorm<Eigen::Infinity>() ;

            //perform line search

            constexpr T dx = 1e-3;

            VectorxType zIncr = z/zMax; //for numerical stability use this as direction
            T zIncrNorm = zIncr.norm();
            if (!std::isfinite(zIncrNorm)) {
                zIncr = z*0;
            }

            T scale = 2*previousSearchScale;

            T objectiveCand = objective(_solution + scale*zIncr);

            if (objectiveCand > currentObjective or !std::isfinite(objectiveCand)) {

                scale = scale/2;
                T bestScale = 0;
                T currentBestOptimized = currentObjective;
                T incr = scale/2;
                constexpr int maxIter = 20; //change scale by at most 20 orders of magnitude (in binary)

                for (int i = 0; i < maxIter; i++) {
                    VectorxType proposal = _solution + scale*zIncr;
                    T objectiveCand = objective(proposal);

                    if (objectiveCand < currentBestOptimized) {
                        T previousBest = bestScale;
                        bestScale = scale;
                        currentBestOptimized = objectiveCand;

                        break; //better keep a good solution, than spending a lot of time trying to fish for a better one.
                    } else {
                        if (scale > bestScale) {
                            scale -= incr;
                        } else {
                            scale += incr;
                        }
                    }

                    incr /= 2;
                }

                previousSearchScale = bestScale;
                scale = bestScale;

            } else {
                previousSearchScale = scale;
            }

            //update the data
            VectorxType solutionDelta = scale*zIncr;
            _solution += solutionDelta; //switch from increase to decrease direction;

            currentObjective = objective(_solution);
            _objective_history.push_back(currentObjective);
            _solution_history.push_back(_solution);

            _solutionDelta_history.push_back(solutionDelta);

            VectorxType targetGradient = gradient(_solution);
            _gradientDelta_history.push_back(targetGradient-g);
            g = targetGradient;

            if (iterationCallback) {
                iterationCallback(*this);
            }

            if (g.norm()/_solution.size() < tol or !std::isfinite(1/scale)) {
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
    VectorxType _gradient;
    std::vector<T> _objective_history;
    std::vector<VectorxType> _solution_history;

    std::vector<VectorxType> _solutionDelta_history;
    std::vector<VectorxType> _gradientDelta_history;
};

} // namespace Optimization
} // namespace StereoVision

#endif // BFGS_H
