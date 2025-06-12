#ifndef ITERATIVE_NUMERICAL_ALGORITHM_OUTPUT_H
#define ITERATIVE_NUMERICAL_ALGORITHM_OUTPUT_H

#include <string>

namespace StereoVision {

enum ConvergenceType {
    Converged = 0,
    MaxIterReached = 1,
    Failed = 2,
    Unknown = 3
};

template<typename T>
class IterativeNumericalAlgorithmOutput {

public:

    IterativeNumericalAlgorithmOutput(T const& val, ConvergenceType convType = Unknown) :
        _val(val),
        _convergence(convType) {

    }

    inline T& value() {
        return _val;
    }

    inline T const& value() const{
        return _val;
    }

    inline ConvergenceType convergence() const {
        return _convergence;
    }

    inline std::string convergenceStr() const {
        switch(_convergence) {
        case Converged :
            return "Converged";
        case MaxIterReached :
            return "Max Iteration Reached";
        case Failed :
            return "Failed to Converge";
        case Unknown :
            return "Status Unknown";
        }
    }

private:
    T _val;
    ConvergenceType _convergence;
};

} //namespace StereoVision

#endif // ITERATIVE_NUMERICAL_ALGORITHM_OUTPUT_H
