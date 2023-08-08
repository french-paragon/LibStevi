#ifndef PRINCIPALCOMPONENTSANALYSIS_H
#define PRINCIPALCOMPONENTSANALYSIS_H

#include <Eigen/Core>
#include <Eigen/SVD>

namespace StereoVision {
namespace Optimization {

template<typename T, int nComps, int vecDims, int nVecs>
Eigen::Matrix<T, nComps, vecDims> principalComponents(Eigen::Matrix<T, vecDims, nVecs> const& data, int nCompDynamic = 1) {

    // also might consider implementing Rayleight-Ritz method.

    using DType = Eigen::Matrix<T, vecDims, nVecs>;
    using QDataT = Eigen::Matrix<T, vecDims, vecDims>;

    Eigen::Matrix<T, vecDims, 1> mean = data.rowwise().mean();
    Eigen::Matrix<T, vecDims, nVecs> detrended = data.colwise() - mean;

    QDataT dataCovariance = detrended*detrended.transpose();

    auto svd = dataCovariance.bdcSvd(Eigen::ComputeFullV);

    QDataT dec = svd.matrixV();

    Eigen::Matrix<T, nComps, vecDims> ret;

    int nRows = nComps;
    int nCols = vecDims;

    if (nComps == Eigen::Dynamic) {
        nRows = nCompDynamic;
    }

    if (vecDims == Eigen::Dynamic) {
        nCols = data.rows();
    }

    ret.resize(nRows, nCols);

    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            ret(i,j) = dec(i,j);
        }
    }

    return ret;

}

} // namespace Optimization
} // namespace StereoVision

#endif // PRINCIPALCOMPONENTSANALYSIS_H
