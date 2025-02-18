#ifndef INPAINTING_H
#define INPAINTING_H

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

#include <array>

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include "../geometry/genericbinarypartitioningtree.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <Eigen/QR>

namespace StereoVision {
namespace ImageProcessing {

/*!
 * \brief nearestInPaintingMonochannel return an inpainted image where the value of pixels is equal to the nearest not inpainted pixel value.
 * \param img the image
 * \param area2Fill a binary mask, pixels with value true will be inpainted
 * \return the inpainted image, with the same shape and strides as the input image
 */
template <typename T, int nDim>
Multidim::Array<T, nDim> nearestInPaintingMonochannel(Multidim::Array<T, nDim> const& img,
                                                      Multidim::Array<bool, nDim> const& area2Fill) {

    if (img.shape() != area2Fill.shape()) {
        return Multidim::Array<T, nDim>();
    }

    std::vector<std::array<int,nDim>> nonInpaintedPoints;
    nonInpaintedPoints.reserve(area2Fill.flatLenght());

    Multidim::IndexConverter<nDim> indexConverter(area2Fill.shape());

    int nIdxs = indexConverter.numberOfPossibleIndices();

    for (int i = 0; i < nIdxs ; i++) {
        auto idx = indexConverter.getIndexFromPseudoFlatId(i);

        bool notToInpaint = !area2Fill.valueUnchecked(idx);

        if (notToInpaint) {
            nonInpaintedPoints.push_back(idx);
        }
    }

    if (nonInpaintedPoints.empty()) {
        return Multidim::Array<T, nDim>(); //impossible to inpaint when no fixed points are present
    }

    using PartitionTree = Geometry::GenericBSP<std::array<int,nDim>, nDim, Geometry::BSPObjectWrapper<std::array<int,nDim>, int>>;

    PartitionTree tree(std::move(nonInpaintedPoints));

    Multidim::Array<T, nDim> ret(img.shape(), img.strides());

    for (int i = 0; i < nIdxs ; i++) {
        std::array<int,nDim> idx = indexConverter.getIndexFromPseudoFlatId(i);

        std::array<int,nDim> nearest = tree.closest(idx);

        ret.atUnchecked(idx) = img.valueUnchecked(nearest);
    }

    return ret;

}

/*!
 * \brief nearestInPaintingBatched perform neared pixel inpainting in a batched manner
 * \param img the image to inpaint
 * \param area2Fill the pixels to inpaint (as a mask without the batched dimensions).
 * \param batchDims the axis of the image that should be considered batched. They should all be different or the behavior is undefined.
 * \return the inpainted image, with the same shape and strides as the input image.
 */
template <typename T, int nDim, int nBatchDims>
Multidim::Array<T, nDim> nearestInPaintingBatched(Multidim::Array<T, nDim> const& img,
                                                      Multidim::Array<bool, nDim-nBatchDims> const& area2Fill,
                                                       std::array<int, nBatchDims> batchDims) {

    static_assert (nDim > nBatchDims, "Not all dimensions can be channels");

    constexpr int mDim = nDim-nBatchDims;

    std::array<int, nBatchDims> batchShape;

    for (int i = 0; i < nBatchDims; i++) {
        batchShape[i] = img.shape()[batchDims[i]];
    }

    Multidim::IndexConverter<mDim> idxConverter(area2Fill.shape());
    Multidim::IndexConverter<nBatchDims> batchIdxConverter(batchShape);
    Multidim::ExcludedDimsSaticIndexConverter<nDim,nBatchDims> axisIdxCompressors(batchDims);

    for (int i = 0; i < mDim; i++) {
        int maskIdx = i;
        int imgAxisIdx = axisIdxCompressors.getCorrespondingUncompressedAxis(i);
        if (area2Fill.shape()[maskIdx] != img.shape()[imgAxisIdx]) {
            return Multidim::Array<T, nDim>(); //shape mismatch
        }
    }

    std::vector<std::array<int,mDim>> nonInpaintedPoints;
    nonInpaintedPoints.reserve(area2Fill.flatLenght());

    int nIdxs = idxConverter.numberOfPossibleIndices();
    int nBatch = batchIdxConverter.numberOfPossibleIndices();

    for (int i = 0; i < nIdxs ; i++) {
        auto idx = idxConverter.getIndexFromPseudoFlatId(i);

        bool notToInpaint = !area2Fill.valueUnchecked(idx);

        if (notToInpaint) {
            nonInpaintedPoints.push_back(idx);
        }
    }

    if (nonInpaintedPoints.empty()) {
        return Multidim::Array<T, nDim>(); //impossible to inpaint when no fixed points are present
    }

    using PartitionTree = Geometry::GenericBSP<std::array<int,mDim>, mDim, Geometry::BSPObjectWrapper<std::array<int,mDim>, int>>;

    PartitionTree tree(std::move(nonInpaintedPoints));

    Multidim::Array<T, nDim> ret(img.shape(), img.strides());

    for (int i = 0; i < nIdxs ; i++) {
        auto idx = idxConverter.getIndexFromPseudoFlatId(i);
        auto imgIdx = axisIdxCompressors.getUncompressedIndex(idx);

        std::array<int,mDim> nearest = idx;
        if (area2Fill.valueUnchecked(idx)) {
            nearest = tree.closest(idx);
        }
        auto imgNearest = axisIdxCompressors.getUncompressedIndex(nearest);

        for (int j = 0; j < nBatch; j++) {
            auto batchPos = batchIdxConverter.getIndexFromPseudoFlatId(j);

            for (int d = 0; d < nBatchDims; d++) {
                imgIdx[batchDims[d]] = batchPos[d];
                imgNearest[batchDims[d]] = batchPos[d];

                ret.atUnchecked(imgIdx) = img.valueUnchecked(imgNearest);
            }
        }
    }

    
    return ret;
}

template <typename T, int nDim, typename ComputeT = float>
Multidim::Array<T, nDim> firstOrderDiffusionInPaintingMonochannel(Multidim::Array<T, nDim> const& img,
                                                                  Multidim::Array<bool, nDim> const& area2Fill) {


    using MatrixAType = Eigen::SparseMatrix<ComputeT>;
    using VectorBType = Eigen::Matrix<ComputeT, Eigen::Dynamic, 1>;

    using SolverType = Eigen::ConjugateGradient<MatrixAType, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<ComputeT>>;

    Multidim::Array<T, nDim> initial = nearestInPaintingMonochannel(img, area2Fill);

    if (initial.empty()) {
        return initial;
    }

    Multidim::IndexConverter<nDim> indexConverter(area2Fill.shape());

    int vecLen = indexConverter.numberOfPossibleIndices();

    MatrixAType D; //differential operator
    MatrixAType I; //identity operator
    VectorBType b = VectorBType::Zero(vecLen);
    VectorBType guess = VectorBType::Zero(vecLen);

    //Constraint I*x = b
    I.resize(vecLen, vecLen);
    I.reserve(Eigen::VectorXi::Constant(vecLen, 1)); //max one entry per column

    //Constraint D*x = 0
    D.resize(2*nDim*vecLen, vecLen); //backward and forward for all dimensions
    D.reserve(Eigen::VectorXi::Constant(vecLen, 4*nDim)); //reserve up to 4 non zero entry per column per dimension

    auto shape = img.shape();

    for (int i = 0; i < vecLen ; i++) {
        std::array<int,nDim> idx = indexConverter.getIndexFromPseudoFlatId(i);

        guess[i] = initial.valueUnchecked(idx);

        bool notToInpaint = !area2Fill.valueUnchecked(idx);

        if (notToInpaint) {
            I.insert(i,i) = 1;
            b[i] = img.valueUnchecked(idx);
        } else {

            int startRow = 0;

            for (int d = 0; d < nDim; d++) {
                std::array<int,nDim> deltaP = idx;
                std::array<int,nDim> deltaM = idx;

                deltaP[d] += 1;
                deltaM[d] -= 1;

                int ip = indexConverter.getPseudoFlatIdFromIndex(deltaP);
                int im = indexConverter.getPseudoFlatIdFromIndex(deltaM);

                if (deltaP[d] < shape[d]) {
                    D.insert(i+startRow,i) = -1;
                    D.insert(i+startRow,ip) = 1;
                }

                if (deltaM[d] >= 0) {
                    D.insert(i+startRow+vecLen,im) = -1;
                    D.insert(i+startRow+vecLen,i) = 1;
                }

                startRow += 2*vecLen;
            }

        }
    }

    MatrixAType A = I.transpose()*I + D.transpose()*D;
    VectorBType Itb = I.transpose()*b; //D is multiplied by 0 here

    SolverType solver;
    solver.compute(A);

    VectorBType solution = solver.solveWithGuess(Itb, guess);

    Multidim::Array<T, nDim> ret(img.shape(), img.strides());

    for (int i = 0; i < vecLen ; i++) {
        std::array<int,nDim> idx = indexConverter.getIndexFromPseudoFlatId(i);

        ret.atUnchecked(idx) = solution[i];

    }

    return ret;

}

} // namespace ImageProcessing
} // namespace StereoVision

#endif // INPAINTING_H
