#ifndef CORNERDETECTORS_H
#define CORNERDETECTORS_H

#include <MultidimArrays/MultidimArrays.h>

#include "../imageProcessing/convolutions.h"
#include "../imageProcessing/standardConvolutionFilters.h"

namespace StereoVision {
namespace SparseMatching {

/*!
 * \brief HarrisCornerMMat compute the coefficient of the M windows of the Harris corner detector
 * \param img the image to perform the computation on
 * \param lowPassRadius the radius of the low pass filter applied to smooth the image.
 * \return
 */
template<typename T>
Multidim::Array<T, 3> HarrisCornerMMat(Multidim::Array<T, 2, Multidim::ConstView> const& img, int lowPassRadius = 3) {

    using MovingAxis = ImageProcessing::Convolution::MovingWindowAxis;

    using Padding = ImageProcessing::Convolution::PaddingInfos;

    using FiltType = ImageProcessing::Convolution::Filter<float, MovingAxis, MovingAxis>;

    constexpr bool normalize = false;
    float sigma = static_cast<float>(lowPassRadius+1)/2;

    Padding diff_pad(lowPassRadius, ImageProcessing::Convolution::PaddingType::Mirror);

    std::array<FiltType, 2> finiteDifferencesFilters =
            ImageProcessing::Convolution::finiteDifferencesKernels<T>(MovingAxis(diff_pad), MovingAxis(diff_pad));

    std::array<FiltType, 2> extendingKernels =
            ImageProcessing::Convolution::extendLinearKernels<T>(MovingAxis(diff_pad), MovingAxis(diff_pad));


    Multidim::Array<T, 2> lp_filtered;

    if (lowPassRadius >= 1) {

        Padding pad(lowPassRadius, ImageProcessing::Convolution::PaddingType::Mirror);

        std::array<FiltType, 2> separatedGaussianFilter =
                ImageProcessing::Convolution::separatedGaussianFilters(sigma, lowPassRadius, normalize, MovingAxis(pad), MovingAxis(pad));

        lp_filtered = separatedGaussianFilter[0].convolve(img);

        for (int i = 1; i < separatedGaussianFilter.size(); i++) {
            lp_filtered = separatedGaussianFilter[i].convolve(lp_filtered);
        }

    } else {
        lp_filtered = img;
    }

    std::array<Multidim::Array<T, 2>, 2> diffs;

    for (int diff = 0; diff < 2; diff++) {

        FiltType filt1 = (diff == 0) ? finiteDifferencesFilters[0] : extendingKernels[0];

        diffs[diff] = filt1.convolve(lp_filtered);

        FiltType filt2 = (diff == 1) ? finiteDifferencesFilters[1] : extendingKernels[1];

        diffs[diff] = filt2.convolve(diffs[diff]);
    }

    std::array<int,3> shape = {img.shape()[0], img.shape()[1], 3};
    Multidim::Array<T, 3> M(shape);

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {

            T d0 = diffs[0].valueUnchecked(i,j);
            T d1 = diffs[1].valueUnchecked(i,j);

            M.atUnchecked(i,j,0) = d0*d0;
            M.atUnchecked(i,j,1) = d0*d1;
            M.atUnchecked(i,j,2) = d1*d1;

        }
    }

    return M;

}

template<typename T>
Multidim::Array<T, 2> HarrisCornerScore(Multidim::Array<T, 2, Multidim::ConstView> const& img, int lowPassRadius = 3) {

    Multidim::Array<T, 3> M = HarrisCornerMMat(img, lowPassRadius);

    std::array<int,3> shape = M.shape();

    Multidim::Array<T, 2> response(shape[0], shape[1]);

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {

            T d02 = M.valueUnchecked(i,j,0);
            T d0d1 = M.valueUnchecked(i,j,1);
            T d12 = M.valueUnchecked(i,j,2);

            T det = d02*d12 - d0d1*d0d1;
            T tr = d02 + d12;

            T score = det/tr;

            if (std::isinf(score) or std::isnan(score)) {
                score = 0;
            }

            response.atUnchecked(i,j) = score;

        }
    }

    return response;

}

template<typename T>
Multidim::Array<T, 2> HarrisCornerScore(Multidim::Array<T, 3, Multidim::ConstView> const& img, int lowPassRadius = 3, int batchDim = 2) {

    if (batchDim < 0 or batchDim >= 3) {
        return Multidim::Array<T, 2>();
    }

    auto shape = img.shape();
    std::array<int, 2> outshape;

    for (int i = 0; i < 3; i++) {
        if (i == batchDim) {
            continue;
        }

        int id = i;

        if (i > batchDim) {
            id-=1;
        }

        outshape[id] = shape[i];
    }

    Multidim::Array<T, 2> out(outshape);

    for (int i = 0; i < outshape[0]; i++) {
        for (int j = 0; j < outshape[1]; j++) {
            out.atUnchecked(i,j) = 0;
        }
    }

    for (int b = 0; b < shape[batchDim]; b++) {

        Multidim::Array<T, 2> response = HarrisCornerScore(img.sliceView(batchDim, b), lowPassRadius);

        for (int i = 0; i < outshape[0]; i++) {
            for (int j = 0; j < outshape[1]; j++) {
                out.atUnchecked(i,j) += response.atUnchecked(i,j);
            }
        }

    }

    return out;

}

template<typename T>
Multidim::Array<T, 2> windowedHarrisCornerScore(Multidim::Array<T, 2, Multidim::ConstView> const& img, int windowRadius = 2, int lowPassRadius = 0) {

    Multidim::Array<T, 3> M = HarrisCornerMMat(img, lowPassRadius);

    std::array<int,3> shape = M.shape();

    Multidim::Array<T, 2> response(shape[0], shape[1]);

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {

            T d02 = 0;
            T d0d1 = 0;
            T d12 = 0;

            for (int di = -windowRadius; di <= windowRadius; di++) {

                if (i+di < 0 or i+di >= shape[0]) {
                    continue;
                }

                for (int dj = -windowRadius; dj <= windowRadius; dj++) {

                    if (j+dj < 0 or j+dj >= shape[1]) {
                        continue;
                    }

                    d02 += M.valueUnchecked(i+di,j+dj,0);
                    d0d1 += M.valueUnchecked(i+di,j+dj,1);
                    d12 += M.valueUnchecked(i+di,j+dj,2);
                }
            }

            T det = d02*d12 - d0d1*d0d1;
            T tr = d02 + d12;

            T score = det/tr;

            if (std::isinf(score) or std::isnan(score)) {
                score = 0;
            }

            response.atUnchecked(i,j) = score;

        }
    }

    return response;

}

template<typename T>
Multidim::Array<T, 2> windowedHarrisCornerScore(Multidim::Array<T, 3, Multidim::ConstView> const& img,
                                                int windowRadius = 2,
                                                int lowPassRadius = 0,
                                                int batchDim = 2) {

    int nArrays = img.shape()[batchDim];

    std::vector<Multidim::Array<T, 3>> M(nArrays);

    std::array<int,3> shape = img.shape();

    std::array<int,2> resp_shape;

    resp_shape[0] = shape[(batchDim == 0) ? 1 : 0];
    resp_shape[1] = shape[(batchDim < 2) ? 2 : 1];

    Multidim::Array<T, 2> response(resp_shape);

    for (int b = 0; b < nArrays; b++) {
        M[b] = HarrisCornerMMat(img.sliceView(batchDim, b), lowPassRadius);
    }

    for (int i = 0; i < resp_shape[0]; i++) {
        for (int j = 0; j < resp_shape[1]; j++) {

            T d02 = 0;
            T d0d1 = 0;
            T d12 = 0;

            for (int b = 0; b < nArrays; b++) {
                for (int di = -windowRadius; di <= windowRadius; di++) {

                    if (i+di < 0 or i+di >= shape[0]) {
                        continue;
                    }

                    for (int dj = -windowRadius; dj <= windowRadius; dj++) {

                        if (j+dj < 0 or j+dj >= shape[1]) {
                            continue;
                        }

                        d02 += M[b].valueUnchecked(i+di,j+dj,0);
                        d0d1 += M[b].valueUnchecked(i+di,j+dj,1);
                        d12 += M[b].valueUnchecked(i+di,j+dj,2);
                    }
                }
            }

            T det = d02*d12 - d0d1*d0d1;
            T tr = d02 + d12;

            T score = det/tr;

            if (std::isinf(score) or std::isnan(score)) {
                score = 0;
            }

            response.atUnchecked(i,j) = score;

        }
    }

    return response;

}

} // namespace SparseMatching
} // namespace StereoVision

#endif // CORNERDETECTORS_H
