#ifndef CORNERDETECTORS_H
#define CORNERDETECTORS_H

#include <MultidimArrays/MultidimArrays.h>

#include "../imageProcessing/convolutions.h"
#include "../imageProcessing/standardConvolutionFilters.h"

namespace StereoVision {
namespace SparseMatching {

template<typename T>
Multidim::Array<T, 2> HarrisCornerScore(Multidim::Array<T, 2, Multidim::ConstView> const& img, int lowPassRadius = 3) {

    using MovingAxis = ImageProcessing::Convolution::MovingWindowAxis;

    using Padding = ImageProcessing::Convolution::PaddingInfos;

    using FiltType = ImageProcessing::Convolution::Filter<float, MovingAxis, MovingAxis>;

    constexpr bool normalize = false;
    float sigma = static_cast<float>(lowPassRadius+1)/2;

    Padding pad(lowPassRadius, ImageProcessing::Convolution::PaddingType::Mirror);
    Padding diff_pad(lowPassRadius, ImageProcessing::Convolution::PaddingType::Mirror);

    std::array<FiltType, 2> separatedGaussianFilter =
            ImageProcessing::Convolution::separatedGaussianFilters(sigma, lowPassRadius, normalize, MovingAxis(pad), MovingAxis(pad));

    std::array<FiltType, 2> finiteDifferencesFilters =
            ImageProcessing::Convolution::finiteDifferencesKernels<T>(MovingAxis(pad), MovingAxis(pad));

    std::array<FiltType, 2> extendingKernels =
            ImageProcessing::Convolution::extendLinearKernels<T>(MovingAxis(pad), MovingAxis(pad));


    Multidim::Array<T, 2> lp_filtered = separatedGaussianFilter[0].convolve(img);

    for (int i = 1; i < separatedGaussianFilter.size(); i++) {
        lp_filtered = separatedGaussianFilter[i].convolve(lp_filtered);
    }

    std::array<Multidim::Array<T, 2>, 2> diffs;

    for (int diff = 0; diff < 2; diff++) {

        FiltType filt1 = (diff == 0) ? finiteDifferencesFilters[0] : extendingKernels[0];

        diffs[diff] = filt1.convolve(lp_filtered);

        FiltType filt2 = (diff == 1) ? finiteDifferencesFilters[1] : extendingKernels[1];

        diffs[diff] = filt2.convolve(diffs[diff]);
    }

    auto shape = img.shape();
    Multidim::Array<T, 2> response(shape);

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[0]; j++) {

            T d0 = diffs[0].valueUnchecked(i,j);
            T d1 = diffs[1].valueUnchecked(i,j);

            T d02 = d0*d0;
            T d0d1 = d0*d1;
            T d12 = d1*d1;

            T det = d02*d12 - d0d1*d0d1;
            T tr = d02 + d12;

            response.atUnchecked(i,j) = det/tr;

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

} // namespace SparseMatching
} // namespace StereoVision

#endif // CORNERDETECTORS_H
