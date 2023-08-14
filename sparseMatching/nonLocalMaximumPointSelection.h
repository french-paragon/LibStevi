#ifndef NONLOCALMAXIMUMPOINTSELECTION_H
#define NONLOCALMAXIMUMPOINTSELECTION_H

#include <MultidimArrays/MultidimArrays.h>

#include "../correlation/unfold.h"
#include "../imageProcessing/morphologicalOperators.h"

namespace StereoVision {
namespace SparseMatching {

template<typename T>
std::vector<std::array<float, 2>> nonLocalMaximumPointSelection(Multidim::Array<T, 2, Multidim::ConstView> const& score, int radius, T threshold, int n = -1) {

    if (n == 0) {
        return std::vector<std::array<float, 2>>();
    }

    Multidim::Array<T, 3> featureVolume = Correlation::unfold<T, T>(radius, radius, score, PaddingMargins());
    Multidim::Array<T, 2> localMax = ImageProcessing::maxFeature(featureVolume);

    std::array<int, 2> shape = score.shape();

    std::vector<std::array<float, 3>> tmp;

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            T val = score.valueUnchecked(i,j);
            if (val == localMax.valueUnchecked(i,j) and val >= threshold) {
                tmp.push_back({static_cast<float>(i),static_cast<float>(j), val});
            }
        }
    }

    std::sort(tmp.begin(), tmp.end(), [] (std::array<float, 3> const& a1, std::array<float, 3> const& a2) {
        return a1[2] > a2[2];
    });

    int n_selected = tmp.size();

    if (n > 0) {
        n_selected = std::min(n, n_selected);
    }

    std::vector<std::array<float, 2>> ret(n_selected);

    for (int i = 0; i < n_selected; i++) {
        ret[i][0] = tmp[i][0];
        ret[i][1] = tmp[i][1];
    }

    return ret;
}

} // namespace SparseMatching
} // namespace StereoVision

#endif // NONLOCALMAXIMUMPOINTSELECTION_H
