#ifndef STEREOVISION_MATCHING_COSTS_H
#define STEREOVISION_MATCHING_COSTS_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022-2023  Paragon<french.paragon@gmail.com>

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

#include <MultidimArrays/MultidimArrays.h>

#include "./correlation_base.h"

#include "../optimization/l1optimization.h"
#include "../optimization/l2optimization.h"
#include "../optimization/leastmedianoptimization.h"
#include "../optimization/sphericaloptimization.h"

#include "../utils/types_manipulations.h"

#include <string>

namespace StereoVision {
namespace Correlation {

enum class matchingFunctions{
    None = -1, //use to implement default implementation when
    CC = 0, //cross correlation
    NCC = 1, //normalized cross correlation
    SSD = 2, //sum of square differences
    SAD = 3, //sum of absolute differences
    ZCC = 4, //zero mean correlation
    ZNCC = 5, //zero mean normalized cross correlation
    ZSSD = 6, //zero mean sum of square differences
    ZSAD = 7, //zero mean sum of absolute differences
    MEDAD = 8, //median absolute difference (equivalent to median square difference)
    ZMEDAD = 9, //zero mean median absolute difference
    HAMMING = 10, //Hamming distance (to used with census and other binary features)
    CENSUS = 11, //Hamming distance (but make some intermediate functions to transform your features into census features)
    KERMI = 12, //kernel based mututal information estimation
};

template<matchingFunctions func>
class MatchingFunctionTraits{
};

template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
inline T_O dotProduct(Multidim::Array<T_S,1, viewConstness> const& source,
                      Multidim::Array<T_T,1, viewConstness> const& target) {

    static_assert ((std::is_integral_v<T_S> and std::is_integral_v<T_T>) or !std::is_integral_v<T_O>,
            "Cannot process floating point inputs for non floating point output");

    T_O score = 0;

    for (int i = 0; i < source.shape()[0]; i++) {
        if (std::is_integral_v<T_O> and (sizeof (T_O) < 2*std::max(sizeof (T_S), sizeof (T_T)))) { //need to renormalize
            score += (static_cast<T_O>(source.valueUnchecked(i))*static_cast<T_O>(target.valueUnchecked(i)))/TypesManipulations::equivalentOneForNormalizing<T_O>();
        } else {
            score += static_cast<T_O>(source.valueUnchecked(i))*static_cast<T_O>(target.valueUnchecked(i));
        }
    }

    return score;

}
template<class T_S, class T_T, class T_O = float>
inline T_O dotProduct(std::vector<T_S> const& source,
                      std::vector<T_T> const& target) {

    static_assert ((std::is_integral_v<T_S> and std::is_integral_v<T_T>) or !std::is_integral_v<T_O>,
            "Cannot process floating point inputs for non floating point output");

    T_O score = 0;

    for (size_t i = 0; i < source.size(); i++) {
        if (std::is_integral_v<T_O> and (sizeof (T_O) < 2*std::max(sizeof (T_S), sizeof (T_T)))) { //need to renormalize
            score += (static_cast<T_O>(source[i])*static_cast<T_O>(target[i]))/TypesManipulations::equivalentOneForNormalizing<T_O>();
        } else {
            score += static_cast<T_O>(source[i])*static_cast<T_O>(target[i]);
        }
    }

    return score;

}

template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
inline T_O SumSquareDiff(Multidim::Array<T_S,1, viewConstness> const& source,
                         Multidim::Array<T_T,1, viewConstness> const& target) {

    static_assert ((std::is_integral_v<T_S> and std::is_integral_v<T_T>) or !std::is_integral_v<T_O>,
            "Cannot process floating point inputs for non floating point output");

    T_O score = 0;

    for (int i = 0; i < source.shape()[0]; i++) {
        T_O tmp = static_cast<T_O>(source.valueUnchecked(i)) - static_cast<T_O>(target.valueUnchecked(i));
        score += tmp*tmp;
    }

    return score;

}

template<class T_S, class T_T, class T_O = float>
inline T_O SumSquareDiff(std::vector<T_S> const& source,
                         std::vector<T_T> const& target) {

    static_assert ((std::is_integral_v<T_S> and std::is_integral_v<T_T>) or !std::is_integral_v<T_O>,
            "Cannot process floating point inputs for non floating point output");

    T_O score = 0;

    for (int i = 0; i < source.size(); i++) {
        T_O tmp = static_cast<T_O>(source[i]) - static_cast<T_O>(target[i]);
        score += tmp*tmp;
    }

    return score;

}

template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
inline T_O SumAbsDiff(Multidim::Array<T_S,1, viewConstness> const& source,
                      Multidim::Array<T_T,1, viewConstness> const& target) {

    static_assert ((std::is_integral_v<T_S> and std::is_integral_v<T_T>) or !std::is_integral_v<T_O>,
            "Cannot process floating point inputs for non floating point output");

    T_O score = 0;

    for (int i = 0; i < source.shape()[0]; i++) {
        T_O tmp = T_O(source.valueUnchecked(i)) - T_O(target.valueUnchecked(i));
        if (std::is_integral_v<T_O>) {
            score += static_cast<T_O>(std::abs(tmp));
        } else {
            score += std::fabs(tmp);
        }
    }

    return score;

}

template<class T_S, class T_T, class T_O = float>
inline T_O SumAbsDiff(std::vector<T_S> const& source,
                      std::vector<T_T> const& target) {

    static_assert ((std::is_integral_v<T_S> and std::is_integral_v<T_T>) or !std::is_integral_v<T_O>,
            "Cannot process floating point inputs for non floating point output");

    T_O score = 0;

    for (int i = 0; i < source.size(); i++) {
        T_O tmp = T_O(source[i]) - T_O(target[i]);
        if (std::is_integral_v<T_O>) {
            score += static_cast<T_O>(std::abs(tmp));
        } else {
            score += std::fabs(tmp);
        }
    }

    return score;

}


template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
inline T_O MedianAbsDiff(Multidim::Array<T_S,1, viewConstness> const& source,
                         Multidim::Array<T_T,1, viewConstness> const& target) {

    static_assert ((std::is_integral_v<T_S> and std::is_integral_v<T_T>) or !std::is_integral_v<T_O>,
            "Cannot process floating point inputs for non floating point output");

    T_O median = 0;

    int medianPos = source.shape()[0]/2;
    std::vector<T_O> diffs(source.shape()[0]);

    for (int i = 0; i < source.shape()[0]; i++) {
        T_O tmp = T_O(source.valueUnchecked(i)) - T_O(target.valueUnchecked(i));
        if (std::is_integral_v<T_O>) {
            diffs[i] = static_cast<T_O>(std::abs(tmp));
        } else {
            diffs[i] = std::fabs(tmp);
        }
    }

    std::nth_element(diffs.begin(), diffs.begin()+medianPos, diffs.end());
    return diffs[medianPos];

}


template<class T_S, class T_T, class T_O = float>
inline T_O MedianAbsDiff(std::vector<T_S> const& source,
                         std::vector<T_T> const& target) {

    static_assert ((std::is_integral_v<T_S> and std::is_integral_v<T_T>) or !std::is_integral_v<T_O>,
            "Cannot process floating point inputs for non floating point output");

    T_O median = 0;

    int medianPos = source.size()/2;
    std::vector<T_O> diffs(source.size());

    for (int i = 0; i < source.shape()[0]; i++) {
        T_O tmp = T_O(source[i]) - T_O(target[i]);
        if (std::is_integral_v<T_O>) {
            diffs[i] = static_cast<T_O>(std::abs(tmp));
        } else {
            diffs[i] = std::fabs(tmp);
        }
    }

    std::nth_element(diffs.begin(), diffs.begin()+medianPos, diffs.end());
    return diffs[medianPos];

}

typedef uint16_t hamming_cv_t;

template<class T_S, class T_T>
inline hamming_cv_t hammingScalar(T_S n1, T_T n2) {

    static_assert (sizeof (T_S) <= sizeof (uint32_t) and sizeof (T_T) <= sizeof (uint32_t),
            "Cannot process types that do not fit into a 32bit integer");

    uint32_t m = reinterpret_cast<uint32_t>(n1) xor reinterpret_cast<uint32_t>(n2);

#ifdef __GNUC__
    return static_cast<hamming_cv_t>(__builtin_popcountl(m));
#else
    std::bitset<std::numeric_limits<census_data_t>::digits> bs( m );
    return bs.count();
#endif
}

template<class T_S, class T_T, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
inline hamming_cv_t hammingDistance(Multidim::Array<T_S,1, viewConstness> const& source,
                                    Multidim::Array<T_T,1, viewConstness> const& target) {

    hamming_cv_t score = 0;

    for (int i = 0; i < source.shape()[0]; i++) {
        score += hammingScalar(source.valueUnchecked(i), target.valueUnchecked(i));
    }

    return score;
}

template<class T_S, class T_T>
inline hamming_cv_t hammingDistance(std::vector<T_S> const& source,
                                    std::vector<T_T> const& target) {

    hamming_cv_t score = 0;

    for (int i = 0; i < source.size(); i++) {
        score += hammingScalar(source[i], target[i]);
    }

    return score;
}

template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
inline T_O KernelBasedMututalInformation(
    Multidim::Array<T_S,1, viewConstness> const& source,
    Multidim::Array<T_T,1, viewConstness> const& target) {

    int nElements = source.flatLenght();

    double computed = 0;

    T_S minSource = source.valueUnchecked(0);
    T_S maxSource = source.valueUnchecked(0);

    T_T minTarget = target.valueUnchecked(0);
    T_T maxTarget = target.valueUnchecked(0);

    for (int i = 1; i < nElements; i++) {
        T_S sourceVal = source.valueUnchecked(i);
        T_S targetVal = target.valueUnchecked(i);

        minSource = std::min(minSource, sourceVal);
        maxSource = std::max(maxSource, sourceVal);

        minTarget = std::min(minTarget, targetVal);
        maxTarget = std::max(maxTarget, targetVal);
    }

    double sourceStd = double(maxSource - minSource)/nElements;
    double targetStd = double(maxTarget - minTarget)/nElements;

    auto kernelSource = [sourceStd] (double val) {
        double tmp = val / sourceStd;
        return std::exp(-tmp*tmp);
    };

    auto kernelTarget = [targetStd] (double val) {
        double tmp = val / targetStd;
        return std::exp(-tmp*tmp);
    };

    double score = 0;

    for (int i = 0; i < nElements; i++) {
        double pSource = 0;
        double pTarget = 0;
        double pJoint = 0;

        for (int j = 0; j < nElements; j++) {
            double sourceP = kernelSource(source.valueUnchecked(i) - source.valueUnchecked(j));
            double targetP = kernelTarget(target.valueUnchecked(i) - target.valueUnchecked(j));

            pSource += sourceP;
            pTarget += targetP;

            pJoint += targetP*sourceP;
        }

        double pJointInd = pSource*pTarget/nElements;

        score += pJoint*std::log(pJoint/pJointInd);
    }

    return score;

}

template<>
class MatchingFunctionTraits<matchingFunctions::None>{
//This specialization is not meant to be used in practice, but provide a default implementation when implementing conditional templates.
public:
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = true;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static T_O featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        (void) source;
        (void) target;
        return 0;
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        (void) A;
        (void) b;
        return Eigen::Matrix<float,dimsIn,1>();
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> subpartBarycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                                    Eigen::Matrix<float,dimsOuts,1> const& b,
                                                                                    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> const& testSetsIdxs) {
        (void) A;
        (void) b;
        (void) testSetsIdxs;
        return Eigen::Matrix<float,dimsIn,1>();
    }
};

namespace matchingcosts_details {
    DECLARE_METHOD_TEST(featureComparison, has_featureComparison);
    DECLARE_METHOD_TEST(barycentricBestApproximation, has_barycentricBestApproximation);
    DECLARE_METHOD_TEST(subpartBarycentricBestApproximation, has_subpartBarycentricBestApproximation);

    template<matchingFunctions func>
    class MatchingFunctionTraitsInfos{
    public:
        template<class T_S, class T_T, class T_O = float>
        static constexpr bool traitsHasFeatureComparison() {
                return has_featureComparison
                <MatchingFunctionTraits<func>,
                T_O(MatchingFunctionTraits<func>::*)(Multidim::Array<T_S,1> const&,
                                                     Multidim::Array<T_T,1> const&)>::value;
        }

        template<int dimsIn, int dimsOuts = Eigen::Dynamic>
        static constexpr bool traitsHasBarycentricBestApproximation() {
                return has_barycentricBestApproximation
                <MatchingFunctionTraits<func>,
                Eigen::Matrix<float,dimsIn,1>(MatchingFunctionTraits<func>::*)
                (Eigen::Matrix<float,dimsOuts,dimsIn> const&,
                 Eigen::Matrix<float,dimsOuts,1> const&)>::value;
        }

        template<int dimsIn, int dimsOuts = Eigen::Dynamic>
        static constexpr bool traitsHasSubpartBarycentricBestApproximation() {
                return has_subpartBarycentricBestApproximation
                <MatchingFunctionTraits<func>,
                Eigen::Matrix<float,dimsIn,1>(MatchingFunctionTraits<func>::*)
                (Eigen::Matrix<float,dimsOuts,dimsIn> const&,
                 Eigen::Matrix<float,dimsOuts,1> const&,
                 Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> const&)>::value;
        }

    };
};

template<>
class MatchingFunctionTraits<matchingFunctions::NCC>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = true;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Score;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static T_O featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return dotProduct<T_S, T_T, T_O>(source, target);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        return Optimization::sphericalAffineBestApproximation(A,b);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::CC>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Score;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return dotProduct<T_S, T_T, T_O>(source, target);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::SSD>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return SumSquareDiff<T_S, T_T, T_O>(source, target);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        return Optimization::affineBestL2Approximation(A,b);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::SAD>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return SumAbsDiff<T_S, T_T, T_O>(source, target);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        return Optimization::affineBestL1Approximation(A,b);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::ZCC>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = true;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Score;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return dotProduct<T_S, T_T, T_O>(source, target);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::ZNCC>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = true;
    static constexpr bool Normalized = true;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Score;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return dotProduct<T_S, T_T, T_O>(source, target);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        return Optimization::sphericalAffineBestApproximation(A,b);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::ZSSD>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = true;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return SumSquareDiff<T_S, T_T, T_O>(source, target);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        return Optimization::affineBestL2Approximation(A,b);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::ZSAD>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = true;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return SumAbsDiff<T_S, T_T, T_O>(source, target);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        return Optimization::affineBestL1Approximation(A,b);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::MEDAD>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return MedianAbsDiff<T_S, T_T, T_O>(source, target);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        return Optimization::affineBestLeastMedianApproximation(A,b);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> subpartBarycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                                    Eigen::Matrix<float,dimsOuts,1> const& b,
                                                                                    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> const& testSetsIdxs) {

        return Optimization::affineBestLeastMedianApproximation(A,b, testSetsIdxs);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::ZMEDAD>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = true;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return MedianAbsDiff<T_S, T_T, T_O>(source, target);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                             Eigen::Matrix<float,dimsOuts,1> const& b) {
        return Optimization::affineBestLeastMedianApproximation(A,b);
    }

    template<int dimsIn, int dimsOuts = Eigen::Dynamic>
    inline static Eigen::Matrix<float,dimsIn,1> subpartBarycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
                                                                                    Eigen::Matrix<float,dimsOuts,1> const& b,
                                                                                    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> const& testSetsIdxs) {

        return Optimization::affineBestLeastMedianApproximation(A,b, testSetsIdxs);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::HAMMING>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = true;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return hammingDistance(source, target);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::CENSUS>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Cost;

    static constexpr bool isCensusBased = true;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                   Multidim::Array<T_T,1, viewConstness> const& target) {
        return hammingDistance(source, target);
    }
};

template<>
class MatchingFunctionTraits<matchingFunctions::KERMI>{
public:
    static const std::string Name;
    static constexpr bool ZeroMean = false;
    static constexpr bool Normalized = false;
    static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Score;

    static constexpr bool isCensusBased = false;

    template<class T_S, class T_T, class T_O = float, Multidim::ArrayDataAccessConstness viewConstness = Multidim::ConstView>
    inline static float featureComparison(Multidim::Array<T_S,1, viewConstness> const& source,
                                          Multidim::Array<T_T,1, viewConstness> const& target) {
        return KernelBasedMututalInformation(source, target);
    }
};



template<matchingFunctions func, class T_CV = float>
inline static constexpr T_CV defaultCvValForMatchFunc() {
    if (MatchingFunctionTraits<func>::extractionStrategy == dispExtractionStartegy::Cost) {
        return std::numeric_limits<T_CV>::max();
    } else {
        return std::numeric_limits<T_CV>::min();
    }
}

template<matchingFunctions func, class T_Disp, class T_CV = float>
/*!
 * \brief optimalDispAndCost return the optimal disparity and cost for a given matching function
 * \param current_disp the current disparity
 * \param current_cost the current cost
 * \param candidate_disp the candidate disparity
 * \param candidate_cost the candidate cost
 * \return the optimal disparity and cost
 */
inline std::pair<T_Disp const&,T_CV const&> optimalDispAndCost(T_Disp const& current_disp,
                                                 T_CV const& current_cost,
                                                 T_Disp const& candidate_disp,
                                                 T_CV const& candidate_cost) {

    if (MatchingFunctionTraits<func>::extractionStrategy == dispExtractionStartegy::Cost) {
        if (candidate_cost < current_cost) {
            return {candidate_disp, candidate_cost};
        }
    } else {
        if (candidate_cost > current_cost) {
            return {candidate_disp, candidate_cost};
        }
    }

    return {current_disp, current_cost};
}

template<matchingFunctions func, typename ImType>
struct MatchingFuncComputeTypeInfos {

    typedef TypesManipulations::accumulation_extended_t<ImType> FeatureType;

};

template<matchingFunctions func>
struct MatchingFuncComputeTypeInfos<func, uint8_t> {

    typedef typename std::conditional
                        <MatchingFunctionTraits<func>::Normalized,
                        int16_t,
                        typename std::conditional<MatchingFunctionTraits<func>::ZeroMean, int16_t, uint8_t>::type>::type FeatureType;

};

template<typename ImType>
struct MatchingFuncComputeTypeInfos<matchingFunctions::HAMMING, ImType> {

    typedef uint32_t FeatureType;

};

template<>
struct MatchingFuncComputeTypeInfos<matchingFunctions::HAMMING, uint8_t> {

    typedef uint32_t FeatureType;

};

typedef typename MatchingFuncComputeTypeInfos<matchingFunctions::HAMMING, uint8_t>::FeatureType census_data_t;

template<typename ImType>
struct MatchingFuncComputeTypeInfos<matchingFunctions::CENSUS, ImType> : public MatchingFuncComputeTypeInfos<matchingFunctions::HAMMING, ImType> {

};

template<>
struct MatchingFuncComputeTypeInfos<matchingFunctions::CENSUS, uint8_t> : public MatchingFuncComputeTypeInfos<matchingFunctions::HAMMING, uint8_t> {

};

template<matchingFunctions matchFunc, class T_I>
using FeatureTypeForMatchFunc = typename std::conditional<MatchingFunctionTraits<matchFunc>::isCensusBased,
                                                            census_data_t,
                                                            typename MatchingFuncComputeTypeInfos<matchFunc, T_I>::FeatureType>::type;

} //namespace Correlation
} //namespace StereoVision

#endif // STEREOVISION_MATCHING_COSTS_H
