#ifndef STEREOVISION_MATCHING_COSTS_H
#define STEREOVISION_MATCHING_COSTS_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022  Paragon<french.paragon@gmail.com>

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
};

template<matchingFunctions func>
class MatchingFunctionTraits{
};

template<class T_S, class T_T, class T_O = float>
inline T_O dotProduct(Multidim::Array<T_S,1> const& source,
					  Multidim::Array<T_T,1> const& target) {

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
inline T_O SumSquareDiff(Multidim::Array<T_S,1> const& source,
						 Multidim::Array<T_T,1> const& target) {

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
inline T_O SumAbsDiff(Multidim::Array<T_S,1> const& source,
					  Multidim::Array<T_T,1> const& target) {

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
inline T_O MedianAbsDiff(Multidim::Array<T_S,1> const& source,
						 Multidim::Array<T_T,1> const& target) {

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

template<class T_S, class T_T>
inline hamming_cv_t hammingDistance(Multidim::Array<T_S,1> const& source,
									Multidim::Array<T_T,1> const& target) {

	hamming_cv_t score = 0;

	for (int i = 0; i < source.shape()[0]; i++) {
		score += hammingScalar(source.valueUnchecked(i), target.valueUnchecked(i));
	}

	return score;
}

template<>
class MatchingFunctionTraits<matchingFunctions::NCC>{
public:
	static const std::string Name;
	static constexpr bool ZeroMean = false;
	static constexpr bool Normalized = true;
	static constexpr dispExtractionStartegy extractionStrategy = dispExtractionStartegy::Score;

	static constexpr bool isCensusBased = false;

	template<class T_S, class T_T, class T_O = float>
	inline static T_O featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return MedianAbsDiff<T_S, T_T, T_O>(source, target);
	}

	template<int dimsIn, int dimsOuts = Eigen::Dynamic>
	inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
																			 Eigen::Matrix<float,dimsOuts,1> const& b) {
		return Optimization::affineBestLeastMedianApproximation(A,b);
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return MedianAbsDiff<T_S, T_T, T_O>(source, target);
	}

	template<int dimsIn, int dimsOuts = Eigen::Dynamic>
	inline static Eigen::Matrix<float,dimsIn,1> barycentricBestApproximation(Eigen::Matrix<float,dimsOuts,dimsIn> const& A,
																			 Eigen::Matrix<float,dimsOuts,1> const& b) {
		return Optimization::affineBestLeastMedianApproximation(A,b);
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
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

	template<class T_S, class T_T, class T_O = float>
	inline static float featureComparison(Multidim::Array<T_S,1> const& source,
								   Multidim::Array<T_T,1> const& target) {
		return hammingDistance(source, target);
	}
};

template<matchingFunctions func, typename ImType>
struct MatchingFuncComputeTypeInfos {

	typedef float FeatureType;

	static constexpr bool SupportFloatCV = true;
	static constexpr bool SupportIntCV = false;
};

template<matchingFunctions func>
struct MatchingFuncComputeTypeInfos<func, uint8_t> {

	typedef typename std::conditional
						<MatchingFunctionTraits<func>::Normalized,
						int16_t,
						typename std::conditional<MatchingFunctionTraits<func>::ZeroMean, int16_t, uint8_t>::type>::type FeatureType;

	static constexpr bool SupportFloatCV = true;
	static constexpr bool SupportIntCV = true;
};

template<typename ImType>
struct MatchingFuncComputeTypeInfos<matchingFunctions::HAMMING, ImType> {

	typedef uint32_t FeatureType;

	static constexpr bool SupportFloatCV = false;
	static constexpr bool SupportIntCV = true;
};

template<>
struct MatchingFuncComputeTypeInfos<matchingFunctions::HAMMING, uint8_t> {

	typedef uint32_t FeatureType;

	static constexpr bool SupportFloatCV = false;
	static constexpr bool SupportIntCV = true;
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
