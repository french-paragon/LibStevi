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
#include "../optimization/sphericaloptimization.h"

#include "../utils/types_manipulations.h"

#include <string>

namespace StereoVision {
namespace Correlation {

enum class matchingFunctions{
	CC = 0,
	NCC = 1,
	SSD = 2,
	SAD = 3,
	ZCC = 4,
	ZNCC = 5,
	ZSSD = 6,
	ZSAD = 7
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
		float tmp = float(source.valueUnchecked(i)) - float(target.valueUnchecked(i));
		if (std::is_integral_v<T_O>) {
			score += static_cast<T_O>(std::abs(tmp));
		} else {
			score += std::fabs(tmp);
		}
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

} //namespace Correlation
} //namespace StereoVision

#endif // STEREOVISION_MATCHING_COSTS_H
