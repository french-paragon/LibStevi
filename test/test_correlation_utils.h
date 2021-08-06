#ifndef TEST_CORRELATION_UTILS_H
#define TEST_CORRELATION_UTILS_H

#include "../correlation/cross_correlations.h"
#include <cmath>

inline float InneficientZeromeanCrossCorrelation(Multidim::Array<float, 2> const& windows1,
												 Multidim::Array<float, 2> const& windows2) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = windows1.shape()[0];
	int w = windows1.shape()[1];

	if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
		return std::nanf("");
	}

	float mean1 = 0;
	float mean2 = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			mean1 += windows1.value<Nc>(i,j);
			mean2 += windows2.value<Nc>(i,j);

		}
	}

	mean1 /= h*w;
	mean2 /= h*w;

	float cc = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			float v1 = windows1.value<Nc>(i,j) - mean1;
			float v2 = windows2.value<Nc>(i,j) - mean2;

			cc += v1*v2;

		}
	}

	return cc;
}

inline float InneficientZeromeanNormalizedCrossCorrelation(Multidim::Array<float, 2> const& windows1,
														   Multidim::Array<float, 2> const& windows2) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = windows1.shape()[0];
	int w = windows1.shape()[1];

	if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
		return std::nanf("");
	}

	float mean1 = 0;
	float mean2 = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			mean1 += windows1.value<Nc>(i,j);
			mean2 += windows2.value<Nc>(i,j);

		}
	}

	mean1 /= h*w;
	mean2 /= h*w;

	float cc = 0;
	float s1 = 0;
	float s2 = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			float v1 = windows1.value<Nc>(i,j) - mean1;
			float v2 = windows2.value<Nc>(i,j) - mean2;

			cc += v1*v2;
			s1 += v1*v1;
			s2 += v2*v2;

		}
	}

	return cc/(sqrtf(s1)*sqrtf(s2));

}

inline float InneficientNormalizedCrossCorrelation(Multidim::Array<float, 2> const& windows1,
												   Multidim::Array<float, 2> const& windows2) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = windows1.shape()[0];
	int w = windows1.shape()[1];

	if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
		return std::nanf("");
	}

	float cc = 0;
	float s1 = 0;
	float s2 = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			float v1 = windows1.value<Nc>(i,j);
			float v2 = windows2.value<Nc>(i,j);

			cc += v1*v2;
			s1 += v1*v1;
			s2 += v2*v2;

		}
	}

	return cc/(sqrtf(s1)*sqrtf(s2));

}

inline float InneficientSumOfSquareDifferences(Multidim::Array<float, 2> const& windows1,
											   Multidim::Array<float, 2> const& windows2) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = windows1.shape()[0];
	int w = windows1.shape()[1];

	if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
		return std::nanf("");
	}

	float ss = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			float v1 = windows1.value<Nc>(i,j);
			float v2 = windows2.value<Nc>(i,j);

			ss += (v1 - v2)*(v1 - v2);

		}
	}

	return ss;

}

inline float InneficientZeroMeanSumOfSquareDifferences(Multidim::Array<float, 2> const& windows1,
													   Multidim::Array<float, 2> const& windows2) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = windows1.shape()[0];
	int w = windows1.shape()[1];

	if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
		return std::nanf("");
	}

	float mean1 = 0;
	float mean2 = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			mean1 += windows1.value<Nc>(i,j);
			mean2 += windows2.value<Nc>(i,j);

		}
	}

	mean1 /= h*w;
	mean2 /= h*w;

	float ss = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			float v1 = windows1.value<Nc>(i,j) - mean1;
			float v2 = windows2.value<Nc>(i,j) - mean2;

			ss += (v1 - v2)*(v1 - v2);

		}
	}

	return ss;

}


inline float InneficientSumOfAbsoluteDifferences(Multidim::Array<float, 2> const& windows1,
												 Multidim::Array<float, 2> const& windows2) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = windows1.shape()[0];
	int w = windows1.shape()[1];

	if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
		return std::nanf("");
	}

	float ss = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			float v1 = windows1.value<Nc>(i,j);
			float v2 = windows2.value<Nc>(i,j);

			ss += std::fabs(v1 - v2);

		}
	}

	return ss;

}

inline float InneficientZeroMeanSumOfAbsoluteDifferences(Multidim::Array<float, 2> const& windows1,
														 Multidim::Array<float, 2> const& windows2) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = windows1.shape()[0];
	int w = windows1.shape()[1];

	if (h != windows2.shape()[0] or w != windows2.shape()[1]) {
		return std::nanf("");
	}

	float mean1 = 0;
	float mean2 = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			mean1 += windows1.value<Nc>(i,j);
			mean2 += windows2.value<Nc>(i,j);

		}
	}

	mean1 /= h*w;
	mean2 /= h*w;

	float ss = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			float v1 = windows1.value<Nc>(i,j) - mean1;
			float v2 = windows2.value<Nc>(i,j) - mean2;

			ss += std::fabs(v1 - v2);

		}
	}

	return ss;

}

template<StereoVision::Correlation::matchingFunctions matchFunc>
inline float InneficientMatchingFunction(Multidim::Array<float, 2> const& windows1,
										 Multidim::Array<float, 2> const& windows2) {

	static_assert (matchFunc == StereoVision::Correlation::matchingFunctions::ZNCC or
			matchFunc == StereoVision::Correlation::matchingFunctions::NCC or
			matchFunc == StereoVision::Correlation::matchingFunctions::ZCC or
			matchFunc == StereoVision::Correlation::matchingFunctions::SSD or
			matchFunc == StereoVision::Correlation::matchingFunctions::ZSSD or
			matchFunc == StereoVision::Correlation::matchingFunctions::SAD or
			matchFunc == StereoVision::Correlation::matchingFunctions::ZSAD,
			"Unsopported matching function type.");

	if (matchFunc == StereoVision::Correlation::matchingFunctions::ZCC) {
		return InneficientZeromeanCrossCorrelation(windows1, windows2);
	} else if (matchFunc == StereoVision::Correlation::matchingFunctions::ZNCC) {
		return InneficientZeromeanNormalizedCrossCorrelation(windows1, windows2);
	} else if (matchFunc == StereoVision::Correlation::matchingFunctions::NCC) {
		return InneficientNormalizedCrossCorrelation(windows1, windows2);
	} else if (matchFunc == StereoVision::Correlation::matchingFunctions::ZSSD) {
		return InneficientZeroMeanSumOfSquareDifferences(windows1, windows2);
	} else if (matchFunc == StereoVision::Correlation::matchingFunctions::SSD) {
		return InneficientSumOfSquareDifferences(windows1, windows2);
	} else if (matchFunc == StereoVision::Correlation::matchingFunctions::ZSAD) {
		return InneficientZeroMeanSumOfAbsoluteDifferences(windows1, windows2);
	} else if (matchFunc == StereoVision::Correlation::matchingFunctions::SAD) {
		return InneficientSumOfAbsoluteDifferences(windows1, windows2);
	}

	return std::nanf("");
}

#endif // TEST_CORRELATION_UTILS_H
