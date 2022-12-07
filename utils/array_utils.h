#ifndef STEREOVISION_ARRAY_UTILS_H
#define STEREOVISION_ARRAY_UTILS_H

#include <array>

namespace StereoVision {

template<typename T, int n>
constexpr std::array<T,n> constantArray(T val) {
	std::array<T,n> arr;

	for (int i = 0; i < n; i++) {
		arr[i] = n;
	}

	return arr;
}

} // namespace StereoVision

#endif // ARRAY_UTILS_H
