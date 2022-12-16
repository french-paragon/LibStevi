#ifndef STEREOVISION_MATH_H
#define STEREOVISION_MATH_H

namespace StereoVision {
namespace Math {

template<int n, typename T>
T iPow(T f) {
	T r = 1;

	for (int i = 0; i < n; i++) {
		r *= f;
	}

	return r;
}

}
}

#endif // STEREOVISION_MATH_H
