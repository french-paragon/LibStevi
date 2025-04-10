#ifndef STEREOVISION_TYPES_MANIPULATIONS_H
#define STEREOVISION_TYPES_MANIPULATIONS_H

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

#include <cstdint>
#include <type_traits>
#include <string>
#include <sstream>
#include <limits>

#define DECLARE_METHOD_TEST(func, name)                                       \
	template<typename T, typename Sign>                                       \
	struct name {                                                             \
	private:                                                                  \
		typedef char yes[1];                                                  \
		typedef char no [2];                                                  \
		template <typename U, U> struct type_check;                           \
		template <typename A> static yes &chk(type_check<Sign, &A::func > *); \
		template <typename  > static no  &chk(...);                           \
	public:                                                                   \
		static bool const value = sizeof(chk<T>(0)) == sizeof(yes);           \
	}

namespace StereoVision {

namespace TypesManipulations {

template <class > struct acc_extended;
template <class T> using accumulation_extended_t = typename acc_extended<T>::type;
template <class T> struct tag { using type = T; };

template <> struct acc_extended<int8_t>  : tag<int16_t> { };
template <> struct acc_extended<int16_t> : tag<int32_t> { };
template <> struct acc_extended<int32_t> : tag<int32_t> { };
template <> struct acc_extended<int64_t> : tag<int64_t> { };

template <> struct acc_extended<uint8_t>  : tag<int16_t> { };
template <> struct acc_extended<uint16_t> : tag<int32_t> { };
template <> struct acc_extended<uint32_t> : tag<int64_t> { };
template <> struct acc_extended<uint64_t> : tag<int64_t> { };

template <> struct acc_extended<float> : tag<float> { };
template <> struct acc_extended<double> : tag<double> { };


template <class T>
/*!
 * \brief the equivalentOneForNormalizing constexpr give a value that normalization based algorithms can consider equivalent to 1 for the T type.
 * \return a value considered equvalent to 1.
 *
 * For floating point number, this value will be 1.
 * For integer types it will be a value that is both large enough to keep a good resolution after normalizing,
 * but small enough to limit the risk of overflow in the next set of operations.
 *
 * For integer types, the value is garanteed to be a power of 2 (for optimization).
 */
inline constexpr T equivalentOneForNormalizing() {
    using I_T = std::conditional_t<std::is_integral_v<T>, T, long>;
	if (std::is_integral_v<T>) {
                return I_T(1) << sizeof (T)*4;
	}
	return 1;
}

template<typename T>
inline std::string dtypeDescr() {
	if (!std::is_integral_v<T> and !std::is_floating_point_v<T>) {
		return "";
	}

	char symbol = 'i';

	if (std::is_integral_v<T> and !std::is_signed_v<T>) {
		symbol = 'u';
	} else if (std::is_floating_point_v<T>) {
		symbol = 'f';
	}

	int nBit = sizeof (T)*8;

	std::stringstream strs;
	strs << symbol << nBit;

	return strs.str();
}

template<typename T>
inline bool matchdescr(std::string descr) {
	return descr == dtypeDescr<T>();
}

template<typename T>
constexpr T defaultWhiteLevel() {

    return (std::is_integral_v<T>) ? std::numeric_limits<T>::max() : static_cast<T>(1.0);
}

template<typename T>
constexpr T defaultBlackLevel() {

    return static_cast<T>(0.0);
}

/*!
 * \brief typeExceedFloat32Precision check if the type T exceed the expected precision of a float32
 * \return true if the function detect that the type T is more accurate than a float32
 *
 * This function work by checking if some double, close to one another but sligthly different, are still different after being casted to type T.
 * This is potentially less accurate than using std::is_same or other form of direct type comparison,
 * but this accomodate arbitrary types (e.g. ceres Jets) without prior knowledge of said type.
 *
 * If type T is not a floating point type, then behavior is undefined.
 */
template<typename T>
constexpr bool typeExceedFloat32Precision() {
    T val = T(1.00000001);
    T test = T(1);

    return val > test;
}

} // namespace TypesManipulations

} // namespace StereoVision

#endif // STEREOVISION_TYPES_MANIPULATIONS_H
