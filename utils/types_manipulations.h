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
	if (std::is_integral_v<T>) {
		return 1 << sizeof (T)*4;
	}
	return 1;
}

} // namespace TypesManipulations

} // namespace StereoVision

#endif // STEREOVISION_TYPES_MANIPULATIONS_H
