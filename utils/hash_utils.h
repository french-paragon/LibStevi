#ifndef STEREOVISION_UTILS_HASH_UTILS_H
#define STEREOVISION_UTILS_HASH_UTILS_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2023  Paragon<french.paragon@gmail.com>

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

/*
 * This header define additional hash function for std containers.
 */

#include <functional>
#include <array>

namespace StereoVision {

template <typename T, typename... Rest>
void hashBunch(std::size_t & seed, const T& v, Rest... rest)
{
	std::hash<T> h;
	seed ^= h(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	(hashBunch(seed, rest), ...);
}

}

#define MAKE_HASHABLE(type, ...) \
	namespace std {\
		template<> struct hash<type> {\
			inline std::size_t operator()(const type &t) const {\
				std::size_t ret = 0;\
				StereoVision::hashBunch(ret, __VA_ARGS__);\
				return ret;\
			}\
		};\
	}

using array2i = std::array<int,2>;
MAKE_HASHABLE(array2i, t[0], t[1]);

using array3i = std::array<int,3>;
MAKE_HASHABLE(array3i, t[0], t[1], t[2]);

#endif // STEREOVISION_UTILS_HASH_UTILS_H
