#ifndef STEREOVISION_PROPAGATION_DIRECTION_H
#define STEREOVISION_PROPAGATION_DIRECTION_H

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

#include <array>

namespace StereoVision {

class PropagationDirection {

public:
enum Direction{
	TopLeftToBottomRight = 0,
	TopRightToBottomLeft = 1,
	BottomLeftToTopRight = 2,
	BottomRightToTopLeft = 3
};

template<Direction dir>
class Traits{
};

template<>
class Traits<TopLeftToBottomRight>{
	constexpr static std::array<int, 2> increments = {1, 1};
};

template<>
class Traits<TopRightToBottomLeft>{
	constexpr static std::array<int, 2> increments = {1, -1};
};

template<>
class Traits<BottomLeftToTopRight>{
	constexpr static std::array<int, 2> increments = {-1, 1};
};

template<>
class Traits<BottomRightToTopLeft>{
	constexpr static std::array<int, 2> increments = {-1, -1};
};

struct IndexRange{
	int initial;
	int final;
};

template<int increment>
static constexpr IndexRange initialAndFinalPos(int rangeSize) {

	static_assert (increment == 1 or increment == -1, "Wrong direction template: increments have to be either 1 or -1");

	int initial = 0;
	int final = rangeSize;

	if (increment == -1) {
		initial = rangeSize;
		final = 0;
	}

	return {initial, final};
}


}; //class PropagationDirection

} //namespace StereoVision

#endif // STEREOVISION_PROPAGATION_DIRECTION_H
