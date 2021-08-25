#ifndef STEREOVISION_UTILS_MARGINS_H
#define STEREOVISION_UTILS_MARGINS_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021  Paragon<french.paragon@gmail.com>

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

namespace StereoVision {

class Margins {

public:

	Margins() :
		_left(0),
		_right(0),
		_top(0),
		_bottom(0)
	{

	}
	Margins(int padding) :
		_left(padding),
		_right(padding),
		_top(padding),
		_bottom(padding)
	{

	}
	Margins(int leftright, int topbottom) :
		_left(leftright),
		_right(leftright),
		_top(topbottom),
		_bottom(topbottom)
	{

	}
	Margins(int left, int top, int right, int bottom) :
		_left(left),
		_right(right),
		_top(top),
		_bottom(bottom)
	{

	}

	Margins(Margins const& other) :
		_left(other._left),
		_right(other._right),
		_top(other._top),
		_bottom(other._bottom)
	{

	}

	Margins& operator=(Margins const& other)
	{
		_left = other._left;
		_right = other._right;
		_top = other._top;
		_bottom = other._bottom;

		return *this;
	}

	inline int left() const { return _left; }
	inline int right() const { return _right; }
	inline int top() const  { return _top; }
	inline int bottom() const  { return _bottom; }

protected:

	int _left;
	int _right;
	int _top;
	int _bottom;

};


class PaddingMargins : public Margins {

public:

	PaddingMargins() :
		Margins(),
		_auto(true)
	{

	}
	PaddingMargins(int padding) :
		Margins(padding),
		_auto(false)
	{

	}
	PaddingMargins(int leftright, int topbottom) :
		Margins(leftright, topbottom),
		_auto(false)
	{

	}
	PaddingMargins(int left, int top, int right, int bottom) :
		Margins(left, top, right, bottom),
		_auto(false)
	{

	}

	PaddingMargins(Margins const& other) :
		Margins(other),
		_auto(false)
	{

	}

	PaddingMargins(PaddingMargins const& other) :
		Margins(other._left, other._top, other._right, other._bottom),
		_auto(other._auto)
	{

	}

	PaddingMargins& operator=(Margins const& other)
	{
		static_cast<Margins*>(this)->operator=(other);
		_auto = false;

		return *this;
	}

	PaddingMargins& operator=(PaddingMargins const& other)
	{
		_left = other._left;
		_right = other._right;
		_top = other._top;
		_bottom = other._bottom;
		_auto = other._auto;

		return *this;
	}

	inline bool isAuto() const { return _auto; }

protected:

	bool _auto;

};

} //namespace StereoVision

#endif // MARGINS_H
