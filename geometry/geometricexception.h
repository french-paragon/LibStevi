#ifndef STEREOVISION_GEOMETRICEXCEPTION_H
#define STEREOVISION_GEOMETRICEXCEPTION_H

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

#include <string>
#include <exception>
#include <cstring>

namespace StereoVision {
namespace Geometry {

class [[deprecated("Usage is not recommanded, use status optional instead to manage failure cases!")]] GeometricException : public std::exception
{
public:
	inline GeometricException(std::string const& what)
	{
		_what = new char[what.size()];
		std::memcpy (_what, what.c_str(), what.size());
	}
	inline GeometricException(GeometricException const& other) :
		GeometricException(other.what())
	{

	}

	inline virtual ~GeometricException() {
		delete _what;
	}

	inline const char* what() const noexcept override {
		return _what;
	}

protected:

	char* _what;
};

} // namespace Geometry
} // namespace StereoVision

#endif // STEREOVISIONAPP_GEOMETRICEXCEPTION_H
