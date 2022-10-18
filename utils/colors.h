#ifndef STEREOVISION_COLORS_H
#define STEREOVISION_COLORS_H

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

#include <string>

namespace StereoVision {

namespace Color {

enum RedGreenBlue {
	Red = 0,
	Green = 1,
	Blue = 2
};

enum ColorModel {
	RGB,
	BGR,
	RGBA,
	BGRA,
	CMYK,
	HSI,
	HSV,
	YUV,
	YUYV,
	YVYU
};

template<ColorModel CM>
class ColorModelTraits{
	static constexpr int nChannels() {
		switch (CM) {
		case YUYV:
		case YVYU:
			return 2;
		case RGB:
		case BGR:
		case HSI:
		case HSV:
		case YUV:
			return 3;
		case RGBA:
		case BGRA:
		case CMYK:
			return 4;
		}
	}

	static inline std::string readableName() {
		switch (CM) {
		case YUYV:
			return "YUYV";
		case YVYU:
			return "YVYU";
		case RGB:
			return "RGB";
		case BGR:
			return "BGR";
		case HSI:
			return "HSI";
		case HSV:
			return "HSV";
		case YUV:
			return "YUV";
		case RGBA:
			return "RGBA";
		case BGRA:
			return "BGRA";
		case CMYK:
			return "CMYK";
		}
	}
};

} // namespace Color
} // namespace StereoVision

#endif // COLORS_H
