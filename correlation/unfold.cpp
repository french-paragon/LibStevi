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

#include "unfold.h"

#include <algorithm>

namespace StereoVision {
namespace Correlation {

namespace CompressorGenerators {

Multidim::Array<int,2> GrPix17R3Filter() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<int,2> out(7,7);

	//superpixel 1
	out.at<Nc>(3,3) = 1;

	//superpixel 2
	out.at<Nc>(3,2) = 2;
	out.at<Nc>(3,1) = 2;

	//superpixel 3
	out.at<Nc>(3,4) = 3;
	out.at<Nc>(3,5) = 3;

	//superpixel 4
	out.at<Nc>(2,3) = 4;
	out.at<Nc>(1,3) = 4;

	//superpixel 5
	out.at<Nc>(4,3) = 5;
	out.at<Nc>(5,3) = 5;

	//superpixel 6
	out.at<Nc>(1,2) = 6;
	out.at<Nc>(2,1) = 6;
	out.at<Nc>(2,2) = 6;

	//superpixel 7
	out.at<Nc>(1,4) = 7;
	out.at<Nc>(2,5) = 7;
	out.at<Nc>(2,4) = 7;

	//superpixel 8
	out.at<Nc>(4,2) = 8;
	out.at<Nc>(4,1) = 8;
	out.at<Nc>(5,2) = 8;

	//superpixel 9
	out.at<Nc>(4,4) = 9;
	out.at<Nc>(4,5) = 9;
	out.at<Nc>(5,4) = 9;

	//superpixel 10
	out.at<Nc>(0,2) = 10;
	out.at<Nc>(0,3) = 10;
	out.at<Nc>(0,4) = 10;

	//superpixel 11
	out.at<Nc>(2,0) = 11;
	out.at<Nc>(3,0) = 11;
	out.at<Nc>(4,0) = 11;

	//superpixel 12
	out.at<Nc>(6,2) = 12;
	out.at<Nc>(6,3) = 12;
	out.at<Nc>(6,4) = 12;

	//superpixel 13
	out.at<Nc>(2,6) = 13;
	out.at<Nc>(3,6) = 13;
	out.at<Nc>(4,6) = 13;

	//superpixel 14
	out.at<Nc>(0,0) = 14;
	out.at<Nc>(0,1) = 14;
	out.at<Nc>(1,0) = 14;
	out.at<Nc>(1,1) = 14;

	//superpixel 15
	out.at<Nc>(5,0) = 15;
	out.at<Nc>(5,1) = 15;
	out.at<Nc>(6,0) = 15;
	out.at<Nc>(6,1) = 15;

	//superpixel 16
	out.at<Nc>(0,5) = 16;
	out.at<Nc>(0,6) = 16;
	out.at<Nc>(1,5) = 16;
	out.at<Nc>(1,6) = 16;

	//superpixel 17
	out.at<Nc>(5,5) = 17;
	out.at<Nc>(5,6) = 17;
	out.at<Nc>(6,5) = 17;
	out.at<Nc>(6,6) = 17;

	return out;

}
Multidim::Array<int,2> GrPix17R4Filter() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<int,2> out(9,9);

	//superpixel 1
	out.at<Nc>(4,4) = 1;

	//superpixel 2
	out.at<Nc>(4,3) = 2;
	out.at<Nc>(4,2) = 2;

	//superpixel 3
	out.at<Nc>(4,5) = 3;
	out.at<Nc>(4,6) = 3;

	//superpixel 4
	out.at<Nc>(3,4) = 4;
	out.at<Nc>(2,4) = 4;

	//superpixel 5
	out.at<Nc>(5,4) = 5;
	out.at<Nc>(6,4) = 5;

	//superpixel 6
	out.at<Nc>(2,2) = 6;
	out.at<Nc>(2,3) = 6;
	out.at<Nc>(3,2) = 6;
	out.at<Nc>(3,3) = 6;

	//superpixel 7
	out.at<Nc>(2,5) = 7;
	out.at<Nc>(2,6) = 7;
	out.at<Nc>(3,5) = 7;
	out.at<Nc>(3,6) = 7;

	//superpixel 8
	out.at<Nc>(5,2) = 8;
	out.at<Nc>(5,3) = 8;
	out.at<Nc>(6,2) = 8;
	out.at<Nc>(6,3) = 8;

	//superpixel 9
	out.at<Nc>(5,5) = 9;
	out.at<Nc>(5,6) = 9;
	out.at<Nc>(6,5) = 9;
	out.at<Nc>(6,6) = 9;

	//superpixel 10
	out.at<Nc>(0,3) = 10;
	out.at<Nc>(0,4) = 10;
	out.at<Nc>(0,5) = 10;
	out.at<Nc>(1,3) = 10;
	out.at<Nc>(1,4) = 10;
	out.at<Nc>(1,5) = 10;

	//superpixel 11
	out.at<Nc>(3,0) = 11;
	out.at<Nc>(4,0) = 11;
	out.at<Nc>(5,0) = 11;
	out.at<Nc>(3,1) = 11;
	out.at<Nc>(4,1) = 11;
	out.at<Nc>(5,1) = 11;

	//superpixel 12
	out.at<Nc>(7,3) = 12;
	out.at<Nc>(7,4) = 12;
	out.at<Nc>(7,5) = 12;
	out.at<Nc>(8,3) = 12;
	out.at<Nc>(8,4) = 12;
	out.at<Nc>(8,5) = 12;

	//superpixel 13
	out.at<Nc>(3,7) = 13;
	out.at<Nc>(4,7) = 13;
	out.at<Nc>(5,7) = 13;
	out.at<Nc>(3,8) = 13;
	out.at<Nc>(4,8) = 13;
	out.at<Nc>(5,8) = 13;

	//superpixel 14
	out.at<Nc>(0,0) = 14;
	out.at<Nc>(0,1) = 14;
	out.at<Nc>(1,0) = 14;
	out.at<Nc>(1,1) = 14;
	out.at<Nc>(0,2) = 14;
	out.at<Nc>(1,2) = 14;
	out.at<Nc>(2,0) = 14;
	out.at<Nc>(2,1) = 14;

	//superpixel 15
	out.at<Nc>(7,0) = 15;
	out.at<Nc>(7,1) = 15;
	out.at<Nc>(8,0) = 15;
	out.at<Nc>(8,1) = 15;
	out.at<Nc>(7,2) = 15;
	out.at<Nc>(8,2) = 15;
	out.at<Nc>(6,0) = 15;
	out.at<Nc>(6,1) = 15;

	//superpixel 16
	out.at<Nc>(0,7) = 16;
	out.at<Nc>(0,8) = 16;
	out.at<Nc>(1,7) = 16;
	out.at<Nc>(1,8) = 16;
	out.at<Nc>(0,6) = 16;
	out.at<Nc>(1,6) = 16;
	out.at<Nc>(2,7) = 16;
	out.at<Nc>(2,8) = 16;

	//superpixel 17
	out.at<Nc>(7,7) = 17;
	out.at<Nc>(7,8) = 17;
	out.at<Nc>(8,7) = 17;
	out.at<Nc>(8,8) = 17;
	out.at<Nc>(7,6) = 17;
	out.at<Nc>(8,6) = 17;
	out.at<Nc>(6,7) = 17;
	out.at<Nc>(6,8) = 17;

	return out;

}

} //namespace CompressorGenerators

} //namespace Correlation
} //namespace StereoVision
