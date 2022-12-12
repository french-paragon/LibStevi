#ifndef STEREOVISION_IMAGECOORDINATES_H
#define STEREOVISION_IMAGECOORDINATES_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021-2022  Paragon<french.paragon@gmail.com>

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

#include "../geometry/core.h"

namespace StereoVision {
namespace Geometry {

enum class ImageAxis : char {
	X,
	Y
};

enum class ImageAnchors : char {
	TopLeft,
	TopRight,
	BottomLeft,
	BottomRight
};

template <typename Pos_T>
Eigen::Matrix<Pos_T, 2,1> Image2HomogeneousCoordinates(Eigen::Matrix<Pos_T, 2,1> const& pt,
													   Eigen::Matrix<Pos_T, 2,1> const& f,
													   Eigen::Matrix<Pos_T, 2,1> const& pp,
													   ImageAnchors imageOrigin) {

	Eigen::Matrix<Pos_T, 2,1> r = pt - pp;
	r[0] /= f[0];
	r[1] /= f[1];

	switch (imageOrigin) {
	case ImageAnchors::TopLeft:
		break;
	case ImageAnchors::TopRight:
		r.y() = -r.y();
		break;
	case ImageAnchors::BottomLeft:
		r.x() = -r.x();
		break;
	case ImageAnchors::BottomRight:
		r.x() = -r.x();
		r.y() = -r.y();
		break;
	}

	return r;

}
template <typename Pos_T>
Eigen::Vector2f Image2HomogeneousCoordinates(Eigen::Matrix<Pos_T, 2,1> const& pt,
											 Pos_T f,
											 Eigen::Matrix<Pos_T, 2,1> const& pp,
											 ImageAnchors imageOrigin) {
	return Image2HomogeneousCoordinates(pt, Eigen::Matrix<Pos_T, 2,1>(f, f), pp, imageOrigin);
}

template <typename Pos_T, int nCols>
Eigen::Array<Pos_T, 2, nCols> Image2HomogeneousCoordinates(Eigen::Array<Pos_T, 2, nCols> const& pt,
														   Eigen::Matrix<Pos_T, 2,1> f,
														   Eigen::Matrix<Pos_T, 2,1> const& pp,
														   ImageAnchors imageOrigin) {

	Eigen::Array<Pos_T, 2, nCols> r = pt;
	r.row(0) -= pp.x();
	r.row(1) -= pp.y();

	r.row(0) /= f[0];
	r.row(1) /= f[1];

	switch (imageOrigin) {
	case ImageAnchors::TopLeft:
		break;
	case ImageAnchors::TopRight:
		r.row(1) *= -1;
		break;
	case ImageAnchors::BottomLeft:
		r.row(0) *= -1;
		break;
	case ImageAnchors::BottomRight:
		r.row(0) *= -1;
		r.row(1) *= -1;
		break;
	}

	return r;

}

template <typename Pos_T, int nCols>
Eigen::Array<Pos_T, 2, nCols> Image2HomogeneousCoordinates(Eigen::Array<Pos_T, 2, nCols> const& pt,
														   float f,
														   Eigen::Matrix<Pos_T, 2,1> const& pp,
														   ImageAnchors imageOrigin) {
	return Image2HomogeneousCoordinates(pt, Eigen::Matrix<Pos_T, 2,1>(f, f), pp, imageOrigin);
}

template <typename Pos_T>
Eigen::Matrix<Pos_T, 2,1> Homogeneous2ImageCoordinates(Eigen::Matrix<Pos_T, 2,1> const& hom,
													   Eigen::Matrix<Pos_T, 2,1> const& f,
													   Eigen::Matrix<Pos_T, 2,1> const& pp,
													   ImageAnchors imageOrigin = ImageAnchors::TopLeft)
{
	Eigen::Matrix<Pos_T, 2,1> r = hom;

	switch (imageOrigin) {
	case ImageAnchors::TopLeft:
		break;
	case ImageAnchors::TopRight:
		r.y() = -r.y();
		break;
	case ImageAnchors::BottomLeft:
		r.x() = -r.x();
		break;
	case ImageAnchors::BottomRight:
		r.x() = -r.x();
		r.y() = -r.y();
		break;
	}

	r[0] *= f[0];
	r[1] *= f[1];
	r += pp;

	return r;

}

template <typename Pos_T>
Eigen::Matrix<Pos_T, 2,1> Homogeneous2ImageCoordinates(Eigen::Matrix<Pos_T, 2,1> const& hom,
													   Pos_T f,
													   Eigen::Matrix<Pos_T, 2,1> const& pp,
													   ImageAnchors imageOrigin = ImageAnchors::TopLeft)
{
	return Homogeneous2ImageCoordinates(hom, Eigen::Matrix<Pos_T, 2,1>(f,f), pp, imageOrigin);
}


} // namespace Geometry
} // namespace StereoVision

#endif // IMAGECOORDINATES_H
