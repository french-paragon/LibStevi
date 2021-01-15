#ifndef IMAGECOORDINATES_H
#define IMAGECOORDINATES_H

#include "geometry/core.h"

namespace StereoVision {
namespace Geometry {

enum class ImageAnchors : char {
	TopLeft,
	TopRight,
	BottomLeft,
	BottomRight
};

Eigen::Vector2f Image2HomogeneousCoordinates(Eigen::Vector2f const& pt,
											 float fx,
											 float fy,
											 Eigen::Vector2f const& pp,
											 ImageAnchors imageOrigin);

Eigen::Vector2f Image2HomogeneousCoordinates(Eigen::Vector2f const& pt,
											 float f,
											 Eigen::Vector2f const& pp,
											 ImageAnchors imageOrigin);

Eigen::Array2Xf Image2HomogeneousCoordinates(Eigen::Array2Xf const& pt,
											 float fx,
											 float fy,
											 Eigen::Vector2f const& pp,
											 ImageAnchors imageOrigin);

Eigen::Array2Xf Image2HomogeneousCoordinates(Eigen::Array2Xf const& pt,
											 float f,
											 Eigen::Vector2f const& pp,
											 ImageAnchors imageOrigin);

} // namespace Geometry
} // namespace StereoVision

#endif // IMAGECOORDINATES_H
