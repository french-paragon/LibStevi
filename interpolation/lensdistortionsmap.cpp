#include "lensdistortionsmap.h"

#include "geometry/lensdistortion.h"

namespace StereoVision {
namespace Interpolation {

ImageArray computeLensDistortionMap(int height,
									int width,
									float f,
									Eigen::Vector2f pp,
									Eigen::Vector3f k123,
									Eigen::Vector2f t12,
									Eigen::Vector2f B12) {

	ImageArray out(height, width, 2);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			float unDist_im_x = j;
			float unDist_im_y = i;

			Eigen::Vector2f fPos(unDist_im_x, unDist_im_y);
			Eigen::Vector2f homogeneous = (fPos - pp)/f;

			Eigen::Vector2f dr = Geometry::radialDistortion(homogeneous, k123);
			Eigen::Vector2f dt = Geometry::tangentialDistortion(homogeneous, t12);

			homogeneous += dr + dt;

			Eigen::Vector2f dPos = Geometry::skewDistortion(homogeneous, B12, f, pp);

			out.at<Multidim::AccessCheck::Nocheck>(i, j, 0) = dPos.y();
			out.at<Multidim::AccessCheck::Nocheck>(i, j, 1) = dPos.x();
		}
	}

	return out;
}

} // namespace Interpolation
} // namespace StereoVision
