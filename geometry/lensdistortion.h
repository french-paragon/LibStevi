#ifndef STEREOVISIONAPP_LENSDISTORTION_H
#define STEREOVISIONAPP_LENSDISTORTION_H

#include "geometry/core.h"

namespace StereoVision {
namespace Geometry {

Eigen::Vector2f radialDistortion(Eigen::Vector2f pos, Eigen::Vector3f k123);
Eigen::Vector2d radialDistortionD(Eigen::Vector2d pos, Eigen::Vector3d k123);

Eigen::Vector2f tangentialDistortion(Eigen::Vector2f pos, Eigen::Vector2f t12);
Eigen::Vector2d tangentialDistortionD(Eigen::Vector2d pos, Eigen::Vector2d t12);

Eigen::Vector2f invertRadialDistorstion(Eigen::Vector2f pos,
										Eigen::Vector3f k123,
										int iters = 5);
Eigen::Vector2f invertTangentialDistorstion(Eigen::Vector2f pos,
											Eigen::Vector2f t12,
											int iters = 5);
Eigen::Vector2f invertRadialTangentialDistorstion(Eigen::Vector2f pos,
												  Eigen::Vector3f k123,
												  Eigen::Vector2f t12,
												  int iters = 5);

Eigen::Vector2f skewDistortion(Eigen::Vector2f pos, Eigen::Vector2f B12, float f, Eigen::Vector2f pp);
Eigen::Vector2d skewDistortionD(Eigen::Vector2d pos, Eigen::Vector2d B12, double f, Eigen::Vector2d pp);

Eigen::Vector2f inverseSkewDistortion(Eigen::Vector2f pos,
									  Eigen::Vector2f B12,
									  float f,
									  Eigen::Vector2f pp);

} // namespace Geometry
} // namespace StereoVision

#endif // STEREOVISIONAPP_LENSDISTORTION_H
