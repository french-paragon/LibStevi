#ifndef STEREOVISIONAPP_LENSDISTORTIONSMAP_H
#define STEREOVISIONAPP_LENSDISTORTIONSMAP_H

#include "stevi_global.h"

#include <eigen3/Eigen/Core>

namespace StereoVision {
namespace Interpolation {

ImageArray computeLensDistortionMap(int height,
									int width,
									float f,
									Eigen::Vector2f pp,
									Eigen::Vector3f k123,
									Eigen::Vector2f t12,
									Eigen::Vector2f B12);

} // namespace Interpolation
} // namespace StereoVisionApp

#endif // STEREOVISIONAPP_LENSDISTORTIONSMAP_H
