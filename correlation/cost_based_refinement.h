#ifndef COST_BASED_REFINEMENT_H
#define COST_BASED_REFINEMENT_H

#include "./correlation_base.h"

namespace StereoVision {
namespace Correlation {

Multidim::Array<float, 2> refineDispEquiangularCostInterpolation(Multidim::Array<float, 3> const& truncatedCostVolume,
																 Multidim::Array<disp_t, 2> const& rawDisparity);

Multidim::Array<float, 2> refineDispParabolaCostInterpolation(Multidim::Array<float, 3> const& truncatedCostVolume,
															  Multidim::Array<disp_t, 2> const& rawDisparity);

} //namespace Correlation
} //namespace StereoVision

#endif // COST_BASED_REFINEMENT_H
