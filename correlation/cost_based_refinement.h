#ifndef COST_BASED_REFINEMENT_H
#define COST_BASED_REFINEMENT_H

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
