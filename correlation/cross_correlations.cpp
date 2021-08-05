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

#include "cross_correlations.h"

namespace StereoVision {
namespace Correlation {

const std::string MatchingFunctionTraits<matchingFunctions::CC>::Name = "CC";
const std::string MatchingFunctionTraits<matchingFunctions::NCC>::Name = "NCC";
const std::string MatchingFunctionTraits<matchingFunctions::SSD>::Name = "SSD";
const std::string MatchingFunctionTraits<matchingFunctions::SAD>::Name = "SAD";
const std::string MatchingFunctionTraits<matchingFunctions::ZCC>::Name = "ZCC";
const std::string MatchingFunctionTraits<matchingFunctions::ZNCC>::Name = "ZNCC";
const std::string MatchingFunctionTraits<matchingFunctions::ZSSD>::Name = "ZSSD";
const std::string MatchingFunctionTraits<matchingFunctions::ZSAD>::Name = "ZSAD";

} //namespace Correlation
} //namespace StereoVision
