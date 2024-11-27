#include "pointcloud_io.h"

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024 Paragon<french.paragon@gmail.com>

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

namespace StereoVision {
namespace IO {

FullPointCloudAccessInterface::FullPointCloudAccessInterface(PointCloudHeaderInterface* header, PointCloudPointAccessInterface* points) :
    headerAccess(header),
    pointAccess(points)
{

}

FullPointCloudAccessInterface::FullPointCloudAccessInterface(FullPointCloudAccessInterface && other) :
    headerAccess(std::move(other.headerAccess)),
    pointAccess(std::move(other.pointAccess))
{
    other.headerAccess = nullptr;
    other.pointAccess = nullptr;
}

} // namespace IO
} // namespace StereoVision
