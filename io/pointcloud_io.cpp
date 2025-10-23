#include "pointcloud_io.h"
#include "sdc_pointcloud_io.h"
#include "pcd_pointcloud_io.h"
#include "las_pointcloud_io.h"
#include "metacloud_io.h"

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
    pointAccess(points) {

}

FullPointCloudAccessInterface::FullPointCloudAccessInterface(FullPointCloudAccessInterface && other) :
    headerAccess(std::move(other.headerAccess)),
    pointAccess(std::move(other.pointAccess)) {
    other.headerAccess = nullptr;
    other.pointAccess = nullptr;
}

StatusOptional<FullPointCloudAccessInterface> openPointCloud(const std::filesystem::path &filePath) {
    // get the file extension of the file
    auto fileExtension = filePath.extension();
    if (fileExtension == ".sdc" || fileExtension == ".SDC") {
        return openPointCloudSdc(filePath);
    } else if (fileExtension == ".pcd" || fileExtension == ".PCD") {
        return openPointCloudPcd(filePath);
    } else if (fileExtension == ".las" || fileExtension == ".LAS") {
        return openPointCloudLas(filePath);
    } else if (fileExtension == ".metacloud" || fileExtension == ".METACLOUD") {
        return openPointCloudMetacloud(filePath);
    } else {
        return StatusOptional<FullPointCloudAccessInterface>::error("Unrecognized file type: \"" + fileExtension.native() + "\"");
    }
}

int64_t PointCloudHeaderInterface::expectedNumberOfPoints() const {
    return -1;
}

int64_t PointCloudPointAccessInterface::expectedNumberOfPoints() const {
    return -1;
}
int64_t PointCloudPointAccessInterface::processedNumberOfPoints() const {
    return -1;
}

} // namespace IO
} // namespace StereoVision
