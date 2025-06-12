#ifndef STEREOVISION_IO_REMOVEATTR_POINTCLOUD_H
#define STEREOVISION_IO_REMOVEATTR_POINTCLOUD_H

#include "pointcloud_io.h"

namespace StereoVision {
namespace IO {

/**
 * @brief Adapter class to remove attribute names or color from a point cloud point access interface.
 * 
 */
class PointCloudPointAttributeRemoverInterface : public PointCloudPointAccessInterface {
public:
    /**
     * @brief Remove attributes or color from the points of a point cloud
     * 
     * @param pointCloudPointAccessInterface The point cloud point access interface. If an error occurs, returns nullptr
     * and does not move the interface. Otherwise, moves the interface.
     * @param attributesToRemove The attributes to remove
     * @param removePointColor Remove the color data (does not modify the attributes)
     * @param removeAllAttributes Remove all attributes
     * @return std::unique_ptr<PointCloudPointAttributeRemoverInterface>. If an error occurs, returns nullptr
     */
    static std::unique_ptr<PointCloudPointAttributeRemoverInterface> create(
        std::unique_ptr<PointCloudPointAccessInterface>& pointCloudPointAccessInterface,
        std::vector<std::string> attributesToRemove, std::optional<bool> removePointColor = std::nullopt,
        std::optional<bool> removeAllAttributes = std::nullopt);

protected:
    virtual bool doesRemovePointColor() const = 0;
    virtual bool doesRemoveAllAttributes() const = 0;
};

/**
 * @brief Remove attributes or color from the points of a point cloud.
 * 
 * @param fullAccessInterface The interface to access the full point cloud. If an error occurs, returns nullptr and
 *  does not move the interface. Otherwise, moves the interface.
 * @param attributesToRemove The attributes to remove from the points
 * @param removePointColor If true, remove the color data (does not modify the attributes)
 * @param removeAllAttributes Remove all attributes
 * @return std::unique_ptr<FullPointCloudAccessInterface>. If an error occurs, returns nullptr
 */
std::unique_ptr<FullPointCloudAccessInterface> RemoveAttributesOrColorFromPointCloud(
    std::unique_ptr<FullPointCloudAccessInterface>& fullAccessInterface, std::vector<std::string> attributesToRemove,
    std::optional<bool> removePointColor = std::nullopt,  std::optional<bool> removeAllAttributes = std::nullopt);

} // StereoVision
} // IO

#endif // STEREOVISION_IO_REMOVEATTR_POINTCLOUD_H