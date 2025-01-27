#ifndef STEREOVISION_IO_MAPPOINTCLOUD_H
#define STEREOVISION_IO_MAPPOINTCLOUD_H

#include "pointcloud_io.h"

namespace StereoVision {
namespace IO {

// same implementation for both PointCloudPoint and PointCloudHeader
template<class AccessInterfaceType>
class AttributeMapperImplementation {
private:
    std::unique_ptr<AccessInterfaceType> accessInterface = nullptr;
    std::vector<std::string> attributeNames;
    std::vector<std::string> originalAttributeNames;
public:
    /**
     * @brief adapter class to map attribute names to other attribute names
     * 
     * @param accessInterface The point cloud point access interface
     * @param attributeMap Maps the original attribute names to a new attribute names
     * @param onlyKeepAttributesInMap If true, only keep attributes in the map and the attributes not in the map will be
     *  ignored. Otherwise, the attributes not in the map but present in the original point cloud will be kept.
     */
    AttributeMapperImplementation(
        std::unique_ptr<AccessInterfaceType> accessInterface, std::map<std::string, std::string> attributeMap,
        bool onlyKeepAttributesInMap = false);

    std::vector<std::string> attributeList() const { return attributeNames; }

   AccessInterfaceType& getAccessInterface() const { return *accessInterface; }

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const;

};

class PointCloudPointAttributeMapper : public PointCloudPointAccessInterface {
private:
    AttributeMapperImplementation<PointCloudPointAccessInterface> mapperImplementation;
public:
    /**
     * @brief adapter class to map attribute names to other attribute names
     * 
     * @param pointCloudPointAccessInterface The point cloud point access interface
     * @param attributeMap Maps the original attribute names to a new attribute names
     * @param onlyKeepAttributesInMap If true, only keep attributes in the map and the attributes not in the map will be
     *  ignored. Otherwise, the attributes not in the map but present in the original point cloud will be kept.
     */
    PointCloudPointAttributeMapper(
        std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterface, std::map<std::string, std::string> attributeMap,
        bool onlyKeepAttributesInMap = false) :
            mapperImplementation(std::move(pointCloudPointAccessInterface), attributeMap, onlyKeepAttributesInMap) {}

    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override {
        return mapperImplementation.getAccessInterface().getPointPosition();
    }

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override {
        return mapperImplementation.getAccessInterface().getPointColor();
    }

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override {
        return mapperImplementation.getAttributeById(id);
    }

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override {
        return mapperImplementation.getAttributeByName(attributeName);
    }

    std::vector<std::string> attributeList() const override {
        return mapperImplementation.attributeList();
    }

    bool gotoNext() override {
        return mapperImplementation.getAccessInterface().gotoNext();
    }
};

class PointCloudHeaderAttributeMapper : public PointCloudHeaderInterface {
private:
    AttributeMapperImplementation<PointCloudHeaderInterface> mapperImplementation;
public:
    /**
     * @brief adapter class to map attribute names to other attribute names
     * 
     * @param pointCloudHeaderInterface The point cloud header interface
     * @param attributeMap Maps the original attribute names to a new attribute names
     * @param onlyKeepAttributesInMap If true, only keep attributes in the map and the attributes not in the map will be
     *  ignored. Otherwise, the attributes not in the map but present in the original point cloud will be kept.
     */
    PointCloudHeaderAttributeMapper(
        std::unique_ptr<PointCloudHeaderInterface> pointCloudHeaderInterface, std::map<std::string, std::string> attributeMap,
        bool onlyKeepAttributesInMap = false) :
            mapperImplementation(std::move(pointCloudHeaderInterface), attributeMap, onlyKeepAttributesInMap) {}

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override {
        return mapperImplementation.getAttributeById(id);
    }

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override {
        return mapperImplementation.getAttributeByName(attributeName);
    }

    std::vector<std::string> attributeList() const override {
        return mapperImplementation.attributeList();
    }
};

template <class AccessInterfaceType>
AttributeMapperImplementation<AccessInterfaceType>::AttributeMapperImplementation(
    std::unique_ptr<AccessInterfaceType> accessInterface_, std::map<std::string, std::string> attributeMap,
    bool onlyKeepAttributesInMap) {

    accessInterface = std::move(accessInterface_);
    if (accessInterface != nullptr) {
        auto FullOriginalAttributeNames = accessInterface->attributeList();
        for (auto &attributeName : FullOriginalAttributeNames) {
            bool shouldAddPair = false;
            std::string originalName = attributeName;
            std::string newName = attributeName;
            if (auto pair = attributeMap.find(attributeName); pair != attributeMap.end()) {
                newName = pair->second;
                shouldAddPair = true;
            } else if (!onlyKeepAttributesInMap) {
                shouldAddPair = true;
            }

            // only add the pair if the new name is not already in the list
            auto it = std::find(attributeNames.begin(), attributeNames.end(), newName);
            if (it != attributeNames.end()) shouldAddPair = false;

            // add the pair
            if (shouldAddPair) {
                attributeNames.push_back(newName);
                originalAttributeNames.push_back(originalName);
            }
        }
    }
}

template <class AccessInterfaceType>
std::optional<PointCloudGenericAttribute> AttributeMapperImplementation<AccessInterfaceType>::getAttributeById(int id) const {

    if (id < 0 || id >= attributeNames.size()) return std::nullopt;
    return accessInterface->getAttributeByName(originalAttributeNames[id].c_str());
}

template <class AccessInterfaceType>
std::optional<PointCloudGenericAttribute> AttributeMapperImplementation<AccessInterfaceType>::getAttributeByName(
    const char *attributeName) const {

    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    if (it != attributeNames.end()) {
        return getAttributeById(std::distance(attributeNames.begin(), it));
    }
    return std::nullopt;
}

/**
 * @brief Map the attributes of a point cloud to other attributes
 * 
 * @param accessInterface The point cloud access interface
 * @param attributeMapPoints Maps the original attribute names to a new attribute names for points
 * @param onlyKeepAttributesInMapPoints If true, only keep attributes in the map and the attributes not in the map will be
 *  ignored. Otherwise, the attributes not in the map but present in the original point cloud will be kept.
 * @param attributeMapHeader Maps the original attribute names to a new attribute names for header
 * @param onlyKeepAttributesInMapHeader If true, only keep attributes in the map and the attributes not in the map will be
 *  ignored. Otherwise, the attributes not in the map but present in the original point cloud will be kept.
 * @return FullPointCloudInterface The full point cloud interface with the mapped attributes
 */
 FullPointCloudAccessInterface mapPointCloudAttributes(
    FullPointCloudAccessInterface&& fullAccessInterface_, std::map<std::string, std::string> attributeMapPoints,
    bool onlyKeepAttributesInMapPoints = false, std::map<std::string, std::string> attributeMapHeader = {},
    bool onlyKeepAttributesInMapHeader = false) {

    auto points = std::move(fullAccessInterface_.pointAccess);
    auto header = std::move(fullAccessInterface_.headerAccess);

    FullPointCloudAccessInterface fullAccessInterface;

    // map the points
    std::unique_ptr<PointCloudPointAccessInterface> pointMapper;
    std::unique_ptr<PointCloudHeaderInterface> headerMapper;

    if (!attributeMapPoints.empty() || onlyKeepAttributesInMapPoints) {
        pointMapper = std::make_unique<PointCloudPointAttributeMapper>(std::move(points), attributeMapPoints, onlyKeepAttributesInMapPoints);
    } else {
        pointMapper = std::move(points);
    }

    if (!attributeMapHeader.empty() || onlyKeepAttributesInMapHeader) {
        headerMapper = std::make_unique<PointCloudHeaderAttributeMapper>(std::move(header), attributeMapHeader, onlyKeepAttributesInMapHeader);
    } else {
        headerMapper = std::move(header);
    }
    
    fullAccessInterface.pointAccess = std::move(pointMapper);
    fullAccessInterface.headerAccess = std::move(headerMapper);

    return std::move(fullAccessInterface);
}

} // StereoVision
} // IO

#endif // STEREOVISION_IO_MAPPOINTCLOUD_H