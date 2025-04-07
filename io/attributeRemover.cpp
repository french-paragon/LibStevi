#include "attributeRemover.h"
#include <iostream>

namespace StereoVision {
namespace IO {

template <bool removePointColor = false, bool removeAllAttributes = false>
class PointCloudPointAttributeRemover : public PointCloudPointAttributeRemoverInterface {
private:
    std::unique_ptr<PointCloudPointAccessInterface> accessInterface = nullptr;
    std::vector<std::string> attributeNames;
public:
    /**
     * @brief adapter class to remove attribute names from a point cloud point access interface
     * 
     * @param pointCloudPointAccessInterface The point cloud point access interface
     * @param attributesToRemove The attributes to remove
     * @param currentAttributeNames The current attribute names to use if any. If not given, use the ones in the interface
     */
    PointCloudPointAttributeRemover(
        std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterface,
            std::vector<std::string> attributesToRemove,
            const std::optional<std::vector<std::string>>& currentAttributeNames) :
                accessInterface{std::move(pointCloudPointAccessInterface)} {
        
        if constexpr (!removeAllAttributes) {
            if (accessInterface != nullptr) {
                if (currentAttributeNames.has_value()) {
                    attributeNames = currentAttributeNames.value();
                } else {
                    attributeNames = accessInterface->attributeList();
                }
                for (const auto& attributeName : attributesToRemove) {
                    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
                    if (it != attributeNames.end()) {
                        attributeNames.erase(it);
                    }
                }
            }   
        }
    }

    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override {
        return accessInterface->getPointPosition();
    }

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override {
        if constexpr (removePointColor) {
            return std::nullopt;
        } else {
            return accessInterface->getPointColor();
        }
    }

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override {
        if constexpr (removeAllAttributes) {
            return std::nullopt;
        } else {
            if (id < 0 || id >= attributeNames.size()) return std::nullopt;
            return accessInterface->getAttributeByName(attributeNames[id].c_str());
        }
    }

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override {
        if constexpr (removeAllAttributes) {
            return std::nullopt;
        } else {
            // find the attribute name in the list
            auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
            if (it != attributeNames.end()) {
                return accessInterface->getAttributeByName(attributeName);
            }
            return std::nullopt;
        }
    }

    std::vector<std::string> attributeList() const override {
        if constexpr (removeAllAttributes) {
            return {};
        } else {
            return attributeNames;
        }
    }

    bool gotoNext() override {
        return accessInterface->gotoNext();
    }

    virtual bool hasData() const override {
        return accessInterface->hasData();
    }

protected:
    virtual bool doesRemovePointColor() const override {
        return removePointColor;
    }
    virtual bool doesRemoveAllAttributes() const override {
        return removeAllAttributes;
    }
};

std::unique_ptr<PointCloudPointAttributeRemoverInterface> PointCloudPointAttributeRemoverInterface::create(
    std::unique_ptr<PointCloudPointAccessInterface>& pointCloudPointAccessInterface,
    std::vector<std::string> attributesToRemove, std::optional<bool> removePointColorParam,
    std::optional<bool> removeAllAttributesParam) {
    
    bool removePointColor = removePointColorParam.value_or(false);
    bool removeAllAttributes = removeAllAttributesParam.value_or(false);

    std::optional<std::vector<std::string>> currentAttributeNames = std::nullopt;

    if (pointCloudPointAccessInterface == nullptr) return nullptr;

    // test if the previous interface is already a PointCloudPointAttributeRemoverInterface. If yes, we can take it
    auto castedToRemover = dynamic_cast<PointCloudPointAttributeRemoverInterface*>(pointCloudPointAccessInterface.get());
    if (castedToRemover != nullptr) {
        if (!removePointColorParam.has_value())
            removePointColor = castedToRemover->doesRemovePointColor();
        if (!removeAllAttributesParam.has_value())
            removeAllAttributes = castedToRemover->doesRemoveAllAttributes();
        currentAttributeNames = castedToRemover->attributeList();
    }

    std::unique_ptr<PointCloudPointAttributeRemoverInterface> pointCloudPointAccessInterfaceRemover;

    if (removePointColor && removeAllAttributes) {
        pointCloudPointAccessInterfaceRemover = std::make_unique<PointCloudPointAttributeRemover<true, true>>(
            std::move(pointCloudPointAccessInterface), attributesToRemove, currentAttributeNames);
    } else if (removePointColor && !removeAllAttributes) {
        pointCloudPointAccessInterfaceRemover =  std::make_unique<PointCloudPointAttributeRemover<true, false>>(
            std::move(pointCloudPointAccessInterface), attributesToRemove, currentAttributeNames);
    } else if (!removePointColor && removeAllAttributes) {
        pointCloudPointAccessInterfaceRemover =  std::make_unique<PointCloudPointAttributeRemover<false, true>>(
            std::move(pointCloudPointAccessInterface), attributesToRemove, currentAttributeNames);
    } else {
        pointCloudPointAccessInterfaceRemover =  std::make_unique<PointCloudPointAttributeRemover<false, false>>(
            std::move(pointCloudPointAccessInterface), attributesToRemove, currentAttributeNames);
    }

    return std::move(pointCloudPointAccessInterfaceRemover);
}

std::unique_ptr<FullPointCloudAccessInterface> RemoveAttributesOrColorFromPointCloud(
    std::unique_ptr<FullPointCloudAccessInterface> &fullAccessInterface, std::vector<std::string> attributesToRemove,
    std::optional<bool> removePointColor, std::optional<bool> removeAllAttributes) {

    if (fullAccessInterface == nullptr || fullAccessInterface->pointAccess == nullptr) return nullptr;

    auto pointCloudPointAccessInterfaceRemover = PointCloudPointAttributeRemoverInterface::create(
        fullAccessInterface->pointAccess, attributesToRemove, removePointColor, removeAllAttributes);
    
    if (pointCloudPointAccessInterfaceRemover == nullptr) {
        return nullptr;
    }

    fullAccessInterface->pointAccess = std::move(pointCloudPointAccessInterfaceRemover);

    return std::move(fullAccessInterface);
}

} // StereoVision
} // IO
