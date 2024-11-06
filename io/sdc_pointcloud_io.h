#ifndef STEREOVISION_IO_SDCPOINTCLOUD_H
#define STEREOVISION_IO_SDCPOINTCLOUD_H

#include "pointcloud_io.h"
#include <filesystem>

namespace StereoVision
{
namespace IO
{

class SdcPointCloudPoint : public PointCloudPointAccessInterface
{
public:
    SdcPointCloudPoint(std::unique_ptr<std::ifstream>&& reader, uint16_t majorVersion, uint16_t minorVersion);
private:
    double time;
    float range;
    float theta;
    float x;
    float y;
    float z;
    uint16_t amplitude;
    uint16_t width;
    uint8_t targettype;
    uint8_t target;
    uint8_t numtarget;
    uint16_t rgindex;
    uint8_t channeldesc;
    
    // from version 5.2
    uint8_t classid;
    // from version 5.3
    float rho;
    // from version 5.4
    int16_t reflectance;

    // version informations
    const uint16_t majorVersion;
    const uint16_t minorVersion;

    std::vector<std::string> attributeNames;
    // reader "head"
    const std::unique_ptr<std::ifstream> reader;

    size_t recordByteSize; // number of bytes in a sdc point record

public:
    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    bool gotoNext() override;
    
    // destructor
    ~SdcPointCloudPoint() override;
};

class SdcPointCloudHeader : public PointCloudHeaderInterface
{
public:
    const uint32_t headerSize;
    const uint16_t majorVersion;
    const uint16_t minorVersion;
    const std::string headerInformation;

private:
    // attribute names order should appear in the same order than the corresponding id of the attributes
    const std::vector<std::string> attributeNames = {"headerSize", "majorVersion", "minorVersion", "headerInformation"};

public:
    // constructor
    SdcPointCloudHeader(const uint32_t headerSize, const uint16_t majorVersion, const uint16_t minorVersion, const std::string& headerInformation);

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;
};


/**
 * @brief Open a point cloud from a sdc file
 *
 * Open a point cloud from a sdc file and returns a FullPointCloudAccessInterface
 * containing the header and the points.
 *
 * @param sdcFilePath The path to the sdc file containing the point cloud
 *
 * @return A FullPointCloudAccessInterface containing the header and the points.
 *         If the file can't be opened, an empty optional is returned
 */
std::optional<FullPointCloudAccessInterface> openPointCloudSdc(const std::filesystem::path& sdcFilePath);

} // namespace IO
} // namespace StereoVision


#endif //STEREOVISION_IO_SDCPOINTCLOUD_H