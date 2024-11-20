#include <cstdint>
#include <optional>
#include <iostream>
#include <fstream>
#include <numeric>
#include "sdc_pointcloud_io.h"
#include "pointcloud_io.h"

namespace StereoVision {
namespace IO {
SdcPointCloudPoint::SdcPointCloudPoint(std::unique_ptr<std::istream> reader, uint16_t majorVersion, uint16_t minorVersion):
    majorVersion{majorVersion}, minorVersion{minorVersion}, reader{std::move(reader)}
{
    // compute the offsets
    std::exclusive_scan(fieldByteSize.begin(), fieldByteSize.end(), fieldOffset.begin(), 0);

    // compute the size of a sdc point record by summing the field sizes
    recordByteSize = std::reduce(fieldByteSize.begin(), fieldByteSize.end() - 3);
    
    // depending on the version, we have different attributes for the point cloud
    attributeNames = {"time", "range", "theta", "x", "y", "z", "amplitude", "width", "targettype", "target", "numtarget", "rgindex", "channeldesc"};
    if (majorVersion >= 5) {
        if (minorVersion >= 2) { // version 5.2
            attributeNames.push_back("classid");
            recordByteSize += sizeof(classid);
        }
        if (minorVersion >= 3) { // version 5.3
            attributeNames.push_back("rho");
            recordByteSize += sizeof(rho);
        }
        if (minorVersion >= 4) { // version 5.4
            attributeNames.push_back("reflectance");
            recordByteSize += sizeof(reflectance);
        }
    }
}

PtGeometry<PointCloudGenericAttribute> SdcPointCloudPoint::getPointPosition() const {
    return PtGeometry<PointCloudGenericAttribute>{x, y, z};
}

std::optional<PtColor<PointCloudGenericAttribute>> SdcPointCloudPoint::getPointColor() const {
    return std::nullopt; // Assuming color is not applicable
}

std::optional<PointCloudGenericAttribute> SdcPointCloudPoint::getAttributeById(int id) const {
    switch (id) {
        case 0:
            return PointCloudGenericAttribute{time};
        case 1:
            return PointCloudGenericAttribute{range};
        case 2:
            return PointCloudGenericAttribute{theta};
        case 3:
            return PointCloudGenericAttribute{x};
        case 4:
            return PointCloudGenericAttribute{y};
        case 5:
            return PointCloudGenericAttribute{z};
        case 6:
            return PointCloudGenericAttribute{amplitude};
        case 7:
            return PointCloudGenericAttribute{width};
        case 8:
            return PointCloudGenericAttribute{targettype};
        case 9:
            return PointCloudGenericAttribute{target};
        case 10:
            return PointCloudGenericAttribute{numtarget};
        case 11:
            return PointCloudGenericAttribute{rgindex};
        case 12:
            return PointCloudGenericAttribute{channeldesc};   
    }

    if (majorVersion >= 5) { 
        if (minorVersion >= 2 && id == 13) { // version 5.2
            return PointCloudGenericAttribute{classid};
        } else if (minorVersion >= 3 && id == 14) { // version 5.3
            return PointCloudGenericAttribute{rho};
        } else if (minorVersion >= 4 && id == 15) { // version 5.4
            return PointCloudGenericAttribute{reflectance};
        }
    }

    return std::nullopt; // Attribute not found
}

std::optional<PointCloudGenericAttribute> SdcPointCloudPoint::getAttributeByName(const char* attributeName) const {
    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    if (it != attributeNames.end()) {
        return getAttributeById(std::distance(attributeNames.begin(), it));
    }
    return std::nullopt;
}

std::vector<std::string> SdcPointCloudPoint::attributeList() const {
    return attributeNames;
}

bool SdcPointCloudPoint::gotoNext() {
    static_assert(sizeof(float) == 4); // check if float is 4 bytes, should be true on most systems
    static_assert(sizeof(double) == 8); // check if double is 8 bytes
    // try to read a record
     std::vector<char> buffer(recordByteSize);
    reader->read(buffer.data(), recordByteSize);
    if (!reader->good()) {
        return false; // end of file or read error
    }
    // read the data
    time = *reinterpret_cast<double*>(buffer.data() + fieldOffset[0]);
    range = *reinterpret_cast<float*>(buffer.data() + fieldOffset[1]);
    theta = *reinterpret_cast<float*>(buffer.data() + fieldOffset[2]);
    x = *reinterpret_cast<float*>(buffer.data() + fieldOffset[3]);
    y = *reinterpret_cast<float*>(buffer.data() + fieldOffset[4]);
    z = *reinterpret_cast<float*>(buffer.data() + fieldOffset[5]);
    amplitude = *reinterpret_cast<uint16_t*>(buffer.data() + fieldOffset[6]);
    width = *reinterpret_cast<uint16_t*>(buffer.data() + fieldOffset[7]);
    targettype = *reinterpret_cast<uint8_t*>(buffer.data() + fieldOffset[8]);
    target = *reinterpret_cast<uint8_t*>(buffer.data() + fieldOffset[9]);
    numtarget = *reinterpret_cast<uint8_t*>(buffer.data() + fieldOffset[10]);
    rgindex = *reinterpret_cast<uint16_t*>(buffer.data() + fieldOffset[11]);
    channeldesc = *reinterpret_cast<uint8_t*>(buffer.data() + fieldOffset[12]);
    // depending on the version, we have different attributes for the point cloud
    if (majorVersion >= 5) {
        if (minorVersion >= 2) { // version 5.2
            classid = *reinterpret_cast<uint8_t*>(buffer.data() + fieldOffset[13]);
        }
        if (minorVersion >= 3) { // version 5.3
            rho = *reinterpret_cast<float*>(buffer.data() + fieldOffset[14]);
        }
        if (minorVersion >= 4) { // version 5.4
            reflectance = *reinterpret_cast<int16_t*>(buffer.data() + fieldOffset[15]);
        }
    }
    return true;
}

SdcPointCloudPoint::~SdcPointCloudPoint()
{
}

SdcPointCloudHeader::SdcPointCloudHeader(const uint32_t headerSize, const uint16_t majorVersion, const uint16_t minorVersion, const std::string& headerInformation)
: headerSize{headerSize}, majorVersion{majorVersion}, minorVersion{minorVersion}, headerInformation{headerInformation}
{}

std::optional<PointCloudGenericAttribute> SdcPointCloudHeader::getAttributeById(int id) const
{
    switch (id) {
        case 0:
            return PointCloudGenericAttribute{headerSize};
        case 1:
            return PointCloudGenericAttribute{majorVersion};
        case 2:
            return PointCloudGenericAttribute{minorVersion};
        case 3:
            return PointCloudGenericAttribute{headerInformation};
        default:
            return std::nullopt; // Attribute not found
    }
}

std::optional<PointCloudGenericAttribute> SdcPointCloudHeader::getAttributeByName(const char *attributeName) const
{
    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    if (it != attributeNames.end()) {
        return getAttributeById(std::distance(attributeNames.begin(), it));
    }
    return std::nullopt; // Attribute not found
}

std::vector<std::string> SdcPointCloudHeader::attributeList() const
{
    return attributeNames;
}

std::optional<FullPointCloudAccessInterface> openPointCloudSdc(const std::filesystem::path &sdcFilePath)
{
    // read the file
    auto inputFile = std::make_unique<std::ifstream>(sdcFilePath, std::ios_base::binary);

    // return null if the file can't be opened
    if (!inputFile->is_open()) return std::nullopt;

    // auto fileSize = std::filesystem::file_size(sdcFilePath);

    // first 4 bytes are the size of the header
    uint32_t headerSize;
    inputFile->read(reinterpret_cast<char*>(&headerSize), 4);
    if (!inputFile->good()) return std::nullopt;

    // read the rest of the header
     std::vector<char> bufferHeader(headerSize - 4);
    inputFile->read(bufferHeader.data(), headerSize - 4);
    if (!inputFile->good()) return std::nullopt;

    // next 2 bytes are the major version
    auto majorVersion = *reinterpret_cast<uint16_t*>(bufferHeader.data());

    // next 2 bytes are the minor version
    auto minorVersion = *reinterpret_cast<uint16_t*>(bufferHeader.data() + 2);

    // next headerSize - 8 bytes are the header informations
    std::string headerInformation{bufferHeader.data() + 4, headerSize - 4};

    auto header = std::make_unique<SdcPointCloudHeader>(headerSize, majorVersion, minorVersion, headerInformation);
    auto cloudpoint = std::make_unique<SdcPointCloudPoint>(std::move(inputFile), majorVersion, minorVersion);
    
    FullPointCloudAccessInterface fullPointInterface;
    // read the first point
    if (cloudpoint->gotoNext()) {
        fullPointInterface.headerAccess = std::move(header);
        fullPointInterface.pointAccess = std::move(cloudpoint);
        return fullPointInterface;
    }
    return std::nullopt;
}

} // namespace IO
} // namespace StereoVision