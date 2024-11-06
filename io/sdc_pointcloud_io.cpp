#include <cstdint>
#include <optional>
#include <iostream>
#include <fstream>
#include "sdc_pointcloud_io.h"
#include "pointcloud_io.h"

namespace StereoVision {
namespace IO {
SdcPointCloudPoint::SdcPointCloudPoint(std::unique_ptr<std::ifstream> &&reader, uint16_t majorVersion, uint16_t minorVersion):
    majorVersion{majorVersion}, minorVersion{minorVersion}, reader{std::move(reader)}
{
    // compute the size of a sdc point record
    recordByteSize = sizeof(time) + sizeof(range) + sizeof(theta) + sizeof(x) + sizeof(y) + sizeof(z) + sizeof(amplitude) + sizeof(width) + sizeof(targettype) + sizeof(target) + sizeof(numtarget) + sizeof(rgindex) + sizeof(channeldesc);
    
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
    // read the first cloudPoint
    gotoNext();
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
    // try to read a record
     std::vector<char> buffer(recordByteSize);
    reader->read(buffer.data(), recordByteSize);
    if (!reader->good()) {
        return false; // end of file or read error
    }
    // read the data
    size_t offset = 0; // offset in bytes
    time = *reinterpret_cast<double*>(buffer.data() + offset);
    offset += sizeof(time);
    range = *reinterpret_cast<float*>(buffer.data() + offset);
    offset += sizeof(range);
    theta = *reinterpret_cast<float*>(buffer.data() + offset);
    offset += sizeof(theta);
    x = *reinterpret_cast<float*>(buffer.data() + offset);
    offset += sizeof(x);
    y = *reinterpret_cast<float*>(buffer.data() + offset);
    offset += sizeof(y);
    z = *reinterpret_cast<float*>(buffer.data() + offset);
    offset += sizeof(z);
    amplitude = *reinterpret_cast<uint16_t*>(buffer.data() + offset);
    offset += sizeof(amplitude);
    width = *reinterpret_cast<uint16_t*>(buffer.data() + offset);
    offset += sizeof(width);
    targettype = *reinterpret_cast<uint8_t*>(buffer.data() + offset);
    offset += sizeof(targettype);
    target = *reinterpret_cast<uint8_t*>(buffer.data() + offset);
    offset += sizeof(target);
    numtarget = *reinterpret_cast<uint8_t*>(buffer.data() + offset);
    offset += sizeof(numtarget);
    rgindex = *reinterpret_cast<uint16_t*>(buffer.data() + offset);
    offset += sizeof(rgindex);
    channeldesc = *reinterpret_cast<uint8_t*>(buffer.data() + offset);
    offset += sizeof(channeldesc);
    if (majorVersion >= 5) {
        if (minorVersion >= 2) { // version 5.2
            classid = *reinterpret_cast<uint8_t*>(buffer.data() + offset);
            offset += sizeof(classid);
        }
        if (minorVersion >= 3) { // version 5.3
            rho = *reinterpret_cast<float*>(buffer.data() + offset);
            offset += sizeof(rho);
        }
        if (minorVersion >= 4) { // version 5.4
            reflectance = *reinterpret_cast<int16_t*>(buffer.data() + offset);
            offset += sizeof(reflectance);
        }
    }
    return true;
}

SdcPointCloudPoint::~SdcPointCloudPoint()
{
    // close the file
    reader->close();
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

    PointCloudHeaderInterface* header = new SdcPointCloudHeader(headerSize, majorVersion, minorVersion, headerInformation);
    PointCloudPointAccessInterface* cloudpoint = new SdcPointCloudPoint(std::move(inputFile), majorVersion, minorVersion);

    return FullPointCloudAccessInterface(header, cloudpoint);
}

} // namespace IO
} // namespace StereoVision