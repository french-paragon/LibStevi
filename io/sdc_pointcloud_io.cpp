#include <cstdint>
#include <optional>
#include <iostream>
#include <fstream>
#include <numeric>
#include "sdc_pointcloud_io.h"
#include "pointcloud_io.h"
#include "fstreamCustomBuffer.h"
#include "bit_manipulations.h"

namespace StereoVision {
namespace IO {

// constants
// buffersize when reading a sdc file
constexpr static size_t sdcFileReaderBufferSize = 1 << 16;

SdcPointCloudPoint::SdcPointCloudPoint(std::unique_ptr<std::istream> reader, uint16_t majorVersion, uint16_t minorVersion):
    majorVersion{majorVersion},
    minorVersion{minorVersion},
    nbAttributes{computeNbAttributes(majorVersion, minorVersion)},
    recordByteSize{computeRecordByteSize(majorVersion, minorVersion)},
    reader{std::move(reader)}
{ }

PtGeometry<PointCloudGenericAttribute> SdcPointCloudPoint::getPointPosition() const {
    return PtGeometry<PointCloudGenericAttribute>{
            fromBytes<float>(dataBufferPtr + fieldOffset[x_id]),
            fromBytes<float>(dataBufferPtr + fieldOffset[y_id]),
            fromBytes<float>(dataBufferPtr + fieldOffset[z_id])
        };
}

std::optional<PtColor<PointCloudGenericAttribute>> SdcPointCloudPoint::getPointColor() const {
    return std::nullopt; // Assuming color is not applicable
}

std::optional<PointCloudGenericAttribute> SdcPointCloudPoint::getAttributeById(int id) const {
    if (id >= nbAttributes || id < 0) return std::nullopt; 
    switch (id) {
        case time_id:
            return PointCloudGenericAttribute{fromBytes<double>(dataBufferPtr + fieldOffset[time_id])};
        case range_id:
            return PointCloudGenericAttribute{fromBytes<float>(dataBufferPtr + fieldOffset[range_id])};
        case theta_id:
            return PointCloudGenericAttribute{fromBytes<float>(dataBufferPtr + fieldOffset[theta_id])};
        case x_id:
            return PointCloudGenericAttribute{fromBytes<float>(dataBufferPtr + fieldOffset[x_id])};
        case y_id:
            return PointCloudGenericAttribute{fromBytes<float>(dataBufferPtr + fieldOffset[y_id])};
        case z_id:
            return PointCloudGenericAttribute{fromBytes<float>(dataBufferPtr + fieldOffset[z_id])};
        case amplitude_id:
            return PointCloudGenericAttribute{fromBytes<uint16_t>(dataBufferPtr + fieldOffset[amplitude_id])};
        case width_id:
            return PointCloudGenericAttribute{fromBytes<uint16_t>(dataBufferPtr + fieldOffset[width_id])};
        case targettype_id:
            return PointCloudGenericAttribute{fromBytes<uint8_t>(dataBufferPtr + fieldOffset[targettype_id])};
        case target_id:
            return PointCloudGenericAttribute{fromBytes<uint8_t>(dataBufferPtr + fieldOffset[target_id])};
        case numtarget_id:
            return PointCloudGenericAttribute{fromBytes<uint8_t>(dataBufferPtr + fieldOffset[numtarget_id])};
        case rgindex_id:
            return PointCloudGenericAttribute{fromBytes<uint16_t>(dataBufferPtr + fieldOffset[rgindex_id])};
        case channeldesc_id:
            return PointCloudGenericAttribute{fromBytes<uint8_t>(dataBufferPtr + fieldOffset[channeldesc_id])};
        case classid_id:
            return PointCloudGenericAttribute{fromBytes<uint8_t>(dataBufferPtr + fieldOffset[classid_id])};
        case rho_id:
            return PointCloudGenericAttribute{fromBytes<float>(dataBufferPtr + fieldOffset[rho_id])};
        case reflectance_id:
            return PointCloudGenericAttribute{fromBytes<int16_t>(dataBufferPtr + fieldOffset[reflectance_id])};
        default:
            return std::nullopt;
    }
}

std::optional<PointCloudGenericAttribute> SdcPointCloudPoint::getAttributeByName(const char* attributeName) const {
    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    if (it != attributeNames.end()) {
        return getAttributeById(std::distance(attributeNames.begin(), it));
    }
    return std::nullopt;
}

std::vector<std::string> SdcPointCloudPoint::attributeList() const {
    return std::vector<std::string>(attributeNames.begin(), attributeNames.begin() + nbAttributes);
}

bool SdcPointCloudPoint::gotoNext() {
    static_assert(sizeof(float) == 4); // check if float is 4 bytes, should be true on most systems
    static_assert(sizeof(double) == 8); // check if double is 8 bytes
    // try to read a record
    // std::vector<char> buffer(recordByteSize);
    reader->read(dataBufferPtr, recordByteSize);
    if (!reader->good()) {
        return false; // end of file or read error
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

    auto inputFile = std::make_unique<ifstreamCustomBuffer<sdcFileReaderBufferSize>>();

    inputFile->open(sdcFilePath, std::ios_base::binary);

    // return null if the file can't be opened
    if (!inputFile->is_open()) return std::nullopt;

    // first 4 bytes are the size of the header
    uint32_t headerSize;
    inputFile->read(reinterpret_cast<char*>(&headerSize), 4);
    if (!inputFile->good()) return std::nullopt;

    // read the rest of the header
     std::vector<char> bufferHeader(headerSize - 4);
    inputFile->read(bufferHeader.data(), headerSize - 4);
    if (!inputFile->good()) return std::nullopt;

    // next 2 bytes are the major version
    auto majorVersion = fromBytes<uint16_t>(bufferHeader.data());

    // next 2 bytes are the minor version
    auto minorVersion = fromBytes<uint16_t>(bufferHeader.data() + 2);

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