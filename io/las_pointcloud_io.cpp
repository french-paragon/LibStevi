#include <cstring>
#include <array>
#include <iostream>
#include "las_pointcloud_io.h"

namespace StereoVision {
namespace IO {

std::optional<PointCloudGenericAttribute> LasPointCloudHeader::getAttributeById(int id) const
{
    return std::optional<PointCloudGenericAttribute>();
}

std::optional<PointCloudGenericAttribute> LasPointCloudHeader::getAttributeByName(const char *attributeName) const
{
    return std::optional<PointCloudGenericAttribute>();
}
std::vector<std::string> LasPointCloudHeader::attributeList() const
{
    return publicHeaderBlock.publicHeaderAttributes;
}
std::unique_ptr<LasPointCloudHeader> LasPointCloudHeader::readHeader(std::istream &reader)
{
    LasPublicHeaderBlock::readPublicHeader(reader);
    return std::unique_ptr<LasPointCloudHeader>();
}

std::optional<LasPublicHeaderBlock> LasPublicHeaderBlock::readPublicHeader(std::istream &reader) {
    // read the data
    LasPublicHeaderBlock header;
    reader.read(header.fileSignature.data(), fileSignature_size);
    reader.read(reinterpret_cast<char*>(&header.fileSourceID), fileSourceID_size);
    reader.read(reinterpret_cast<char*>(&header.globalEncoding), globalEncoding_size);
    reader.read(reinterpret_cast<char*>(&header.projectID_GUID_Data1), projectID_GUID_Data1_size);
    reader.read(reinterpret_cast<char*>(&header.projectID_GUID_Data2), projectID_GUID_Data2_size);
    reader.read(reinterpret_cast<char*>(&header.projectID_GUID_Data3), projectID_GUID_Data3_size);
    reader.read(reinterpret_cast<char*>(header.projectID_GUID_Data4.data()), projectID_GUID_Data4_size);
    reader.read(reinterpret_cast<char*>(&header.versionMajor), versionMajor_size);
    reader.read(reinterpret_cast<char*>(&header.versionMinor), versionMinor_size);
    reader.read(reinterpret_cast<char*>(header.systemIdentifier.data()), systemIdentifier_size);
    reader.read(reinterpret_cast<char*>(header.generatingSoftware.data()), generatingSoftware_size);
    reader.read(reinterpret_cast<char*>(&header.fileCreationDayOfYear), fileCreationDayOfYear_size);
    reader.read(reinterpret_cast<char*>(&header.fileCreationYear), fileCreationYear_size);
    reader.read(reinterpret_cast<char*>(&header.headerSize), headerSize_size);
    reader.read(reinterpret_cast<char*>(&header.offsetToPointData), offsetToPointData_size);
    reader.read(reinterpret_cast<char*>(&header.numberOfVariableLengthRecords), numberOfVariableLengthRecords_size);
    reader.read(reinterpret_cast<char*>(&header.pointDataRecordFormat), pointDataRecordFormat_size);
    reader.read(reinterpret_cast<char*>(&header.pointDataRecordLength), pointDataRecordLength_size);
    reader.read(reinterpret_cast<char*>(&header.legacyNumberOfPointRecords), legacyNumberOfPointRecords_size);
    reader.read(reinterpret_cast<char*>(header.legacyNumberOfPointsByReturn.data()), legacyNumberOfPointsByReturn_size);
    reader.read(reinterpret_cast<char*>(&header.xScaleFactor), xScaleFactor_size);
    reader.read(reinterpret_cast<char*>(&header.yScaleFactor), yScaleFactor_size);
    reader.read(reinterpret_cast<char*>(&header.zScaleFactor), zScaleFactor_size);
    reader.read(reinterpret_cast<char*>(&header.xOffset), xOffset_size);
    reader.read(reinterpret_cast<char*>(&header.yOffset), yOffset_size);
    reader.read(reinterpret_cast<char*>(&header.zOffset), zOffset_size);
    reader.read(reinterpret_cast<char*>(&header.maxX), maxX_size);
    reader.read(reinterpret_cast<char*>(&header.minX), minX_size);
    reader.read(reinterpret_cast<char*>(&header.maxY), maxY_size);
    reader.read(reinterpret_cast<char*>(&header.minY), minY_size);
    reader.read(reinterpret_cast<char*>(&header.maxZ), maxZ_size);
    reader.read(reinterpret_cast<char*>(&header.minZ), minZ_size);
    reader.read(reinterpret_cast<char*>(&header.startOfWaveformDataPacketRecord), startOfWaveformDataPacketRecord_size);
    reader.read(reinterpret_cast<char*>(&header.startOfFirstExtendedVariableLengthRecord), startOfFirstExtendedVariableLengthRecord_size);
    reader.read(reinterpret_cast<char*>(&header.numberOfExtendedVariableLengthRecords), numberOfExtendedVariableLengthRecords_size);
    reader.read(reinterpret_cast<char*>(&header.numberOfPointRecords), numberOfPointRecords_size);
    reader.read(reinterpret_cast<char*>(header.numberOfPointsByReturn.data()), numberOfPointsByReturn_size);

    // display the data
    std::cout << "file signature: " << std::string(header.fileSignature.data(), fileSignature_size) << std::endl;
    std::cout << "file source id: " << header.fileSourceID << std::endl;
    std::cout << "global encoding: " << header.globalEncoding << std::endl;
    std::cout << "project id guid data1: " << header.projectID_GUID_Data1 << std::endl;
    std::cout << "project id guid data2: " << header.projectID_GUID_Data2 << std::endl;
    std::cout << "project id guid data3: " << header.projectID_GUID_Data3 << std::endl;
    std::cout << "project id guid data4: ";
    for (auto&& guiddata4 : header.projectID_GUID_Data4) {
        std::cout << guiddata4 << " ";
    }
    std::cout << std::endl;
    std::cout << "version major: " << static_cast<uint16_t>(header.versionMajor) << std::endl;
    std::cout << "version minor: " << static_cast<uint16_t>(header.versionMinor) << std::endl;
    std::cout << "system identifier: " << std::string(header.systemIdentifier.data(), systemIdentifier_size) << std::endl;
    std::cout << "generating software: " << std::string(header.generatingSoftware.data(), generatingSoftware_size) << std::endl;
    std::cout << "file creation day of year: " << header.fileCreationDayOfYear << std::endl;
    std::cout << "file creation year: " << header.fileCreationYear << std::endl;
    std::cout << "header size: " << header.headerSize << std::endl;
    std::cout << "offset to point data: " << header.offsetToPointData << std::endl;
    std::cout << "number of variable length records: " << header.numberOfVariableLengthRecords << std::endl;
    std::cout << "point data record format: " << static_cast<uint16_t>(header.pointDataRecordFormat) << std::endl;
    std::cout << "point data record length: " << header.pointDataRecordLength << std::endl;
    std::cout << "legacy number of point records: " << header.legacyNumberOfPointRecords << std::endl;
    std::cout << "legacy number of points by return: ";
    for (auto&& legacyNumberOfPointsByReturn : header.legacyNumberOfPointsByReturn) {
        std::cout << legacyNumberOfPointsByReturn << " ";
    }
    std::cout << std::endl;
    std::cout << "x scale factor: " << header.xScaleFactor << std::endl;
    std::cout << "y scale factor: " << header.yScaleFactor << std::endl;
    std::cout << "z scale factor: " << header.zScaleFactor << std::endl;
    std::cout << "x offset: " << header.xOffset << std::endl;
    std::cout << "y offset: " << header.yOffset << std::endl;
    std::cout << "z offset: " << header.zOffset << std::endl;
    std::cout << "max x: " << header.maxX << std::endl;
    std::cout << "min x: " << header.minX << std::endl;
    std::cout << "max y: " << header.maxY << std::endl;
    std::cout << "min y: " << header.minY << std::endl;
    std::cout << "max z: " << header.maxZ << std::endl;
    std::cout << "min z: " << header.minZ << std::endl;
    std::cout << "start of waveform data packet record: " << header.startOfWaveformDataPacketRecord << std::endl;
    std::cout << "start of first extended variable length record: " << header.startOfFirstExtendedVariableLengthRecord << std::endl;
    std::cout << "number of extended variable length records: " << header.numberOfExtendedVariableLengthRecords << std::endl;
    std::cout << "number of point records: " << header.numberOfPointRecords << std::endl;
    std::cout << "number of points by return: ";
    for (auto&& numberOfPointsByReturn : header.numberOfPointsByReturn) {
        std::cout << numberOfPointsByReturn << " ";
    }
    std::cout << std::endl;

    if (reader.fail()) return std::nullopt;

    return header;

}

void LasPublicHeaderBlock::writePublicHeader(std::ostream &writer, const LasPublicHeaderBlock &header) {
    writer.write(header.fileSignature.data(), fileSignature_size);
    writer.write(reinterpret_cast<const char*>(&header.fileSourceID), fileSourceID_size);
    writer.write(reinterpret_cast<const char*>(&header.globalEncoding), globalEncoding_size);
    writer.write(reinterpret_cast<const char*>(&header.projectID_GUID_Data1), projectID_GUID_Data1_size);
    writer.write(reinterpret_cast<const char*>(&header.projectID_GUID_Data2), projectID_GUID_Data2_size);
    writer.write(reinterpret_cast<const char*>(&header.projectID_GUID_Data3), projectID_GUID_Data3_size);
    writer.write(reinterpret_cast<const char*>(header.projectID_GUID_Data4.data()), projectID_GUID_Data4_size);
    writer.write(reinterpret_cast<const char*>(&header.versionMajor), versionMajor_size);
    writer.write(reinterpret_cast<const char*>(&header.versionMinor), versionMinor_size);
    writer.write(reinterpret_cast<const char*>(header.systemIdentifier.data()), systemIdentifier_size);
    writer.write(reinterpret_cast<const char*>(header.generatingSoftware.data()), generatingSoftware_size);
    writer.write(reinterpret_cast<const char*>(&header.fileCreationDayOfYear), fileCreationDayOfYear_size);
    writer.write(reinterpret_cast<const char*>(&header.fileCreationYear), fileCreationYear_size);
    writer.write(reinterpret_cast<const char*>(&header.headerSize), headerSize_size);
    writer.write(reinterpret_cast<const char*>(&header.offsetToPointData), offsetToPointData_size);
    writer.write(reinterpret_cast<const char*>(&header.numberOfVariableLengthRecords), numberOfVariableLengthRecords_size);
    writer.write(reinterpret_cast<const char*>(&header.pointDataRecordFormat), pointDataRecordFormat_size);
    writer.write(reinterpret_cast<const char*>(&header.pointDataRecordLength), pointDataRecordLength_size);
    writer.write(reinterpret_cast<const char*>(&header.legacyNumberOfPointRecords), legacyNumberOfPointRecords_size);
    writer.write(reinterpret_cast<const char*>(header.legacyNumberOfPointsByReturn.data()), legacyNumberOfPointsByReturn_size);
    writer.write(reinterpret_cast<const char*>(&header.xScaleFactor), xScaleFactor_size);
    writer.write(reinterpret_cast<const char*>(&header.yScaleFactor), yScaleFactor_size);
    writer.write(reinterpret_cast<const char*>(&header.zScaleFactor), zScaleFactor_size);
    writer.write(reinterpret_cast<const char*>(&header.xOffset), xOffset_size);
    writer.write(reinterpret_cast<const char*>(&header.yOffset), yOffset_size);
    writer.write(reinterpret_cast<const char*>(&header.zOffset), zOffset_size);
    writer.write(reinterpret_cast<const char*>(&header.maxX), maxX_size);
    writer.write(reinterpret_cast<const char*>(&header.minX), minX_size);
    writer.write(reinterpret_cast<const char*>(&header.maxY), maxY_size);
    writer.write(reinterpret_cast<const char*>(&header.minY), minY_size);
    writer.write(reinterpret_cast<const char*>(&header.maxZ), maxZ_size);
    writer.write(reinterpret_cast<const char*>(&header.minZ), minZ_size);
    writer.write(reinterpret_cast<const char*>(&header.startOfWaveformDataPacketRecord), startOfWaveformDataPacketRecord_size);
    writer.write(reinterpret_cast<const char*>(&header.startOfFirstExtendedVariableLengthRecord), startOfFirstExtendedVariableLengthRecord_size);
    writer.write(reinterpret_cast<const char*>(&header.numberOfExtendedVariableLengthRecords), numberOfExtendedVariableLengthRecords_size);
    writer.write(reinterpret_cast<const char*>(&header.numberOfPointRecords), numberOfPointRecords_size);
    writer.write(reinterpret_cast<const char*>(header.numberOfPointsByReturn.data()), numberOfPointsByReturn_size);
}

}
}
