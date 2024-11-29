#ifndef STEREOVISION_IO_LASPOINTCLOUD_H
#define STEREOVISION_IO_LASPOINTCLOUD_H

#include <array>
#include "pointcloud_io.h"

namespace StereoVision {
namespace IO {

class LasPublicHeaderBlock {
public:
    const std::vector<std::string> publicHeaderAttributes =
        {"fileSignature", "fileSourceID", "globalEncoding", "projectID_GUID_Data1", "projectID_GUID_Data2",
         "projectID_GUID_Data3", "projectID_GUID_Data4", "versionMajor", "versionMinor", "systemIdentifier",
         "generatingSoftware", "fileCreationDayOfYear", "fileCreationYear", "headerSize", "offsetToPointData",
         "numberOfVariableLengthRecords", "pointDataRecordFormat", "pointDataRecordLength", "legacyNumberOfPointRecords",
         "legacyNumberOfPointsByReturn", "xScaleFactor", "yScaleFactor", "zScaleFactor", "xOffset", "yOffset",
         "zOffset", "maxX", "minX", "maxY", "minY", "maxZ", "minZ", "startOfWaveformDataPacketRecord",
         "startOfFirstExtendedVariableLengthRecord", "numberOfExtendedVariableLengthRecords", "numberOfPointRecords",
         "numberOfPointsByReturn"};

private:
    std::array<char, 4>  fileSignature;       // "LASF"
    uint16_t fileSourceID;                    // File Source ID
    uint16_t globalEncoding;                  // Global Encoding
    uint32_t projectID_GUID_Data1;             // Project ID - GUID Data 1
    uint16_t projectID_GUID_Data2;            // Project ID - GUID Data 2
    uint16_t projectID_GUID_Data3;            // Project ID - GUID Data 3
    std::array<uint8_t, 8> projectID_GUID_Data4; // Project ID - GUID Data 4
    uint8_t versionMajor;                     // Version Major
    uint8_t versionMinor;                     // Version Minor
    std::array<char, 32> systemIdentifier;    // System Identifier
    std::array<char, 32> generatingSoftware;  // Generating Software
    uint16_t fileCreationDayOfYear;           // File Creation Day of Year
    uint16_t fileCreationYear;                // File Creation Year
    uint16_t headerSize;                      // Header Size
    uint32_t offsetToPointData;                // Offset to Point Data
    uint32_t numberOfVariableLengthRecords;    // Number of Variable Length Records
    uint8_t pointDataRecordFormat;            // Point Data Record Format
    uint16_t pointDataRecordLength;           // Point Data Record Length
    uint32_t legacyNumberOfPointRecords;       // Legacy Number of Point Records
    std::array<uint32_t, 5> legacyNumberOfPointsByReturn; // Legacy Number of Point by Return
    double xScaleFactor;                            // X Scale Factor
    double yScaleFactor;                            // Y Scale Factor
    double zScaleFactor;                            // Z Scale Factor
    double xOffset;                                 // X Offset
    double yOffset;                                 // Y Offset
    double zOffset;                                 // Z Offset
    double maxX;                                    // Max X
    double minX;                                    // Min X
    double maxY;                                    // Max Y
    double minY;                                    // Min Y
    double maxZ;                                    // Max Z
    double minZ;                                    // Min Z
    uint64_t startOfWaveformDataPacketRecord; // Start of Waveform Data Packet Record
    uint64_t startOfFirstExtendedVariableLengthRecord; // Start of First Extended Variable Length Record
    uint32_t numberOfExtendedVariableLengthRecords; // Number of Extended Variable Length Records
    uint64_t numberOfPointRecords;        // Number of Point Records
    std::array<uint64_t, 15> numberOfPointsByReturn;  // Number of Points by Return

    // size in bytes of the data in the block
    static constexpr size_t fileSignature_size = sizeof(decltype(fileSignature)::value_type) * 4;
    static constexpr size_t fileSourceID_size = sizeof(fileSourceID);
    static constexpr size_t globalEncoding_size = sizeof(globalEncoding);
    static constexpr size_t projectID_GUID_Data1_size = sizeof(projectID_GUID_Data1);
    static constexpr size_t projectID_GUID_Data2_size = sizeof(projectID_GUID_Data2);
    static constexpr size_t projectID_GUID_Data3_size = sizeof(projectID_GUID_Data3);
    static constexpr size_t projectID_GUID_Data4_size = sizeof(decltype(projectID_GUID_Data4)::value_type) * 8;
    static constexpr size_t versionMajor_size = sizeof(versionMajor);
    static constexpr size_t versionMinor_size = sizeof(versionMinor);
    static constexpr size_t systemIdentifier_size = sizeof(decltype(systemIdentifier)::value_type) * 32;
    static constexpr size_t generatingSoftware_size = sizeof(decltype(generatingSoftware)::value_type) * 32;
    static constexpr size_t fileCreationDayOfYear_size = sizeof(fileCreationDayOfYear);
    static constexpr size_t fileCreationYear_size = sizeof(fileCreationYear);
    static constexpr size_t headerSize_size = sizeof(headerSize);
    static constexpr size_t offsetToPointData_size = sizeof(offsetToPointData);
    static constexpr size_t numberOfVariableLengthRecords_size = sizeof(numberOfVariableLengthRecords);
    static constexpr size_t pointDataRecordFormat_size = sizeof(pointDataRecordFormat);
    static constexpr size_t pointDataRecordLength_size = sizeof(pointDataRecordLength);
    static constexpr size_t legacyNumberOfPointRecords_size = sizeof(legacyNumberOfPointRecords);
    static constexpr size_t legacyNumberOfPointsByReturn_size = sizeof(decltype(legacyNumberOfPointsByReturn)::value_type) * 5;
    static constexpr size_t xScaleFactor_size = sizeof(xScaleFactor);
    static constexpr size_t yScaleFactor_size = sizeof(yScaleFactor);
    static constexpr size_t zScaleFactor_size = sizeof(zScaleFactor);
    static constexpr size_t xOffset_size = sizeof(xOffset);
    static constexpr size_t yOffset_size = sizeof(yOffset);
    static constexpr size_t zOffset_size = sizeof(zOffset);
    static constexpr size_t maxX_size = sizeof(maxX);
    static constexpr size_t minX_size = sizeof(minX);
    static constexpr size_t maxY_size = sizeof(maxY);
    static constexpr size_t minY_size = sizeof(minY);
    static constexpr size_t maxZ_size = sizeof(maxZ);
    static constexpr size_t minZ_size = sizeof(minZ);
    static constexpr size_t startOfWaveformDataPacketRecord_size = sizeof(startOfWaveformDataPacketRecord);
    static constexpr size_t startOfFirstExtendedVariableLengthRecord_size = sizeof(startOfFirstExtendedVariableLengthRecord);
    static constexpr size_t numberOfExtendedVariableLengthRecords_size = sizeof(numberOfExtendedVariableLengthRecords);
    static constexpr size_t numberOfPointRecords_size = sizeof(numberOfPointRecords);
    static constexpr size_t numberOfPointsByReturn_size = sizeof(decltype(numberOfPointsByReturn)::value_type) * 15;

    // offsets in bytes in the block
    static constexpr size_t fileSignature_offset = 0;
    static constexpr size_t fileSourceID_offset = fileSignature_offset + fileSignature_size;
    static constexpr size_t globalEncoding_offset = fileSourceID_offset + fileSourceID_size;
    static constexpr size_t projectID_GUID_Data1_offset = globalEncoding_offset + globalEncoding_size;
    static constexpr size_t projectID_GUID_Data2_offset = projectID_GUID_Data1_offset + projectID_GUID_Data1_size;
    static constexpr size_t projectID_GUID_Data3_offset = projectID_GUID_Data2_offset + projectID_GUID_Data2_size;
    static constexpr size_t projectID_GUID_Data4_offset = projectID_GUID_Data3_offset + projectID_GUID_Data3_size;
    static constexpr size_t versionMajor_offset = projectID_GUID_Data4_offset + projectID_GUID_Data4_size;
    static constexpr size_t versionMinor_offset = versionMajor_offset + versionMajor_size;
    static constexpr size_t systemIdentifier_offset = versionMinor_offset + versionMinor_size;
    static constexpr size_t generatingSoftware_offset = systemIdentifier_offset + systemIdentifier_size;
    static constexpr size_t fileCreationDayOfYear_offset = generatingSoftware_offset + generatingSoftware_size;
    static constexpr size_t fileCreationYear_offset = fileCreationDayOfYear_offset + fileCreationDayOfYear_size;
    static constexpr size_t headerSize_offset = fileCreationYear_offset + fileCreationYear_size;
    static constexpr size_t offsetToPointData_offset = headerSize_offset + headerSize_size;
    static constexpr size_t numberOfVariableLengthRecords_offset = offsetToPointData_offset + offsetToPointData_size;
    static constexpr size_t pointDataRecordFormat_offset = numberOfVariableLengthRecords_offset + numberOfVariableLengthRecords_size;
    static constexpr size_t pointDataRecordLength_offset = pointDataRecordFormat_offset + pointDataRecordFormat_size;
    static constexpr size_t legacyNumberOfPointRecords_offset = pointDataRecordLength_offset + pointDataRecordLength_size;
    static constexpr size_t legacyNumberOfPointsByReturn_offset = legacyNumberOfPointRecords_offset + legacyNumberOfPointRecords_size;
    static constexpr size_t xScaleFactor_offset = legacyNumberOfPointsByReturn_offset + legacyNumberOfPointsByReturn_size;
    static constexpr size_t yScaleFactor_offset = xScaleFactor_offset + xScaleFactor_size;
    static constexpr size_t zScaleFactor_offset = yScaleFactor_offset + yScaleFactor_size;
    static constexpr size_t xOffset_offset = zScaleFactor_offset + zScaleFactor_size;
    static constexpr size_t yOffset_offset = xOffset_offset + xOffset_size;
    static constexpr size_t zOffset_offset = yOffset_offset + yOffset_size;    
    static constexpr size_t maxX_offset = zOffset_offset + zOffset_size;
    static constexpr size_t minX_offset = maxX_offset + maxX_size;
    static constexpr size_t maxY_offset = minX_offset + minX_size;
    static constexpr size_t minY_offset = maxY_offset + maxY_size;
    static constexpr size_t maxZ_offset = minY_offset + minY_size;
    static constexpr size_t minZ_offset = maxZ_offset + maxZ_size;
    static constexpr size_t startOfWaveformDataPacketRecord_offset = minZ_offset + minZ_size;
    static constexpr size_t startOfFirstExtendedVariableLengthRecord_offset = startOfWaveformDataPacketRecord_offset + startOfWaveformDataPacketRecord_size;
    static constexpr size_t numberOfExtendedVariableLengthRecords_offset = startOfFirstExtendedVariableLengthRecord_offset + startOfFirstExtendedVariableLengthRecord_size;
    static constexpr size_t numberOfPointRecords_offset = numberOfExtendedVariableLengthRecords_offset + numberOfExtendedVariableLengthRecords_size;
    static constexpr size_t numberOfPointsByReturn_offset = numberOfPointRecords_offset + numberOfPointRecords_size;

    static constexpr size_t headerBlockByteSize = 375;
public:
    // read the header of the LAS file
    static std::optional<LasPublicHeaderBlock> readPublicHeader(std::istream& reader);
    // write the header of the LAS file
    static void writePublicHeader(std::ostream& writer, const LasPublicHeaderBlock& header);
};

class LasPointCloudHeader : public PointCloudHeaderInterface {
public:
    LasPublicHeaderBlock publicHeaderBlock;

protected:
    // attribute names for the header
public:
    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    static std::unique_ptr<LasPointCloudHeader> readHeader(std::istream& reader);
};

}
}

#endif //STEREOVISION_IO_LASPOINTCLOUD_H