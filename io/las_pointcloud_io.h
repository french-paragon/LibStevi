#ifndef STEREOVISION_IO_LASPOINTCLOUD_H
#define STEREOVISION_IO_LASPOINTCLOUD_H

#include <array>
#include "pointcloud_io.h"
#include "bit_manipulations.h"

namespace StereoVision {
namespace IO {

// forward declaration
template <bool extended> class LasGenericVariableLengthRecord;

using LasVariableLengthRecord = LasGenericVariableLengthRecord<false>;
using LasExtendedVariableLengthRecord = LasGenericVariableLengthRecord<true>;

class LasPublicHeaderBlock {
public:
    inline static const std::vector<std::string> publicHeaderAttributes =
        {"fileSignature", "fileSourceID", "globalEncoding", "projectID_GUID_Data1", "projectID_GUID_Data2",
         "projectID_GUID_Data3", "projectID_GUID_Data4", "versionMajor", "versionMinor", "systemIdentifier",
         "generatingSoftware", "fileCreationDayOfYear", "fileCreationYear", "headerSize", "offsetToPointData",
         "numberOfVariableLengthRecords", "pointDataRecordFormat", "pointDataRecordLength", "legacyNumberOfPointRecords",
         "legacyNumberOfPointsByReturn", "xScaleFactor", "yScaleFactor", "zScaleFactor", "xOffset", "yOffset",
         "zOffset", "maxX", "minX", "maxY", "minY", "maxZ", "minZ", "startOfWaveformDataPacketRecord",
         "startOfFirstExtendedVariableLengthRecord", "numberOfExtendedVariableLengthRecords", "numberOfPointRecords",
         "numberOfPointsByReturn"};

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

private:
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
    static_assert(headerBlockByteSize == numberOfPointsByReturn_offset + numberOfPointsByReturn_size, "Header block size should be 375 bytes");
public:
    // read the header of the LAS file
    static std::optional<LasPublicHeaderBlock> readPublicHeader(std::istream& reader);
    // write the header of the LAS file
    static void writePublicHeader(std::ostream& writer, const LasPublicHeaderBlock& header);
};

/// Class representing a variable length record
template<bool isExtended>
class LasGenericVariableLengthRecord {
private:
    // the size of the elements in bytes
    static constexpr auto reserved_size = sizeof(uint16_t);
    static constexpr auto userId_size = sizeof(uint8_t)*16;
    static constexpr auto recordId_size = sizeof(uint16_t);
    static constexpr auto recordLengthAfterHeader_size = [] { if constexpr (isExtended) return sizeof(uint64_t); else return sizeof(uint16_t); }();
    static constexpr auto description_size = sizeof(uint8_t)*32;

    // offsets in bytes in the block
    static constexpr auto reserved_offset = 0;
    static constexpr auto userId_offset = reserved_offset + reserved_size;
    static constexpr auto recordId_offset = userId_offset + userId_size;
    static constexpr auto recordLengthAfterHeader_offset = recordId_offset + recordId_size;
    static constexpr auto description_offset = recordLengthAfterHeader_offset + recordLengthAfterHeader_size;
    static constexpr auto data_offset = description_offset + description_size;
    static constexpr auto vlrHeaderSize = [] { if constexpr (isExtended) return 60; else return 54; }();
    static_assert(data_offset == vlrHeaderSize, "Variable length record header size is incorrect");

    // header data and actual data
    std::array<char, vlrHeaderSize> vlrHeaderData;
    std::vector<std::byte> data;

public:
    // getters
    uint16_t getReserved() const;
    std::string getUserId() const;
    uint16_t getRecordId() const;
    auto getRecordLengthAfterHeader() const;
    std::string getDescription() const;
    inline std::vector<std::byte> getData() const { return data; }

    // read one variable length record

    /***
     * @brief read one variable length record
     * 
     * @param reader The input stream to read from.
     * @return An optional variable length record or std::nullopt if the read fails.
     * 
     */
    static std::optional<LasGenericVariableLengthRecord> readVariableLengthRecord(std::istream& reader);

    /***
     * @brief read all the variable length records in the file and return An optional vector
     * If the read fails, std::nullopt is returned
     * 
     * @param reader The input stream to read from.
     * @param recordCount The number of variable length records to read.
     * @return An optional vector of variable length records or std::nullopt if the read fails.
     */
    static std::optional<std::vector<LasGenericVariableLengthRecord>> readVariableLengthRecords(std::istream& reader, size_t recordCount);
    
};

class LasPointCloudHeader : public PointCloudHeaderInterface {
public:
    LasPublicHeaderBlock publicHeaderBlock;
    std::vector<LasVariableLengthRecord> variableLengthRecords;
    std::vector<LasExtendedVariableLengthRecord> extendedVariableLengthRecords;
protected:
    // attribute names for the header
public:
    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    static std::unique_ptr<LasPointCloudHeader> readHeader(std::istream& reader);
};

class LasPointCloudPoint : public PointCloudPointAccessInterface {
protected:
    size_t recordByteSize; // number of bytes in the las point record
    char* dataBuffer;
    
    /**
     * @brief Construct a new Las Point Cloud Point object
     * 
     * @param recordByteSize the size of the point record
     */
    LasPointCloudPoint(size_t recordByteSize);

    /**
     * @brief Construct a new Las Point Cloud Point object with a data buffer
     * 
     * @param recordByteSize the size of the point record
     * @param dataBuffer the data buffer
     *
     **/
    LasPointCloudPoint(size_t recordByteSize, char* dataBuffer);

private:
    std::vector<char> dataBufferContainer;
public:
    inline auto* getRecordDataBuffer() const { return dataBuffer; }
    inline auto getRecordByteSize() const { return recordByteSize; }
};

/**
 * @brief
 *
 * Open a point cloud from a las file and returns a FullPointCloudAccessInterface
 * containing the header and the points.
 *
 * @param lasFilePath The path to the las file containing the point cloud
 *
 * @return A FullPointCloudAccessInterface containing the header and the points.
 *         If the file can't be opened, an empty optional is returned
 */
std::optional<FullPointCloudAccessInterface> openPointCloudLas(const std::filesystem::path& lasFilePath);

/*************** Implementation ***************/

template <bool isExtended>
inline uint16_t LasGenericVariableLengthRecord<isExtended>::getReserved() const {
    return fromBytes<uint16_t>(vlrHeaderData.data() + reserved_offset);
}

template <bool isExtended>
inline std::string LasGenericVariableLengthRecord<isExtended>::getUserId() const {
    const auto begin = vlrHeaderData.begin() + userId_offset;
    const auto end = std::find(begin, begin + userId_size, '\0');
    return std::string{begin, end};
}

template <bool isExtended>
inline uint16_t LasGenericVariableLengthRecord<isExtended>::getRecordId() const {
    return fromBytes<uint16_t>(vlrHeaderData.data() + recordId_offset);
}

template <bool isExtended>
inline auto LasGenericVariableLengthRecord<isExtended>::getRecordLengthAfterHeader() const {
    if constexpr (isExtended) {
        return fromBytes<uint64_t>(vlrHeaderData.data() + recordLengthAfterHeader_offset);
    } else {
        return fromBytes<uint16_t>(vlrHeaderData.data() + recordLengthAfterHeader_offset);
    }
}

template <bool isExtended>
inline std::string LasGenericVariableLengthRecord<isExtended>::getDescription() const {
    const auto begin = vlrHeaderData.begin() + description_offset;
    const auto end = std::find(begin, begin + description_size, '\0');
    return std::string{begin, end};
}

template <bool isExtended>
inline std::optional<LasGenericVariableLengthRecord<isExtended>> LasGenericVariableLengthRecord<isExtended>::readVariableLengthRecord(std::istream &reader) {
    LasGenericVariableLengthRecord<isExtended> vlr;
    reader.read(vlr.vlrHeaderData.data(), vlrHeaderSize);
    if (reader.fail()) return std::nullopt;
    vlr.data.resize(vlr.getRecordLengthAfterHeader());
    reader.read(reinterpret_cast<char*>(vlr.data.data()), vlr.getRecordLengthAfterHeader());
    if (reader.fail()) return std::nullopt;

    return vlr;
}

template <bool isExtended>
inline std::optional<std::vector<LasGenericVariableLengthRecord<isExtended>>> LasGenericVariableLengthRecord<isExtended>::readVariableLengthRecords(std::istream &reader, size_t recordCount) {
    std::vector<LasGenericVariableLengthRecord<isExtended>> records;
    records.reserve(recordCount);
    for (size_t i = 0; i < recordCount; i++) {
        // read the data of the record
        auto vlr = readVariableLengthRecord(reader);
        if (vlr) {
            records.push_back(std::move(*vlr));
        } else {
            return std::nullopt;
        }
    }
    return records;
}

}
}

#endif //STEREOVISION_IO_LASPOINTCLOUD_H