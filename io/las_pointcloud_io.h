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

// the name of the extra attributes, the data type, the size and the offset. The offset is in bytes and 0 corresponds
// to the first byte AFTER the minimum record size of the record.
struct LasExtraAttributesInfos {
    std::vector<std::string> name;
    std::vector<uint8_t> type;
    std::vector<size_t> size;
    std::vector<size_t> offset;
};

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
    // constructor
    LasPublicHeaderBlock();
    // read the header of the LAS file
    static std::optional<LasPublicHeaderBlock> readPublicHeader(std::istream& reader);
    // write the header of the LAS file
    static bool writePublicHeader(std::ostream& writer, const LasPublicHeaderBlock& header);
    // get an attribute by its id
    std::optional<PointCloudGenericAttribute> getPublicHeaderAttributeById(int id) const;
    // get the list of attributes
    inline std::vector<std::string> const& publicHeaderAttributeList() const { return publicHeaderAttributes; }
    
    // getters
    
    auto getPointDataRecordLength() const { return pointDataRecordLength; }
    auto getPointDataRecordFormat() const { return pointDataRecordFormat; }
};

/// Class representing a variable length record (or extended variable length record)
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
    // default constructor
    LasGenericVariableLengthRecord() = default;
    
    // constructor with user ID,record ID and data
    LasGenericVariableLengthRecord(const std::string& userId, uint16_t recordId, const std::vector<std::byte>& data);
    
    // getters
    uint16_t getReserved() const;
    std::string getUserId() const;
    uint16_t getRecordId() const;
    auto getRecordLengthAfterHeader() const;
    std::string getDescription() const;
    inline const std::vector<std::byte>& getData() const { return data; }
    inline const std::array<char, vlrHeaderSize>& getHeaderData() const { return vlrHeaderData; }

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

// structure used by las to describe the extra bytes
struct LasExtraBytesDescriptor {
    std::array<uint8_t, 2> reserved{};
    uint8_t data_type{};
    uint8_t options{};
    std::array<char, 32> name{};
    std::array<uint8_t, 4> unused{};
    std::array<std::byte, 8> no_data{};
    std::array<uint8_t, 16> deprecated1{};
    std::array<std::byte, 8> min{};
    std::array<uint8_t, 16> deprecated2{};
    std::array<std::byte, 8> max{};
    std::array<uint8_t, 16> deprecated3{};
    double scale{};
    std::array<uint8_t, 16> deprecated4{};
    double offset{};
    std::array<uint8_t, 16> deprecated5{};
    std::array<char, 32> description{};

    // constructors

    /**
     * @brief Construct a new Las Extra Bytes Descriptor object 
     * 
     * @param data_type The data type as defined in the LAS specification
     * @param name The name of the attribute. The maximum length is 32 characters, otherwise it will be truncated
     * @param size The size of the attribute in bytes if the data type is unknown (i.e. 0). If not, it is ignored.
     * @param description The description of the attribute
     * @param no_data A value that should be used when the attribute is not present. If the data_type is a unsigned
     * integer, the underlying type is considered to be uint64_t. If the data_type is a signed integer, the underlying
     * type is considered to be int64_t and if the data_type is a floating point, the underlying type is considered to be
     * double.
     * @param min The minimum value of the attribute. The underlying type is the same as no_data.
     * @param max The maximum value of the attribute. The underlying type is the same as min.
     * @param scale The scale factor
     * @param offset The offset
     */
    LasExtraBytesDescriptor(uint8_t data_type, const std::string& name, size_t size = 0,
        const std::optional<std::string>& description = std::nullopt,
        const std::optional<std::variant<uint64_t, int64_t, double>>& no_data = std::nullopt,
        const std::optional<std::variant<uint64_t, int64_t, double>>& min = std::nullopt,
        const std::optional<std::variant<uint64_t, int64_t, double>>& max = std::nullopt,
        std::optional<double> scale = std::nullopt, std::optional<double> offset = std::nullopt);

    LasExtraBytesDescriptor(const std::array<uint8_t, 2>& reserved, uint8_t data_type,
        uint8_t options, const std::array<char, 32>& name, const std::array<uint8_t, 4>& unused,
        const std::array<std::byte, 8>& no_data, const std::array<uint8_t, 16>& deprecated1,
        const std::array<std::byte, 8>& min, const std::array<uint8_t, 16>& deprecated2,
        const std::array<std::byte, 8>& max, const std::array<uint8_t, 16>& deprecated3,
        double scale, const std::array<uint8_t, 16>& deprecated4, double offset,
        const std::array<uint8_t, 16>& deprecated5, const std::array<char, 32>& description);

    // constructor: read the data from a buffer
    LasExtraBytesDescriptor(const char* buffer);
    LasExtraBytesDescriptor(const std::byte* buffer): LasExtraBytesDescriptor(reinterpret_cast<const char*>(buffer)) {}

    // convert to raw bytes
    void toBytes(char* buffer) const;
    void toBytes(std::byte* buffer) const { toBytes(reinterpret_cast<char*>(buffer)); }

    std::vector<std::byte> toBytes() const;
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

    /**
     * @brief Get the pointwise extra attributes names, data types, sizes and offsets
     * 
     * @param ignoreUndocumentedExtraBytes If true, the undocumented extra bytes (unknown sequence of bytes) will be ignored
     * @return A struct containing the names, data types, sizes and offsets
     */
    LasExtraAttributesInfos getPointwiseExtraAttributesInfos(bool ignoreUndocumentedExtraBytes = true) const;

    // read the header
    static std::unique_ptr<LasPointCloudHeader> readHeader(std::istream& reader);

    // write
    // TODO: make static ? NO: write method for each of them + virtual write method
    bool writePublicHeader(std::ostream& writer) const {return LasPublicHeaderBlock::writePublicHeader(writer, publicHeaderBlock); }
    bool writeVLRs(std::ostream& writer) const;
    bool writeEVLRs(std::ostream& writer) const;

    // generate a list of extra bytes descriptors from the VLRs and EVLRs
    static std::vector<LasExtraBytesDescriptor> extraBytesDescriptorsFromVlrs(const std::vector<LasVariableLengthRecord>& variableLengthRecords,
        const std::vector<LasExtendedVariableLengthRecord>& extendedVariableLengthRecords);

    // from the extra bytes descriptors, generate a list of names, a list of data types (as defined in LAS extra bytes descriptor) a list of sizes, and a list of offsets in bytes
    static LasExtraAttributesInfos generateExtraBytesInfo(const std::vector<LasExtraBytesDescriptor>& extraBytesDescriptors,
            bool ignoreUndocumentedExtraBytes = true);

    // from the extra bytes infos (names, data types, sizes and offsets), generate the vlr data for the extra bytes vlr/evlr
    static std::vector<std::byte> generateExtraBytesVlrData(const LasExtraAttributesInfos& extraAttributesInfos);
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

    virtual size_t getFormat() const = 0; // return the las format

    virtual size_t getMinimumNumberOfAttributes() const = 0;

    virtual LasExtraAttributesInfos getExtraAttributesInfos() const = 0;

    bool write(std::ostream& writer) const;

    /**
     * @brief Obtain an adapter to a LasPointCloudPoint from any PointCloudPointAccessInterface. The adapted interface
     * can be the given interface if the object is already a LasPointCloudPoint or a wrapper otherwise.
     * If the given interface is null, a nullptr is returned.
     * 
     * @param pointCloudPointAccessInterface the interface to adapt
     * @return a shared_ptr to the adapted interface
     */
    static std::shared_ptr<LasPointCloudPoint> createAdapter(PointCloudPointAccessInterface* pointCloudPointAccessInterface);
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

/**
 * @brief
 * 
 * Open a LAS point cloud from a stream and returns a FullPointCloudAccessInterface
 * containing the header and the points.
 * 
 * @param reader The stream to open the point cloud from
 * 
 * @return A FullPointCloudAccessInterface containing the header and the points.
 *         If the stream can't be opened, an empty optional is returned
 * 
 */
std::optional<FullPointCloudAccessInterface> openPointCloudLas(std::unique_ptr<std::istream> reader);

/**
 * @brief Write a point cloud to a las file
 * 
 * @param writer The stream to write the point cloud to
 * @param pointCloud The point cloud to write to the stream
 *
 * @return True if the point cloud was written to the stream, false otherwise
 * 
 */
bool writePointCloudLas(std::ostream& writer, FullPointCloudAccessInterface& pointCloud);

/**
 * @brief
 * 
 * Write a point cloud to a las file.
 * 
 * @param lasFilePath The path to the las file to write.
 * @param pointCloud The point cloud to write to the las file.
 * @param dataStorageType The data storage type to use. If not specified, the data storage type defined in the header will be used
 * 
 * @return True if the point cloud was written to the las file, false otherwise
 */
bool writePointCloudLas(const std::filesystem::path& lasFilePath, FullPointCloudAccessInterface& pointCloud);


/*************** Implementation ***************/

template <bool isExtended>
inline LasGenericVariableLengthRecord<isExtended>::LasGenericVariableLengthRecord(const std::string &userId,
    uint16_t recordId, const std::vector<std::byte> &data) {
    
    this->data = data;
    std::array<char, userId_size> userIdArray{};
    std::copy_n(userId.begin(), std::min(userId.length(), userId_size), userIdArray.begin());
    std::memcpy(vlrHeaderData.data() + userId_offset, userIdArray.data(), userId_size);
    std::memcpy(vlrHeaderData.data() + recordId_offset, &recordId, recordId_size);

    if constexpr (isExtended) {
        uint64_t dataSize = static_cast<uint64_t>(data.size());
        std::memcpy(vlrHeaderData.data() + recordLengthAfterHeader_offset, &dataSize, recordLengthAfterHeader_size);
    } else {
        uint16_t dataSize = static_cast<uint16_t>(data.size());
        std::memcpy(vlrHeaderData.data() + recordLengthAfterHeader_offset, &dataSize, recordLengthAfterHeader_size);
    }
}

template <bool isExtended>
inline uint16_t LasGenericVariableLengthRecord<isExtended>::getReserved() const
{
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
