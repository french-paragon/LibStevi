#ifndef STEREOVISION_IO_SDCPOINTCLOUD_H
#define STEREOVISION_IO_SDCPOINTCLOUD_H

#include "pointcloud_io.h"
#include <filesystem>
#include <array>
#include <numeric>

namespace StereoVision
{
namespace IO
{

class SdcPointCloudPoint : public PointCloudPointAccessInterface
{
public:
    // ****************************************
    // version informations
    const uint16_t majorVersion;
    const uint16_t minorVersion;

    // ************ attribute ids ************
    static constexpr auto time_id = 0;
    static constexpr auto range_id = 1;
    static constexpr auto theta_id = 2;
    static constexpr auto x_id = 3;
    static constexpr auto y_id = 4;
    static constexpr auto z_id = 5;
    static constexpr auto amplitude_id = 6;
    static constexpr auto width_id = 7;
    static constexpr auto targettype_id = 8;
    static constexpr auto target_id = 9;
    static constexpr auto numtarget_id = 10;
    static constexpr auto rgindex_id = 11;
    static constexpr auto channeldesc_id = 12;
    // from version 5.2
    static constexpr auto classid_id = 13;
    // from version 5.3
    static constexpr auto rho_id = 14;
    // from version 5.4
    static constexpr auto reflectance_id = 15;

    // ************ size of attributes ************
    static constexpr auto time_size = sizeof(double);
    static constexpr auto range_size = sizeof(float);
    static constexpr auto theta_size = sizeof(float);
    static constexpr auto x_size = sizeof(float);
    static constexpr auto y_size = sizeof(float);
    static constexpr auto z_size = sizeof(float);
    static constexpr auto amplitude_size = sizeof(uint16_t);
    static constexpr auto width_size = sizeof(uint16_t);
    static constexpr auto targettype_size = sizeof(uint8_t);
    static constexpr auto target_size = sizeof(uint8_t);
    static constexpr auto numtarget_size = sizeof(uint8_t);
    static constexpr auto rgindex_size = sizeof(uint16_t);
    static constexpr auto channeldesc_size = sizeof(uint8_t);
    // from version 5.2
    static constexpr auto classid_size = sizeof(uint8_t);
    // from version 5.3
    static constexpr auto rho_size = sizeof(float);
    // from version 5.4
    static constexpr auto reflectance_size = sizeof(int16_t);

    static constexpr std::array fieldByteSize = {time_size, range_size, theta_size, x_size, y_size,
        z_size, amplitude_size, width_size, targettype_size, target_size, numtarget_size,
        rgindex_size, channeldesc_size, classid_size, rho_size, reflectance_size};

    // ************ offsets ************
    static constexpr auto offset0  = size_t{0};
    static constexpr auto offset1  = offset0  + fieldByteSize[0];
    static constexpr auto offset2  = offset1  + fieldByteSize[1];
    static constexpr auto offset3  = offset2  + fieldByteSize[2];
    static constexpr auto offset4  = offset3  + fieldByteSize[3];
    static constexpr auto offset5  = offset4  + fieldByteSize[4];
    static constexpr auto offset6  = offset5  + fieldByteSize[5];
    static constexpr auto offset7  = offset6  + fieldByteSize[6];
    static constexpr auto offset8  = offset7  + fieldByteSize[7];
    static constexpr auto offset9  = offset8  + fieldByteSize[8];
    static constexpr auto offset10 = offset9  + fieldByteSize[9];
    static constexpr auto offset11 = offset10 + fieldByteSize[10];
    static constexpr auto offset12 = offset11 + fieldByteSize[11];
    static constexpr auto offset13 = offset12 + fieldByteSize[12];
    static constexpr auto offset14 = offset13 + fieldByteSize[13];
    static constexpr auto offset15 = offset14 + fieldByteSize[14];

    static constexpr auto fieldOffset = std::array{offset0, offset1, offset2, offset3, offset4, offset5, offset6,
                                        offset7, offset8, offset9, offset10, offset11, offset12, offset13, offset14,
                                        offset15};

    // *************** field names ***********
    static constexpr std::string_view time_attName = "time";
    static constexpr std::string_view range_attName = "range";
    static constexpr std::string_view theta_attName = "theta";
    static constexpr std::string_view x_attName = "x";
    static constexpr std::string_view y_attName = "y";
    static constexpr std::string_view z_attName = "z";
    static constexpr std::string_view amplitude_attName = "amplitude";
    static constexpr std::string_view width_attName = "width";
    static constexpr std::string_view targettype_attName = "targettype";
    static constexpr std::string_view target_attName = "target";
    static constexpr std::string_view numtarget_attName = "numtarget";
    static constexpr std::string_view rgindex_attName = "rgindex";
    static constexpr std::string_view channeldesc_attName = "channeldesc";
    // from version 5.2
    static constexpr std::string_view classid_attName = "classid";
    // from version 5.3
    static constexpr std::string_view rho_attName = "rho";
    // from version 5.4
    static constexpr std::string_view reflectance_attName = "reflectance";
    
    static constexpr auto attributeNames = std::array{time_attName, range_attName, theta_attName, x_attName, y_attName,
        z_attName, amplitude_attName, width_attName, targettype_attName, target_attName, numtarget_attName,
        rgindex_attName, channeldesc_attName, classid_attName, rho_attName, reflectance_attName};

    // ***************************************        

    static constexpr auto dataBufferMaxSize = fieldOffset.back() + fieldByteSize.back();
    std::array<char, dataBufferMaxSize> dataBuffer;
    char* dataBufferPtr = dataBuffer.data();

    // number of attributes, it depends on the version 
    const size_t nbAttributes;

private:
    // reader "head"
    const std::unique_ptr<std::istream> reader;

    const size_t recordByteSize; // number of bytes in a sdc point record

    std::vector<std::string> exposedAttributeNames;
    std::vector<size_t> exposedIdToInternalId;

public:
    SdcPointCloudPoint(std::unique_ptr<std::istream> reader, uint16_t majorVersion, uint16_t minorVersion);

    virtual int64_t expectedNumberOfPoints() const override;
    virtual int64_t processedNumberOfPoints() const override;

    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    bool gotoNext() override;
    bool hasData() const override;
    
    inline auto getRecordByteSize() const { return recordByteSize; }
    inline auto* getRecordDataBuffer() const { return dataBufferPtr; }

    /**
     * @brief write a point cloud to a stream
     * 
     * @param stream the stream to write to
     * @return true if successful
     * @return false if failed
     */
    bool write(std::ostream& stream) const;
    
private:
    // compute the number of attributes based on the version
    static inline size_t computeNbAttributes(uint16_t majorVersion, uint16_t minorVersion) {
        if (majorVersion >= 5) {
            if (minorVersion >= 4) { // version 5.4
                return 16;
            } else if (minorVersion >= 3) { // version 5.3
                return 15;
            } else if (minorVersion >= 2) { // version 5.2
                return 14;
            }
        }
        return 13;
    }

    // compute the size of a sdc point record based on the version
    static inline size_t computeRecordByteSize(uint16_t majorVersion, uint16_t minorVersion) {
        const auto lastId = computeNbAttributes(majorVersion, minorVersion)-1;
        return fieldOffset[lastId] + fieldByteSize[lastId];
    }
};

class SdcPointCloudHeader : public PointCloudHeaderInterface
{
public:
    const uint32_t headerSize;
    const uint16_t majorVersion;
    const uint16_t minorVersion;
    const std::vector<std::byte> headerInformation;

private:
    // attribute names order should appear in the same order than the corresponding id of the attributes
    const std::vector<std::string> attributeNames = {"headerSize", "majorVersion", "minorVersion", "headerInformation"};

public:
    // constructor
    SdcPointCloudHeader(const uint32_t headerSize, const uint16_t majorVersion, const uint16_t minorVersion, const std::vector<std::byte>& headerInformation);

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    /**
     * @brief Write the point cloud header to a stream
     * 
     * @param stream The stream to write the header to
     * @return true if the header was written successfully
     * @return false otherwise
     */
    bool write(std::ostream& stream) const;
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
StatusOptional<FullPointCloudAccessInterface> openPointCloudSdc(const std::filesystem::path& sdcFilePath);

/**
 * @brief
 * 
 * Open a sdc point cloud from a stream and returns a FullPointCloudAccessInterface
 * containing the header and the points.
 * 
 * @param stream The stream to open the point cloud from
 * 
 * @return A FullPointCloudAccessInterface containing the header and the points.
 *         If the stream can't be opened, an empty optional is returned
 * 
 */
StatusOptional<FullPointCloudAccessInterface> openPointCloudSdc(std::unique_ptr<std::istream> stream);

/**
 * @brief
 * 
 * Write a sdc point cloud to a sdc file.
 * 
 * @param sdcFilePath The path to the sdc file to write.
 * @param pointCloud The point cloud to write to the sdc file.
 * 
 * @return True if the point cloud was written to the sdc file, false otherwise
 */
bool writePointCloudSdc(const std::filesystem::path& sdcFilePath, FullPointCloudAccessInterface& pointCloud);

/**
 * @brief write a sdc point cloud to a output stream
 * 
 * @param stream the stream to write to
 * @param pointCloud the point cloud to write
 * @return true if successful
 * @return false if failed
 * 
 */
bool writePointCloudSdc(std::ostream& stream, FullPointCloudAccessInterface& pointCloud);

} // namespace IO
} // namespace StereoVision


#endif //STEREOVISION_IO_SDCPOINTCLOUD_H
