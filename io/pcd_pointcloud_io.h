#ifndef STEREOVISION_IO_PCDPOINTCLOUD_H
#define STEREOVISION_IO_PCDPOINTCLOUD_H

#include "pointcloud_io.h"
#include <filesystem>
#include <array>

namespace StereoVision
{
namespace IO
{

enum class PcdDataStorageType
{
    ascii,
    binary,
    binary_compressed
};

/// @brief basic class for a point in a pcd file
class PcdPointCloudPoint : public PointCloudPointAccessInterface
{
protected:
    const std::vector<std::string> attributeNames; // names of the fields
    const std::vector<size_t> fieldByteSize;
    std::vector<size_t> fieldOffset; // in bytes
    const std::vector<uint8_t> fieldType; // can be 'F' (floating point), 'I' (signed integer), 'U' (unsigned integer)
    const std::vector<size_t> fieldCount;
    PcdDataStorageType dataStorageType;

    bool containsColor = false; // true if the point cloud has color
    bool containsPosition = false; // true if the point cloud has position
    // indices for the points position and color
    int rgbaIndex;
    int xIndex;
    int yIndex;
    int zIndex;

    size_t recordByteSize; // number of bytes in a pcd point record

    std::vector<std::byte> dataBuffer;

    // reader
    const std::unique_ptr<std::istream> reader;
public:
    PcdPointCloudPoint(std::unique_ptr<std::istream> reader, const std::vector<std::string>& attributeNames,
        const std::vector<size_t>& fieldByteSize, const std::vector<uint8_t>& fieldType,
        const std::vector<size_t>& fieldCount, PcdDataStorageType dataStorageType);
    
    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    bool gotoNext() override;
    
    // getters

    inline auto getFieldByteSize() const { return fieldByteSize; }
    inline auto getFieldType() const { return fieldType; }
    inline auto getFieldCount() const { return fieldCount; }

    static bool writePoint(std::ostream& writer, const PcdPointCloudPoint& point, PcdDataStorageType dataStorageType);
    static bool writePointAscii(std::ostream& writer, const PcdPointCloudPoint& point);
    static bool writePointBinary(std::ostream& writer, const PcdPointCloudPoint& point);

    /**
     * @brief Obtain an adapter to a PcdPointCloudPoint from any PointCloudPointAccessInterface. The adapted interface
     * can be the given interface if the object is already a PcdPointCloudPoint or a wrapper otherwise.
     * If the given interface is null, a nullptr is returned.
     * 
     * @param pointCloudPointAccessInterface the interface to adapt
     * @return a shared_ptr to the adapted interface
     */
    static std::shared_ptr<PcdPointCloudPoint> createAdapter(PointCloudPointAccessInterface* pointCloudPointAccessInterface);

    // destructor
    ~PcdPointCloudPoint() override;
private:
    bool gotoNextAscii();
    bool gotoNextBinary();
    bool gotoNextBinaryCompressed();
};

/// @brief basic class for a header in a pcd file
class PcdPointCloudHeader : public PointCloudHeaderInterface
{
public:
    double version = 0.7;
    std::vector<std::string> fields = {};
    std::vector<size_t> size = {};
    std::vector<uint8_t> type = {};
    std::vector<size_t> count = {};
    size_t width = 0;
    size_t height = 0;
    std::vector<double> viewpoint = {0, 0, 0, 1, 0, 0, 0};
    size_t points = 0;
    PcdDataStorageType data = PcdDataStorageType::ascii;

protected:
    // attribute names for the header
    std::vector<std::string> attributeNames = {"version", "fields", "size", "type", "count", "width", "height", "viewpoint", "points", "data"};
public:
    // constructor
    PcdPointCloudHeader();

    PcdPointCloudHeader(const double version, const std::vector<std::string>& fields,
        const std::vector<size_t>& size, const std::vector<uint8_t>& type, const std::vector<size_t>& count,
        const size_t width, const size_t height, const std::vector<double>& viewpoint, const size_t points,
        const PcdDataStorageType dataStorageType);

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    static std::unique_ptr<PcdPointCloudHeader> readHeader(std::istream& reader);

    static bool getNextHeaderLine(std::istream& reader, std::string& line, std::vector<std::string>& lineSplit, std::stringstream& lineStream);

    /**
     * @brief Write the header of the pcd file. It also obtains the position of the fields width, height and points.
     * 
     * @param writer The writer to use
     * @param header The header interface to write
     * @param headerWidthPos The position of the width field in the stream.
     * @param headerHeightPos The position of the heigh field in the stream.
     * @param headerPointsPos The position of the points field in the stream.
     * @return true is success.
     * @return false if it fails to write the data.
     */
    static bool writeHeader(std::ostream& writer, const PcdPointCloudHeader& header, 
                               std::streampos& headerWidthPos, std::streampos& headerHeightPos, std::streampos& headerPointsPos);

    /**
     * @brief Obtain an adapter to a PcdPointCloudHeader from any PointCloudHeaderInterface. The adapted interface
     * can be the given interface if the object is already a PcdPointCloudHeader or a wrapper otherwise.
     * If the given interface is null, a nullptr is returned.
     * 
     * @param pointCloudHeaderInterface the interface to adapt
     * @return a shared_ptr to the adapted interface
     */
    static std::shared_ptr<PcdPointCloudHeader> createAdapter(PointCloudHeaderInterface* pointCloudHeaderInterface);
};

/**
 * @brief
 *
 * Open a point cloud from a pcd file and returns a FullPointCloudAccessInterface
 * containing the header and the points.
 *
 * @param pcdFilePath The path to the pcd file containing the point cloud
 *
 * @return A FullPointCloudAccessInterface containing the header and the points.
 *         If the file can't be opened, an empty optional is returned
 */
std::optional<FullPointCloudAccessInterface> openPointCloudPcd(const std::filesystem::path& pcdFilePath);

/**
 * @brief
 * 
 * Write a point cloud to a pcd file.
 * 
 * @param pcdFilePath The path to the pcd file to write.
 * @param pointCloud The point cloud to write to the pcd file.
 * @param dataStorageType The data storage type to use. If not specified, the data storage type defined in the header will be used
 * 
 * @return True if the point cloud was written to the pcd file, false otherwise
 */
bool writePointCloudPcd(const std::filesystem::path& pcdFilePath, FullPointCloudAccessInterface& pointCloud,
    std::optional<PcdDataStorageType> dataStorageType = std::nullopt);

} // namespace IO
} // namespace StereoVision


#endif //STEREOVISION_IO_PCDPOINTCLOUD_H