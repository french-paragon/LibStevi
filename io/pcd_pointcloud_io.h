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

class PcdPointCloudPoint : public PointCloudPointAccessInterface
{
private:
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
    const std::unique_ptr<std::ifstream> reader;
public:
    PcdPointCloudPoint(std::unique_ptr<std::ifstream> reader, const std::vector<std::string>& attributeNames,
        const std::vector<size_t>& fieldByteSize, const std::vector<uint8_t>& fieldType,
        const std::vector<size_t>& fieldCount, PcdDataStorageType dataStorageType);
    
    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    bool gotoNext() override;
    
    // destructor
    ~PcdPointCloudPoint() override;
private:
    bool gotoNextAscii();
    bool gotoNextBinary();
    bool gotoNextBinaryCompressed();
};

class PcdPointCloudHeader : public PointCloudHeaderInterface
{
private:
    //*Friend fuctions
    friend std::optional<FullPointCloudAccessInterface> openPointCloudPcd(const std::filesystem::path& pcdFilePath);

    const double version;
    const std::vector<std::string> fields = {};
    const std::vector<size_t> size;
    const std::vector<uint8_t> type;
    const std::vector<size_t> count;
    const size_t width;
    const size_t height;
    const std::vector<double> viewpoint;
    const size_t points;
    const PcdDataStorageType data;

    // attribute names for the header
    const std::vector<std::string> attributeNames = {"version", "fields", "size", "type", "count", "width", "height", "viewpoint", "points", "data"};
public:
    // constructor
    PcdPointCloudHeader(const double version, const std::vector<std::string>& fields,
        const std::vector<size_t>& size, const std::vector<uint8_t>& type, const std::vector<size_t>& count,
        const size_t width, const size_t height, const std::vector<double>& viewpoint, const size_t points,
        const PcdDataStorageType dataStorageType);

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;
};


/**
 * @brief Open a point cloud from a pcd file
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

} // namespace IO
} // namespace StereoVision


#endif //STEREOVISION_IO_PCDPOINTCLOUD_H