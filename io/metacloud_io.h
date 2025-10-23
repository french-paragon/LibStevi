#ifndef STEREOVISION_IO_METACLOUD_H
#define STEREOVISION_IO_METACLOUD_H

#include "pointcloud_io.h"
#include "bit_manipulations.h"

namespace StereoVision {
namespace IO {

enum class MetaCloudSimpleType {
    // unsigned
    uint8,
    uint16,
    uint32,
    uint64,
    // signed
    int8,
    int16,
    int32,
    int64,
    // floating point
    float32,
    float64,
    // string, only allowed for the metacloud attributes
    string
};

// to string for the MetaCloudSimpleType
std::string to_string(MetaCloudSimpleType type);

struct MetaCloudHeaderExtraAttributeDescriptor {
    std::filesystem::path path; // path to the extra attribute file
    std::vector<std::string> attributeNames; // names of the extra attributes
    std::vector<MetaCloudSimpleType> attributeTypes; // types of the extra attributes
};

struct MetaCloudHeaderIndexFileDescriptor {
    std::filesystem::path path; // path to the index file
    std::string type; // type of the index method
    std::string name; // name of the index method
};

// metacloud header
class MetaCloudHeader : public PointCloudHeaderInterface {
public:
    // paths to the point files
    std::vector<std::filesystem::path> pointFilePaths = {};

    // extra attributes
    std::vector<MetaCloudHeaderExtraAttributeDescriptor> extraAttributeDescriptors = {};

    // indices files for spatial indexing
    std::vector<MetaCloudHeaderIndexFileDescriptor> indexFileDescriptors = {};

    // metacloud attributes
    std::vector<std::string> headerAttributeNames = {};
    std::vector<MetaCloudSimpleType> headerAttributeTypes = {};
    std::vector<PointCloudGenericAttribute> headerAttributeValues = {};

public:
    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    static std::unique_ptr<MetaCloudHeader> readHeader(std::istream& reader,
        const std::filesystem::path& metacloudFileFolderPath = std::filesystem::path{});

protected:
    // skips comments and whitespace in the header file
    static std::istream& ignoreCommentsAndWs(std::istream& in);
    // process a line of the metacloud header and return the tokens + whether they were quoted or not (for strings)
    static std::tuple<std::vector<std::string>, std::vector<bool>> getUnquotedTokens(std::string& line);
    // checks if the line is a tag
    static bool isLineHeaderTag(const std::vector<std::string>& tokens, const std::vector<bool>& wasQuoted);

};

/// @brief access interface to access the extra attributes of the metacloud
class MetaCloudExtraAttributeReader : public PointCloudPointAccessInterface {
    std::vector<std::string> attributeNames; // names of the extra attributes
    std::vector<MetaCloudSimpleType> attributeTypes; // types of the extra attributes
    std::vector<size_t> attributeOffsets;
    std::vector<size_t> attributeSizes;

    std::unique_ptr<std::istream> reader;
    size_t recordByteSize;
    std::vector<std::byte> recordBuffer;
    std::vector<PointCloudGenericAttribute> attributeValues;
public:
    MetaCloudExtraAttributeReader(std::unique_ptr<std::istream> reader, const std::vector<std::string>& attributeNames,
        const std::vector<MetaCloudSimpleType>& attributeTypes);

    // getPointPosition and getPointColor should not be used for extra attributes
    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;
    // getPointPosition and getPointColor should not be used for extra attributes
    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override { return std::nullopt; }

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override { return attributeNames; }

    bool gotoNext() override;
    bool hasData() const override;

    static std::unique_ptr<MetaCloudExtraAttributeReader> create(std::unique_ptr<std::istream> reader,
        const std::vector<std::string>& attributeNames, const std::vector<MetaCloudSimpleType>& attributeTypes);

};

class MetaCloudPoint : public PointCloudPointAccessInterface {
    std::vector<std::unique_ptr<FullPointCloudAccessInterface>> pointCloudInterfaces;
    std::vector<std::unique_ptr<MetaCloudExtraAttributeReader>> extraAttributeAccessors;
    // TODO: accessors for index files

    std::vector<std::string> attributeNames;
    std::vector<MetaCloudSimpleType> attributeTypes;
    std::vector<bool> isExtraAttribute;
    size_t currentPointCloud = 0; // index of the current point cloud we are in.
    // map the id of an attribute to its corresponding extra attribute accessor if it is an extra attribute.
    std::vector<size_t> attributeIdToExtraAttributeAccessor;

public:
    MetaCloudPoint(std::vector<std::unique_ptr<FullPointCloudAccessInterface>>&& pointCloudInterfaces,
        std::vector<std::unique_ptr<MetaCloudExtraAttributeReader>>&& extraAttributeAccessors);
        
    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override { return attributeNames; }

    virtual int64_t expectedNumberOfPoints() const override;

    bool gotoNext() override;
    bool hasData() const override;
};

/**
 * @brief
 *
 * Open a point cloud from a metacloud file and returns a FullPointCloudAccessInterface
 * containing the header and the points.
 *
 * @param metacloudFilePath The path to the metacloud file containing the point cloud
 *
 * @return A FullPointCloudAccessInterface containing the header and the points.
 *         If the file can't be opened, an error message is returned
 */
StatusOptional<FullPointCloudAccessInterface> openPointCloudMetacloud(const std::filesystem::path& metacloudFilePath);

/**
 * @brief
 * 
 * Open a metacloud point cloud from a stream and returns a FullPointCloudAccessInterface
 * containing the header and the points.
 * 
 * @param reader The stream to open the point cloud from
 * @param metacloudFileFolderPath The path to the folder containing the metacloud file. Only used for relative paths.
 * By default, the path is empty.
 * 
 * @return A FullPointCloudAccessInterface containing the header and the points.
 *         If the stream can't be opened, an error message is returned
 * 
 */
StatusOptional<FullPointCloudAccessInterface> openPointCloudMetacloud(std::unique_ptr<std::istream> reader,
    const std::filesystem::path& metacloudFileFolderPath = std::filesystem::path{});

}
}

#endif //STEREOVISION_IO_METACLOUD_H
