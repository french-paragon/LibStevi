#ifndef STEREOVISION_IO_METACLOUD_H
#define STEREOVISION_IO_METACLOUD_H

#include "pointcloud_io.h"
#include "bit_manipulations.h"
#include <map>

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

    static std::unique_ptr<MetaCloudHeader> readHeader(std::istream& reader);

protected:
    // skips comments and whitespace in the header file
    static std::istream& ignoreCommentsAndWs(std::istream& in);
    // process a line of the metacloud header and return the tokens + whether they were quoted or not (for strings)
    static std::tuple<std::vector<std::string>, std::vector<bool>> getUnquotedTokens(std::string& line);
    // checks if the line is a tag
    static bool isLineHeaderTag(const std::vector<std::string>& tokens, const std::vector<bool>& wasQuoted);

};

}
}

#endif //STEREOVISION_IO_METACLOUD_H
