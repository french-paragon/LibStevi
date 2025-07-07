#include "metacloud_io.h"
#include <iostream>
#include <map>
#include <regex>
#include <numeric>
#include <string>
#include <cmath>
#include <fstream>

namespace StereoVision {
namespace IO {

// map a string to a MetaCloudSimpleType
const std::map<std::string, MetaCloudSimpleType> MetaCloudSimpleTypeMap = {
    // unsigned
    {"uint8", MetaCloudSimpleType::uint8},
    {"uint16", MetaCloudSimpleType::uint16},
    {"uint32", MetaCloudSimpleType::uint32}, {"uint", MetaCloudSimpleType::uint32},
    {"uint64", MetaCloudSimpleType::uint64},
    // signed
    {"int8", MetaCloudSimpleType::int8},
    {"int16", MetaCloudSimpleType::int16},
    {"int32", MetaCloudSimpleType::int32}, {"int", MetaCloudSimpleType::int32},
    {"int64", MetaCloudSimpleType::int64},
    // floats
    {"float32", MetaCloudSimpleType::float32}, {"float", MetaCloudSimpleType::float32}, // same as float32
    {"float64", MetaCloudSimpleType::float64}, {"double", MetaCloudSimpleType::float64}, // same as float64
    // string
    {"string", MetaCloudSimpleType::string}
};

static std::optional<MetaCloudSimpleType> getMetaCloudSimpleType(PointCloudGenericAttribute const& attribute);

// get the size of a MetaCloudSimpleType. Since we cannot determine the size of a string, we return 0 for it.
static size_t sizeOfMetaCloudSimpleType(MetaCloudSimpleType type);

// cast a generic attribute to another generic attribute given a type
static std::optional<PointCloudGenericAttribute> castMetaCloudAttribute(const PointCloudGenericAttribute& attribute,
    MetaCloudSimpleType type);

// *----------------------- IMPLEMENTATION --------------------------------------

std::string to_string(MetaCloudSimpleType type) {
    switch (type) {
        case MetaCloudSimpleType::uint8: return "uint8";
        case MetaCloudSimpleType::uint16: return "uint16";
        case MetaCloudSimpleType::uint32: return "uint32";
        case MetaCloudSimpleType::uint64: return "uint64";
        case MetaCloudSimpleType::int8: return "int8";
        case MetaCloudSimpleType::int16: return "int16";
        case MetaCloudSimpleType::int32: return "int32";
        case MetaCloudSimpleType::int64: return "int64";
        case MetaCloudSimpleType::float32: return "float32";
        case MetaCloudSimpleType::float64: return "float64";
        case MetaCloudSimpleType::string: return "string";
        default: return "string";
    }
}

std::optional<FullPointCloudAccessInterface> openPointCloudMetacloud(const std::filesystem::path &metacloudFilePath) {

    // remove the file name
    auto metacloudFileDirectoryPath = metacloudFilePath;
    metacloudFileDirectoryPath.remove_filename();

    // open the file
    auto reader = std::make_unique<std::ifstream>();

    constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
    reader->precision(maxPrecision); // set the precision for the reader

    reader->open(metacloudFilePath, std::ios_base::binary);

    if (!reader->is_open()) return std::nullopt;

    return openPointCloudMetacloud(std::move(reader), metacloudFileDirectoryPath);
}

std::optional<FullPointCloudAccessInterface> openPointCloudMetacloud(std::unique_ptr<std::istream> reader,
    const std::filesystem::path& metacloudFileDirectoryPath) {
    // read the file header
    auto header = StereoVision::IO::MetaCloudHeader::readHeader(*reader, metacloudFileDirectoryPath);
    // test if header ptr is not null
    if (header == nullptr) {
        return std::nullopt;
    }

    auto& pointFilePaths = header->pointFilePaths;
    auto& extraAttributeDescriptors = header->extraAttributeDescriptors;

    // from the path to the point files, open the files
    std::vector<std::unique_ptr<FullPointCloudAccessInterface>> pointCloudInterfaces;
    std::vector<std::unique_ptr<MetaCloudExtraAttributeReader>> extraAttributeAccessors;

    for (auto&& pointCloudPath : pointFilePaths) {
        auto pointCloudOpt = openPointCloud(pointCloudPath);
        if (pointCloudOpt == std::nullopt) {
            return std::nullopt;
        }
        auto pointCloudPtr = std::make_unique<FullPointCloudAccessInterface>(std::move(*pointCloudOpt));
        pointCloudInterfaces.push_back(std::move(pointCloudPtr));
    }

    for (auto&& extraAttributeDescriptor : extraAttributeDescriptors) {
        auto& path = extraAttributeDescriptor.path;
        auto& attributeNames = extraAttributeDescriptor.attributeNames;
        auto& attributeTypes = extraAttributeDescriptor.attributeTypes;

        // open the file
        auto fileStream = std::make_unique<std::ifstream>();

        constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
        fileStream->precision(maxPrecision); // set the precision for the reader

        fileStream->open(path, std::ios_base::binary);

        if (!fileStream->is_open()) {
            return std::nullopt;
        }

        auto accessor = std::make_unique<MetaCloudExtraAttributeReader>(std::move(fileStream), attributeNames,
            attributeTypes);
        
        if (accessor == nullptr) {
            return std::nullopt;
        }

        extraAttributeAccessors.push_back(std::move(accessor));

    }

    // create a point cloud
    auto pointCloud = std::make_unique<AutoProcessCounterPointAccessInterface<MetaCloudPoint>>(std::move(pointCloudInterfaces),
        std::move(extraAttributeAccessors));

    if (pointCloud == nullptr) {
        return std::nullopt;
    }

    // return the point cloud full access interface
    FullPointCloudAccessInterface fullPointInterface;
    fullPointInterface.headerAccess = std::move(header);
    fullPointInterface.pointAccess = std::move(pointCloud);
    return fullPointInterface;

    return std::nullopt;
}

std::optional<MetaCloudSimpleType> getMetaCloudSimpleType(PointCloudGenericAttribute const &attribute) {
    // visit the variant
    return std::visit(
        [](auto&& arg) -> std::optional<MetaCloudSimpleType> {
            using T = std::decay_t<decltype(arg)>;
            if  constexpr (std::is_same_v<T, uint8_t>) {
                return MetaCloudSimpleType::uint8;
            } else if  constexpr (std::is_same_v<T, uint16_t>) {
                return MetaCloudSimpleType::uint16;
            } else if  constexpr (std::is_same_v<T, uint32_t>) {
                return MetaCloudSimpleType::uint32;
            } else if  constexpr (std::is_same_v<T, uint64_t>) {
                return MetaCloudSimpleType::uint64;
            } else if  constexpr (std::is_same_v<T, int8_t>) {
                return MetaCloudSimpleType::int8;
            } else if  constexpr (std::is_same_v<T, int16_t>) {
                return MetaCloudSimpleType::int16;
            } else if  constexpr (std::is_same_v<T, int32_t>) {
                return MetaCloudSimpleType::int32;
            } else if  constexpr (std::is_same_v<T, int64_t>) {
                return MetaCloudSimpleType::int64;
            } else if  constexpr (std::is_same_v<T, float>) {
                return MetaCloudSimpleType::float32;
            } else if  constexpr (std::is_same_v<T, double>) {
                return MetaCloudSimpleType::float64;
            } else if  constexpr (std::is_same_v<T, std::string>) {
                return MetaCloudSimpleType::string;
            } else {
                return std::nullopt;
            }
        },
        attribute);
}

size_t sizeOfMetaCloudSimpleType(MetaCloudSimpleType type) {
    if (type == MetaCloudSimpleType::uint8 || type == MetaCloudSimpleType::int8) {
        return 1;
    } else if (type == MetaCloudSimpleType::uint16 || type == MetaCloudSimpleType::int16) {
        return 2;
    } else if (type == MetaCloudSimpleType::uint32 || type == MetaCloudSimpleType::int32
            || type == MetaCloudSimpleType::float32) {
        
        return 4;
    } else if (type == MetaCloudSimpleType::uint64 || type == MetaCloudSimpleType::int64
            || type == MetaCloudSimpleType::float64) {
        return 8;
    } else if (type == MetaCloudSimpleType::string) {
        return 0; // cannot determine the size of a string
    } else {
        return 0; // should never happen
    }
}

std::optional<PointCloudGenericAttribute> castMetaCloudAttribute(const PointCloudGenericAttribute &attribute,
    MetaCloudSimpleType type) {
    
    // no need to do anything if the type is already correct
    auto attributeTypeOpt = getMetaCloudSimpleType(attribute);
    if (attributeTypeOpt.has_value() && attributeTypeOpt.value() == type) {
        return attribute;
    }

    // cast to the correct type
    switch (type) {
        case MetaCloudSimpleType::uint8:
            return castedPointCloudAttribute<uint8_t>(attribute);
        case MetaCloudSimpleType::uint16:
            return castedPointCloudAttribute<uint16_t>(attribute);
        case MetaCloudSimpleType::uint32:
            return castedPointCloudAttribute<uint32_t>(attribute);
        case MetaCloudSimpleType::uint64:
            return castedPointCloudAttribute<uint64_t>(attribute);
        case MetaCloudSimpleType::int8:
            return castedPointCloudAttribute<int8_t>(attribute);
        case MetaCloudSimpleType::int16:
            return castedPointCloudAttribute<int16_t>(attribute);
        case MetaCloudSimpleType::int32:
            return castedPointCloudAttribute<int32_t>(attribute);
        case MetaCloudSimpleType::int64:
            return castedPointCloudAttribute<int64_t>(attribute);
        case MetaCloudSimpleType::float32:
            return castedPointCloudAttribute<float>(attribute);
        case MetaCloudSimpleType::float64:
            return castedPointCloudAttribute<double>(attribute);
        case MetaCloudSimpleType::string:
            return castedPointCloudAttribute<std::string>(attribute);
        default:
            return std::nullopt;
    }
}

std::optional<PointCloudGenericAttribute> MetaCloudHeader::getAttributeById(int id) const {
    if (id < 0 || id >= headerAttributeNames.size()) return std::nullopt;
    return headerAttributeValues[id];
}

std::optional<PointCloudGenericAttribute> MetaCloudHeader::getAttributeByName(const char *attributeName) const {
    // find the index of the attribute
    auto it = std::find(headerAttributeNames.begin(), headerAttributeNames.end(), attributeName);
    if (it == headerAttributeNames.end()) return std::nullopt;
    return getAttributeById(std::distance(headerAttributeNames.begin(), it));
}

std::vector<std::string> MetaCloudHeader::attributeList() const {
    return headerAttributeNames;
}

std::unique_ptr<MetaCloudHeader> MetaCloudHeader::readHeader(std::istream &reader,
    const std::filesystem::path& metacloudFileDirectoryPath) {

    if (!reader.good()) return nullptr;

    // paths to the point files
    std::vector<std::filesystem::path> pointFilePaths = {};
    std::vector<MetaCloudHeaderExtraAttributeDescriptor> extraAttributeDescriptors = {};
    std::vector<MetaCloudHeaderIndexFileDescriptor> indexFileDescriptors = {};
    std::vector<std::string> headerAttributeNames = {};
    std::vector<MetaCloudSimpleType> headerAttributeTypes = {};
    std::vector<PointCloudGenericAttribute> headerAttributeValues = {};

    std::string line;
    std::string currentTag;
    bool isTagValid = false;

    // lambda function that adapt the path with respect to the metacloud file directory path if the path is relative
    auto adaptPath = [&metacloudFileDirectoryPath](std::filesystem::path path) {
        if (path.is_relative() && !metacloudFileDirectoryPath.empty()) {
            return metacloudFileDirectoryPath / path;
        } else {
            return path;
        }
    };

    while(reader >> ignoreCommentsAndWs && std::getline(reader, line)) {
        auto [tokens, wasQuoted] = getUnquotedTokens(line);
        if (isLineHeaderTag(tokens, wasQuoted)) {
            currentTag = tokens[0];
            isTagValid = true;
        } else {
            if (isTagValid) {
                // * ################# POINTS_FILES (mandatory) #######################
                if (currentTag == "POINTS_FILES") {
                    // TODO
                    auto [tokens, wasQuoted] = getUnquotedTokens(line);
                    for (int i = 0; i < tokens.size(); i++) {
                        pointFilePaths.push_back(adaptPath(tokens[i]));
                    }
                // * ################# EXTRA_ATTRIBUTES (optional) ####################
                } else if (currentTag == "EXTRA_ATTRIBUTES") {
                    auto [tokens, wasQuoted] = getUnquotedTokens(line);
                    if (tokens.size() >= 3 && tokens.size() % 2 == 1) { // path + same number of names and types
                        MetaCloudHeaderExtraAttributeDescriptor descriptor;
                        descriptor.path = adaptPath(tokens[0]);
                        auto nbAttributes = (tokens.size()-1) / 2;
                        for (int i = 0; i < nbAttributes; i++) {
                            // try to map to the simple type
                            try {
                                auto type = MetaCloudSimpleTypeMap.at(tokens[1+i*2]);
                                if (type == MetaCloudSimpleType::string) return nullptr; // string is not supported for extra types
                                
                                descriptor.attributeNames.push_back(tokens[2+i*2]);
                                descriptor.attributeTypes.push_back(type);

                            } catch (...) {
                                return nullptr; // invalid type
                            }
                        }
                        extraAttributeDescriptors.push_back(descriptor);
                    } else {
                        return nullptr; // invalid extra attributes
                    }
                // * ################# INDEX_FILES (optional) #########################
                } else if (currentTag == "INDEX_FILES") {
                    auto [tokens, wasQuoted] = getUnquotedTokens(line);
                    if (tokens.size() != 3) return nullptr; // invalid entry
                    MetaCloudHeaderIndexFileDescriptor descriptor;
                    descriptor.path = adaptPath(tokens[0]);
                    descriptor.type = tokens[1];
                    descriptor.name = tokens[2];
                    indexFileDescriptors.push_back(descriptor);
                // * ################# METACLOUD_ATTRIBUTES (optional) ################
                } else if (currentTag == "METACLOUD_ATTRIBUTES") {
                    auto [tokens, wasQuoted] = getUnquotedTokens(line);
                    if (tokens.size() != 2) return nullptr; // invalid entry
                    headerAttributeNames.push_back(tokens[0]);
                    // find the type using regex
                    std::regex intRegex(R"(^[+-]?[0-9]+$)");
                    std::regex floatRegex(R"(^[+-]?([0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?|[.][0-9]+([eE][+-]?[0-9]+)?)$)");
                    MetaCloudSimpleType type;
                    std::stringstream ss{tokens[1]};
                    if (!wasQuoted[1] && std::regex_match(tokens[1], intRegex)) { // integer
                        headerAttributeTypes.push_back(MetaCloudSimpleType::int64);
                        int64_t value;
                        ss >> value;
                        headerAttributeValues.push_back(value);
                    } else if (!wasQuoted[1] && std::regex_match(tokens[1], floatRegex)) { // float
                        headerAttributeTypes.push_back(MetaCloudSimpleType::float64);
                        double value;
                        ss >> value;
                        headerAttributeValues.push_back(value);
                    } else { // string
                        headerAttributeTypes.push_back(MetaCloudSimpleType::string);
                        headerAttributeValues.push_back(tokens[1]);
                    }
                }
            } else {
                break;
            }
        }
    }

    reader >> ignoreCommentsAndWs;
    if (!reader.eof()) return nullptr; // we should be at the end of the file

    // create the header
    auto header = std::make_unique<MetaCloudHeader>();
    
    header->pointFilePaths = pointFilePaths;
    header->extraAttributeDescriptors = extraAttributeDescriptors;
    header->indexFileDescriptors = indexFileDescriptors;
    header->headerAttributeNames = headerAttributeNames;
    header->headerAttributeTypes = headerAttributeTypes;
    header->headerAttributeValues = headerAttributeValues;
    
    return header;
}

std::istream& MetaCloudHeader::ignoreCommentsAndWs(std::istream& in) {
    while (in){
        in >> std::ws;
        // ignore comments
        if (in.peek() == '#') {
            // skip the rest of the line
            std::string line;
            std::getline(in, line);
        } else {
            break;
        }
    }
    return in;
}

std::tuple<std::vector<std::string>, std::vector<bool>> MetaCloudHeader::getUnquotedTokens(std::string &line) {
    std::stringstream ss{line};

    std::vector<std::string> tokenList;
    std::vector<bool> wasQuotedList;
    
    while(ss) {
        ss >> ignoreCommentsAndWs;
        std::string token;
        bool isQuoted = false;
        if (ss.peek() == '"') {
            ss >> std::quoted(token);
            isQuoted = true;
        } else {
            ss >> token;
            // ignore comments
            if (auto pos = token.find("#"); pos != std::string::npos) {
                token.resize(pos);
                if (!token.empty()) {
                    tokenList.push_back(token);
                    wasQuotedList.push_back(isQuoted);
                }
                break;
            }
        }
        if (!token.empty()) {
            tokenList.push_back(token);
            wasQuotedList.push_back(isQuoted);
        } else {
            break;
        }
    }

    return {tokenList, wasQuotedList};
}

bool MetaCloudHeader::isLineHeaderTag(const std::vector<std::string>& tokens, const std::vector<bool>& wasQuoted) {
    return tokens.size() == 1 &&
           wasQuoted.size() == 1 &&
           wasQuoted[0] == false &&
           (tokens[0] == "POINTS_FILES" ||
            tokens[0] == "EXTRA_ATTRIBUTES" ||
            tokens[0] == "INDEX_FILES" ||
            tokens[0] == "METACLOUD_ATTRIBUTES");

}

MetaCloudExtraAttributeReader::MetaCloudExtraAttributeReader(std::unique_ptr<std::istream> reader,
    const std::vector<std::string> &attributeNames, const std::vector<MetaCloudSimpleType> &attributeTypes) :
        reader{std::move(reader)}, attributeNames{attributeNames}, attributeTypes{attributeTypes} {

    for (auto type : attributeTypes) {
        size_t size = sizeOfMetaCloudSimpleType(type);
        if (size == 0) break; // we cannot determine the size of the element... (a string)
        attributeSizes.push_back(size);
    }
    // offsets are the cumulative sum of the sizes
    attributeOffsets.resize(attributeSizes.size());
    std::exclusive_scan(attributeSizes.begin(), attributeSizes.end(), attributeOffsets.begin(), size_t{0});

    if (attributeOffsets.size() != 0) {
        recordByteSize = attributeOffsets.back() + attributeSizes.back();
    }

    recordBuffer.resize(recordByteSize);
    this->attributeNames.resize(attributeSizes.size());
    attributeValues.resize(attributeSizes.size());

    // default values for attribute Values
    for (size_t i = 0; i < attributeValues.size(); i++) {
        attributeValues[i] = castMetaCloudAttribute(EmptyParam{}, attributeTypes[i]).value_or(EmptyParam{});
    }

    // first call to gotoNext
    gotoNext();
}

PtGeometry<PointCloudGenericAttribute> MetaCloudExtraAttributeReader::getPointPosition() const {
        return {std::nan(""), std::nan(""), std::nan("")};
}

std::optional<PointCloudGenericAttribute> MetaCloudExtraAttributeReader::getAttributeById(int id) const {
    if (id < 0 || id >= attributeNames.size()) return std::nullopt;
    return attributeValues[id];
}

std::optional<PointCloudGenericAttribute> MetaCloudExtraAttributeReader::getAttributeByName(
    const char *attributeName) const {
    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    if (it == attributeNames.end()) return std::nullopt;
    return getAttributeById(std::distance(attributeNames.begin(), it));
}

bool MetaCloudExtraAttributeReader::gotoNext() {
    if (reader != nullptr && reader->good()) {
        reader->read(reinterpret_cast<char*>(recordBuffer.data()), recordByteSize);
        for (size_t i = 0; i < attributeSizes.size(); i++) {
            auto type = attributeTypes[i];
            auto size = attributeSizes[i];
            auto offset = attributeOffsets[i];
            auto dataPtr = recordBuffer.data() + offset;
            PointCloudGenericAttribute attributeValue;
            switch (type) {
                case MetaCloudSimpleType::uint8:
                    attributeValue = fromBytes<uint8_t>(dataPtr);
                    break;
                case MetaCloudSimpleType::uint16:
                    attributeValue = fromBytes<uint16_t>(dataPtr);
                    break;
                case MetaCloudSimpleType::uint32:
                    attributeValue = fromBytes<uint32_t>(dataPtr);
                    break;
                case MetaCloudSimpleType::uint64:
                    attributeValue = fromBytes<uint64_t>(dataPtr);
                    break;
                case MetaCloudSimpleType::int8:
                    attributeValue = fromBytes<int8_t>(dataPtr);
                    break;
                case MetaCloudSimpleType::int16:
                    attributeValue = fromBytes<int16_t>(dataPtr);
                    break;
                case MetaCloudSimpleType::int32:
                    attributeValue = fromBytes<int32_t>(dataPtr);
                    break;
                case MetaCloudSimpleType::int64:
                    attributeValue = fromBytes<int64_t>(dataPtr);
                    break;
                case MetaCloudSimpleType::float32:
                    attributeValue = fromBytes<float>(dataPtr);
                    break;
                case MetaCloudSimpleType::float64:
                    attributeValue = fromBytes<double>(dataPtr);
                    break;
                case MetaCloudSimpleType::string:
                    {
                        auto vChar = vectorFromBytes<char>(dataPtr, size);
                        attributeValue = std::string{vChar.begin(), vChar.end()};
                    }
                    break;
                default:
                    return false; // should never happen
            }
            attributeValues[i] = attributeValue;
        }
        return true;
    } else {
        return false;
    }
}

bool MetaCloudExtraAttributeReader::hasData() const {
    if (reader == nullptr) return false;
    return reader->good();
}

MetaCloudPoint::MetaCloudPoint(std::vector<std::unique_ptr<FullPointCloudAccessInterface>> && pointCloudInterfaces_,
        std::vector<std::unique_ptr<MetaCloudExtraAttributeReader>> && extraAttributeAccessors_) :
        pointCloudInterfaces{std::move(pointCloudInterfaces_)},
        extraAttributeAccessors{std::move(extraAttributeAccessors_)} {

    // lambda
    auto setAttributesFromInterface = [&](
            const auto& pointCloudPointInterface,
            bool isExtraAttributeAccessor = false,
            size_t extraAttributeAccessorId = std::numeric_limits<size_t>::max()) {

        for (auto& attributeName : pointCloudPointInterface->attributeList()) {
            // test if the attribute name is not already in the list
            if (std::find(attributeNames.begin(), attributeNames.end(), attributeName) == attributeNames.end()) {
                // try to get the type of the attribute
                auto attributeValueOpt = pointCloudPointInterface->getAttributeByName(attributeName.c_str());
                if (attributeValueOpt.has_value()) {
                    auto typeOpt = getMetaCloudSimpleType(attributeValueOpt.value());
                    if (typeOpt.has_value()) {
                        attributeNames.push_back(attributeName);
                        attributeTypes.push_back(typeOpt.value());
                        isExtraAttribute.push_back(isExtraAttributeAccessor);
                        attributeIdToExtraAttributeAccessor.push_back(extraAttributeAccessorId);
                    }
                }
            }
        }
    };

    // try to get the attribute names and types from the first point cloud point access interface
    if (pointCloudInterfaces.size() > 0 && pointCloudInterfaces[0] != nullptr
            && pointCloudInterfaces[0]->pointAccess != nullptr) {

        setAttributesFromInterface(pointCloudInterfaces[0]->pointAccess);
    }

    // same thing for the extra attributes descriptors

    for (size_t i = 0; i < extraAttributeAccessors.size(); i++) {
        if (extraAttributeAccessors[i] != nullptr) {
            setAttributesFromInterface(extraAttributeAccessors[i].get(), true, i);
        }
    }
}

PtGeometry<PointCloudGenericAttribute> MetaCloudPoint::getPointPosition() const
{
    if (currentPointCloud >= pointCloudInterfaces.size() || pointCloudInterfaces[currentPointCloud] == nullptr
        || pointCloudInterfaces[currentPointCloud]->pointAccess == nullptr) {
        return {std::nan(""), std::nan(""), std::nan("")};
    }

    return pointCloudInterfaces[currentPointCloud]->pointAccess->getPointPosition();
}

std::optional<PtColor<PointCloudGenericAttribute>> MetaCloudPoint::getPointColor() const {
    if (currentPointCloud >= pointCloudInterfaces.size() || pointCloudInterfaces[currentPointCloud] == nullptr
        || pointCloudInterfaces[currentPointCloud]->pointAccess == nullptr) {
        return std::nullopt;
    }
            
    return pointCloudInterfaces[currentPointCloud]->pointAccess->getPointColor();
}

std::optional<PointCloudGenericAttribute> MetaCloudPoint::getAttributeById(int id) const {

    std::optional<PointCloudGenericAttribute> attribute = std::nullopt;

    if (id >= 0 && id < attributeNames.size()) {
        if (isExtraAttribute[id]) {
            auto extraAttributeAccessorId = attributeIdToExtraAttributeAccessor[id];
            if (extraAttributeAccessorId < extraAttributeAccessors.size()
                && extraAttributeAccessors[extraAttributeAccessorId] != nullptr) {
                
                attribute = extraAttributeAccessors[extraAttributeAccessorId]
                    ->getAttributeByName(attributeNames[id].c_str());
            }
        } else if (currentPointCloud < pointCloudInterfaces.size() && pointCloudInterfaces[currentPointCloud] != nullptr
                    && pointCloudInterfaces[currentPointCloud]->pointAccess != nullptr) {
                
            attribute = pointCloudInterfaces[currentPointCloud]->pointAccess
                ->getAttributeByName(attributeNames[id].c_str());
        }
    }
    // force the cast to the correct type
    return castMetaCloudAttribute(attribute.value_or(EmptyParam{}), attributeTypes[id]);
}

std::optional<PointCloudGenericAttribute> MetaCloudPoint::getAttributeByName(const char *attributeName) const {
    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    if (it == attributeNames.end()) return std::nullopt;
    return getAttributeById(std::distance(attributeNames.begin(), it));
}

int MetaCloudPoint::expectedNumberOfPoints() const {

    int count = 0;

    for (std::unique_ptr<FullPointCloudAccessInterface> const& pointsAccessInterfaces : pointCloudInterfaces) {

        if (pointsAccessInterfaces.get() == nullptr) {
            return -1;
        }

        int expectedCount = pointsAccessInterfaces->expectedNumberOfPoints();

        if (expectedCount < 0) {
            return -1;
        }

        count += expectedCount;

    }

    return count;

}

bool MetaCloudPoint::gotoNext() {
    if (currentPointCloud >= pointCloudInterfaces.size() || pointCloudInterfaces[currentPointCloud] == nullptr
        || pointCloudInterfaces[currentPointCloud]->pointAccess == nullptr) {
        return false;
    }

    // try gotoNext with the current point cloud, otherwise switch to the next one
    if (!pointCloudInterfaces[currentPointCloud]->pointAccess->gotoNext()) {
        // try to switch to the next point cloud
        currentPointCloud++;
        if (currentPointCloud >= pointCloudInterfaces.size() || pointCloudInterfaces[currentPointCloud] == nullptr
            || pointCloudInterfaces[currentPointCloud]->pointAccess == nullptr) {
            return false;
        }
    }

    // gotoNext for all extra attribute accessors
    for (size_t i = 0; i < extraAttributeAccessors.size(); i++) {
        if (extraAttributeAccessors[i] == nullptr || !extraAttributeAccessors[i]->gotoNext()) {
            return false;
        }
    }

    return true;
}
bool MetaCloudPoint::hasData() const {

    if (currentPointCloud >= pointCloudInterfaces.size() || pointCloudInterfaces[currentPointCloud] == nullptr
        || pointCloudInterfaces[currentPointCloud]->pointAccess == nullptr) {
        return false;
    }

    if (currentPointCloud == pointCloudInterfaces.size() - 1) {
        return pointCloudInterfaces[currentPointCloud]->pointAccess->hasData();
    }

    return true;
}

}
}
