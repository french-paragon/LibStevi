#include "metacloud_io.h"
#include <iostream>
#include <regex>

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

std::optional<MetaCloudSimpleType> getMetaCloudSimpleType(PointCloudGenericAttribute const& attribute);

// cast a generic attribute to another generic attribute given a type
std::optional<PointCloudGenericAttribute> castMetaCloudAttribute(const PointCloudGenericAttribute& attribute, MetaCloudSimpleType type);

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

// ----------------------- IMPLEMENTATION --------------------------------------

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

std::optional<PointCloudGenericAttribute> castMetaCloudAttribute(const PointCloudGenericAttribute &attribute, MetaCloudSimpleType type) {
    
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
    return getAttributeById(std::distance(headerAttributeNames.begin(), it));
}

std::vector<std::string> MetaCloudHeader::attributeList() const {
    return headerAttributeNames;
}

std::unique_ptr<MetaCloudHeader> MetaCloudHeader::readHeader(std::istream &reader) {
    
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
                        pointFilePaths.push_back(std::filesystem::path(tokens[i]));
                    }
                // * ################# EXTRA_ATTRIBUTES (optional) ####################
                } else if (currentTag == "EXTRA_ATTRIBUTES") {
                    auto [tokens, wasQuoted] = getUnquotedTokens(line);
                    if (tokens.size() >= 3 && tokens.size() % 2 == 1) { // path + same number of names and types
                        MetaCloudHeaderExtraAttributeDescriptor descriptor;
                        descriptor.path = tokens[0];
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
                    descriptor.path = tokens[0];
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


}
}
