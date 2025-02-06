#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <string>
#include "io/metacloud_io.h"


int main(int argc, char const *argv[]) {

    std::filesystem::path path;

    if (argc > 1) {
        path = argv[1];
    }

    // open the file
    std::ifstream reader{path};

    if (!reader.is_open()) {
        std::cerr << "Could not open file " << path << std::endl;
        return 1;
    }

    // read the file header
    auto header = StereoVision::IO::MetaCloudHeader::readHeader(reader);

    if (header == nullptr) {
        std::cerr << "Could not read file " << path << std::endl;
        return 1;
    }

    auto& pointFilePaths = header->pointFilePaths;
    auto& extraAttributeDescriptors = header->extraAttributeDescriptors;
    auto& indexFileDescriptors = header->indexFileDescriptors;
    auto& headerAttributeNames = header->headerAttributeNames;
    auto& headerAttributeTypes = header->headerAttributeTypes;
    auto& headerAttributeValues = header->headerAttributeValues;

    // print pointFilePaths
    std::cout << "Point file paths: ";
    for (const auto& path : pointFilePaths) {
        std::cout << path << ", ";
    }
    std::cout << std::endl;

    // print extraAttributeDescriptors
    std::cout << "Extra attribute descriptors: " << std::endl;
    for (const auto& descriptor : extraAttributeDescriptors) {
        std::cout << "[path: " << descriptor.path << ", {";
        for (const auto& attribute : descriptor.attributeNames) {
            std::cout << "\"" << attribute << "\", ";
        }
        std::cout << "}, {";
        for (const auto& attribute : descriptor.attributeTypes) {
            std::cout << "\"" << to_string(attribute) << "\", ";
        }
        std::cout << "}]" << ", ";
    }
    std::cout << std::endl;
    
    // print indexFileDescriptors
    std::cout << "Index file descriptors: " << std::endl;
    for (const auto& descriptor : indexFileDescriptors) {
        std::cout << "[path: " << descriptor.path << ", type: \"" << descriptor.type << "\", name: \"" << descriptor.name << "\"]" << ", ";
    }
    std::cout << std::endl;

    // print headerAttributeNames
    std::cout << "Header attribute names: " << std::endl;
    for (const auto& name : headerAttributeNames) {
        std::cout << '[' << name << ']' << ", ";
    }
    std::cout << std::endl;

    // print headerAttributeTypes
    std::cout << "Header attribute types: " << std::endl;
    for (const auto& type : headerAttributeTypes) {
        std::cout << '[' << to_string(type) << ']' << ", ";
    }

    std::cout << std::endl;

    for (const auto& value : headerAttributeValues) {
        std::cout << '[' << StereoVision::IO::castedPointCloudAttribute<std::string>(value) << ']' << ", ";
    }

    std::cout << std::endl << std::endl;

    // print headerAttributeValues
    std::cout << "------ Header attribute values ------" << std::endl;
    std::cout << std::endl;
    // iterate over headerAttributeNames and get the attribute by name
    for (const auto& name : header->attributeList()) {
        auto valueOpt = header->getAttributeByName(name.c_str());
        if (valueOpt) {
            auto value = *valueOpt;
            std::cout << name << " : " << StereoVision::IO::castedPointCloudAttribute<std::string>(value) << std::endl;
        }
    }
    return 0;
}