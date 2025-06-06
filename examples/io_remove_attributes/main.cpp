#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include "io/attributeRemover.h"

// function that converts any std::vector to a string
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    for (const auto& item : vec) {
        os << item << " ";
    }
    return os;
}

int main(int argc, char const *argv[]) {

    std::filesystem::path lasFilePath;
    std::vector<std::string> attributesToRemove;

    if (argc > 1) {
        lasFilePath = argv[1];
    } else {
		std::cout << "No las file provided" << std::endl;
		return 1;
	}

    if (argc > 2) {
        attributesToRemove = std::vector<std::string>(argv + 2, argv + argc);
    }

    std::cout << "This will open a pointcloud and remove attributes" << std::endl;

    auto fullAccessOpt = StereoVision::IO::openPointCloud(lasFilePath);

    if (!fullAccessOpt) {
        std::cout << "Could not open the las file, check the path" << std::endl;
        return 1;
    }
    std::cout << "file opened" << std::endl;

    // attributes that will be removed
    std::cout<< "Attributes to remove: ";
    for (auto& att : attributesToRemove) {
        std::cout << att << ' ';
    }
    std::cout << std::endl;

    auto& fullAccess = *fullAccessOpt;
    auto& header = fullAccess.headerAccess;
    auto& cloudpoint = fullAccess.pointAccess;

    // display all the attributes:
    // test if the header is null
    if (header == nullptr) {
        std::cout << "header is null" << std::endl;
    }

    std::cout << '\n';


    std::cout << "Point cloud attributes: ";
    for (auto& att : cloudpoint->attributeList()) {
        std::cout << att << ' ';
    }

    std::cout << "\n\n";

    std::cout << "--------------- first point before remove ---------------\n";
    for (auto& att : cloudpoint->attributeList()) {
        std::cout << att << ": " << StereoVision::IO::castedPointCloudAttribute<std::string>(cloudpoint->getAttributeByName(att.c_str()).value_or(std::string{})) << '\n';
    }
    // point
    auto pointGeo = cloudpoint->castedPointGeometry<double>();
    std::cout << "point geometry: " << pointGeo.x << ' ' << pointGeo.y << ' ' << pointGeo.z << '\n';
    auto pointColor = cloudpoint->castedPointColor<double>();
    if (pointColor.has_value())
    {
        std::cout << "point color: " << pointColor->r << ' ' << pointColor->g << ' ' << pointColor->b << ' ' << pointColor->a << '\n';
    }

    std::cout << "--------------first point after remove -----------------\n";
    
    // re-open the file
    auto fullAccessOpt2 = StereoVision::IO::openPointCloud(lasFilePath);
    if (!fullAccessOpt2) {
        std::cout << "Could not open the las file, check the path" << std::endl;
        return 1;
    }
    std::cout << "file opened" << std::endl;
    auto& fullAccess2 = *fullAccessOpt2;
    auto fullAccessRemover = std::make_unique<StereoVision::IO::FullPointCloudAccessInterface>(std::move(fullAccess2));

    fullAccessRemover = std::move(StereoVision::IO::RemoveAttributesOrColorFromPointCloud(fullAccessRemover, attributesToRemove));

    auto& cloudpoint2 = fullAccessRemover->pointAccess;

    for (auto& att : cloudpoint2->attributeList()) {
        std::cout << att << ": " << StereoVision::IO::castedPointCloudAttribute<std::string>(cloudpoint2->getAttributeByName(att.c_str()).value_or(std::string{})) << '\n';
    }
    // point
    pointGeo = cloudpoint2->castedPointGeometry<double>();
    std::cout << "point geometry: " << pointGeo.x << ' ' << pointGeo.y << ' ' << pointGeo.z << '\n';
    pointColor = cloudpoint2->castedPointColor<double>();
    if (pointColor.has_value()) {
        std::cout << "point color: " << pointColor->r << ' ' << pointColor->g << ' ' << pointColor->b << ' ' << pointColor->a << '\n';
    }
    
    std::cout << "--------------first point after removing all attributes -----------------\n";

    fullAccessRemover = std::move(StereoVision::IO::RemoveAttributesOrColorFromPointCloud(fullAccessRemover, {}, std::nullopt, true));

    for (auto& att : fullAccessRemover->pointAccess->attributeList()) {
        std::cout << att << ": " << StereoVision::IO::castedPointCloudAttribute<std::string>(fullAccessRemover->pointAccess->getAttributeByName(att.c_str()).value_or(std::string{})) << '\n';
    }
    // point
    pointGeo = fullAccessRemover->pointAccess->castedPointGeometry<double>();
    std::cout << "point geometry: " << pointGeo.x << ' ' << pointGeo.y << ' ' << pointGeo.z << '\n';
    pointColor = fullAccessRemover->pointAccess->castedPointColor<double>();
    if (pointColor.has_value()) {
        std::cout << "point color: " << pointColor->r << ' ' << pointColor->g << ' ' << pointColor->b << ' ' << pointColor->a << '\n';
    }

    std::cout << "--------------first point after removing color -----------------\n";

    fullAccessRemover = std::move(StereoVision::IO::RemoveAttributesOrColorFromPointCloud(fullAccessRemover, {}, true, std::nullopt));

    for (auto& att : fullAccessRemover->pointAccess->attributeList()) {
        std::cout << att << ": " << StereoVision::IO::castedPointCloudAttribute<std::string>(fullAccessRemover->pointAccess->getAttributeByName(att.c_str()).value_or(std::string{})) << '\n';
    }
    // point
    pointGeo = fullAccessRemover->pointAccess->castedPointGeometry<double>();
    std::cout << "point geometry: " << pointGeo.x << ' ' << pointGeo.y << ' ' << pointGeo.z << '\n';
    pointColor = fullAccessRemover->pointAccess->castedPointColor<double>();
    if (pointColor.has_value()) {
        std::cout << "point color: " << pointColor->r << ' ' << pointColor->g << ' ' << pointColor->b << ' ' << pointColor->a << '\n';
    }

    return 0;
}