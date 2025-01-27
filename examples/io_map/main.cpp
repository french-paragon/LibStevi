#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include "io/attributeMap_pointcloud_io.h"

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

    if (argc > 1) {
        lasFilePath = argv[1];
    }
	else {
		std::cout << "No las file provided" << std::endl;
		return 1;
	}

    std::cout << "This will open a file and replace map the attributes original x to y, original y to x and original z to newZ" << std::endl;
    std::map<std::string, std::string> attributeMap = {{"x", "y"}, {"y", "x"}, {"z", "newZ"}};
    bool onlyKeepAttributesInMap = false;
	// start timer
	auto startTime = std::chrono::high_resolution_clock::now();

    auto fullAccessOpt = StereoVision::IO::openPointCloud(lasFilePath);

    if (!fullAccessOpt) {
        std::cout << "Could not open the las file, check the path" << std::endl;
        return 1;
    }
    std::cout << "file opened" << std::endl;

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

    std::cout << "--------------- first point before map ---------------\n";
    for (auto& att : cloudpoint->attributeList()) {
        std::cout << att << ": " << StereoVision::IO::castedPointCloudAttribute<std::string>(cloudpoint->getAttributeByName(att.c_str()).value_or(std::string{})) << '\n';
    }
    // point
    auto pointGeo = cloudpoint->castedPointGeometry<double>();
    std::cout << "point geometry: " << pointGeo.x << ' ' << pointGeo.y << ' ' << pointGeo.z << '\n';
    auto pointColor = cloudpoint->castedPointColor<double>();
    if (pointColor.has_value())
    {
        std::cout << "point color: " << pointColor->r << ' ' << pointColor->g << ' ' << pointColor->b << '\n';
    }

    std::cout << "--------------first point after map -----------------\n";
    
    // re-open the file
    auto fullAccessOpt2 = StereoVision::IO::openPointCloud(lasFilePath);
    if (!fullAccessOpt2) {
        std::cout << "Could not open the las file, check the path" << std::endl;
        return 1;
    }
    std::cout << "file opened" << std::endl;
    auto& fullAccess2 = *fullAccessOpt2;

    auto fullAccessMap = std::move(mapPointCloudAttributes(std::move(*fullAccessOpt2), attributeMap, onlyKeepAttributesInMap));
    
    auto& cloudpoint2 = fullAccessMap.pointAccess;

    for (auto& att : cloudpoint2->attributeList()) {
        std::cout << att << ": " << StereoVision::IO::castedPointCloudAttribute<std::string>(cloudpoint2->getAttributeByName(att.c_str()).value_or(std::string{})) << '\n';
    }
    // point
    pointGeo = cloudpoint2->castedPointGeometry<double>();
    std::cout << "point geometry: " << pointGeo.x << ' ' << pointGeo.y << ' ' << pointGeo.z << '\n';
    pointColor = cloudpoint2->castedPointColor<double>();
    if (pointColor.has_value()) {
        std::cout << "point color: " << pointColor->r << ' ' << pointColor->g << ' ' << pointColor->b << '\n';
    }

    return 0;
}