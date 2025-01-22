#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include "io/las_pointcloud_io.h"

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


    std::cout << "header attributes: ";
    // display all the header attributes:
    for (auto& att : header->attributeList()) {
        std::cout << '"' << att << "\" ";
    }

    for (auto& att : header->attributeList()) {

        std::cout << att << ": " << StereoVision::IO::castedPointCloudAttribute<std::string>(header->getAttributeByName(att.c_str()).value_or(std::string{})) << '\n';
    }
    std::cout << "Point cloud attributes: ";
    for (auto& att : cloudpoint->attributeList()) {
        std::cout << att << ' ';
    }

    std::cout << "\n\n";

	size_t pointCount = 1;
    for (int i = 0; i < 10; i++)
    {
		pointCount++;
        std::cout << "--------------- point " << i << " ---------------\n";
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
        if (!cloudpoint->gotoNext()) break;
    }
    std::cout << "-------------------------------------------------\n";

    while (cloudpoint->gotoNext())
    {
        pointCount++;
    }
    
    std::cout << "Total number of points: " << pointCount << '\n';

    // stop the timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    std::cout << "Elapsed time for reading: " << elapsed.count() << " s\n";
    std::cout << "-------------------------------------------------\n";

    
    // write the las
    startTime = std::chrono::high_resolution_clock::now();
    
    std::filesystem::path lasFilePathOut = lasFilePath;
    lasFilePathOut.replace_extension("out.las");
    
    // re-open the file
    auto fullAccessOpt2 = StereoVision::IO::openPointCloud(lasFilePath);
    if (!fullAccessOpt2) {
        std::cout << "Could not open the las file, check the path" << std::endl;
        return 1;
    }
    std::cout << "file opened" << std::endl;
    auto& fullAccess2 = *fullAccessOpt2;

    if (!StereoVision::IO::writePointCloudLas(lasFilePathOut, fullAccess2)) {
        std::cout << "Could not write the las file, check the path" << std::endl;
        return 1;
    }

    // stop the timer
    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;
    std::cout << "Elapsed time for writing: " << elapsed.count() << " s\n";
    std::cout << "-------------------------------------------------\n";

    return 0;
}