#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include "io/sdc_pointcloud_io.h"

int main(int argc, char const *argv[]) {

    std::filesystem::path sdcFilePath;

    if (argc > 1) {
        sdcFilePath = argv[1];
    }
	else {
		std::cout << "No sdc file provided" << std::endl;
		return 1;
	}

	// start timer
	auto startTime = std::chrono::high_resolution_clock::now();

    auto fullAccessOpt = StereoVision::IO::openPointCloudSdc(sdcFilePath);

    if (!fullAccessOpt) {
        std::cout << "Could not open the sdc file, check the path" << std::endl;
        return 1;
    }

    auto& fullAccess = *fullAccessOpt;
    auto& header = fullAccess.headerAccess;
    auto& cloudpoint = fullAccess.pointAccess;

    // display all the attributes:
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
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    std::cout << "-------------------------------------------------\n";

    // sdc write
    std::filesystem::path sdcFilePathOut = sdcFilePath;
    sdcFilePathOut.replace_extension("out.sdc");
    
    // re-open the file
    auto fullAccessOpt2 = StereoVision::IO::openPointCloudSdc(sdcFilePath);

    if (!fullAccessOpt2) {
        std::cout << "Could not open the sdc file, check the path" << std::endl;
        return 1;
    }
    auto& fullAccess2 = *fullAccessOpt2;
    // start timer + write
    startTime = std::chrono::high_resolution_clock::now();
    if (!StereoVision::IO::writePointCloudSdc(sdcFilePathOut, fullAccess2)) return 1;
    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;
    std::cout << "Elapsed time for writing: " << elapsed.count() << " s\n";
    std::cout << "-------------------------------------------------\n";

    return 0;
}