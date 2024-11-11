#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include "io/pcd_pointcloud_io.h"

int main(int argc, char const *argv[]) {

    std::filesystem::path pcdFilePath;

    if (argc > 1) {
        pcdFilePath = argv[1];
    }
	else {
		std::cout << "No pcd file provided" << std::endl;
		return 1;
	}

	// start timer
	auto startTime = std::chrono::high_resolution_clock::now();

    auto fullAccessOpt = StereoVision::IO::openPointCloudPcd(pcdFilePath);

    if (!fullAccessOpt) {
        std::cout << "Could not open the pcd file, check the path" << std::endl;
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

    for (auto& att : header->attributeList()) {
        std::cout << '"' << att << "\" ";
    }

    std::cout << '\n';


    // display all the header attributes:
    std::cout << "version: " << std::get<double>(header->getAttributeByName("version").value_or(double{})) << '\n';
    std::cout << "fields: " << std::get<std::string>(header->getAttributeByName("fields").value_or(std::string{})) << '\n';
    std::cout << "size: " << std::get<std::string>(header->getAttributeByName("size").value_or(std::string{})) << '\n';
    std::cout << "type: " << std::get<std::string>(header->getAttributeByName("type").value_or(std::string{})) << '\n';
    std::cout << "count: " << std::get<std::string>(header->getAttributeByName("count").value_or(std::string{})) << '\n';
    std::cout << "width: " << std::get<size_t>(header->getAttributeByName("width").value_or(size_t{})) << '\n';
    std::cout << "height: " << std::get<size_t>(header->getAttributeByName("height").value_or(size_t{})) << '\n';
    std::cout << "viewpoint: " << std::get<std::string>(header->getAttributeByName("viewpoint").value_or(std::string{})) << '\n';
    std::cout << "points: " << std::get<size_t>(header->getAttributeByName("points").value_or(size_t{})) << '\n';
    std::cout << "data: " << std::get<std::string>(header->getAttributeByName("data").value_or(std::string{})) << '\n';

    // std::cout << "Point cloud attributes: ";
    // for (auto& att : cloudpoint->attributeList()) {
    //     std::cout << att << ' ';
    // }

    std::cout << "\n\n";

	size_t pointCount = 1;
    for (int i = 0; i < 10; i++)
    {
		// pointCount++;
        // std::cout << "--------------- point " << i << " ---------------\n";
        // std::cout << "time: " << std::get<double>(cloudpoint->getAttributeByName("time").value_or(double{})) << '\n';
        // std::cout << "range: " << std::get<float>(cloudpoint->getAttributeByName("range").value_or(float{})) << '\n';
        // std::cout << "theta: " << std::get<float>(cloudpoint->getAttributeByName("theta").value_or(float{})) << '\n';
        // std::cout << "x: " << std::get<float>(cloudpoint->getAttributeByName("x").value_or(float{})) << '\n';
        // std::cout << "y: " << std::get<float>(cloudpoint->getAttributeByName("y").value_or(float{})) << '\n';
        // std::cout << "z: " << std::get<float>(cloudpoint->getAttributeByName("z").value_or(float{})) << '\n';
        // std::cout << "amplitude: " << std::get<uint16_t>(cloudpoint->getAttributeByName("amplitude").value_or(uint16_t{})) << '\n';
        // std::cout << "width: " << std::get<uint16_t>(cloudpoint->getAttributeByName("width").value_or(uint16_t{})) << '\n';
        // std::cout << "targettype: " << std::get<uint8_t>(cloudpoint->getAttributeByName("targettype").value_or(uint8_t{})) << '\n';
        // std::cout << "target: " << std::get<uint8_t>(cloudpoint->getAttributeByName("target").value_or(uint8_t{})) << '\n';
        // std::cout << "numtarget: " << std::get<uint8_t>(cloudpoint->getAttributeByName("numtarget").value_or(uint8_t{})) << '\n';
        // std::cout << "rgindex: " << std::get<uint16_t>(cloudpoint->getAttributeByName("rgindex").value_or(uint16_t{})) << '\n';
        // std::cout << "channeldesc: " << std::get<uint8_t>(cloudpoint->getAttributeByName("channeldesc").value_or(uint8_t{})) << '\n';
        // std::cout << "classid: " << std::get<uint8_t>(cloudpoint->getAttributeByName("classid").value_or(uint8_t{})) << '\n';
        // std::cout << "rho: " << std::get<float>(cloudpoint->getAttributeByName("rho").value_or(float{})) << '\n';
        // std::cout << "reflectance: " << std::get<int16_t>(cloudpoint->getAttributeByName("reflectance").value_or(int16_t{})) << '\n';
        // if (!cloudpoint->gotoNext()) break;
    }
        std::cout << "-------------------------------------------------\n";

		// while (cloudpoint->gotoNext())
		// {
		// 	pointCount++;
		// }
		
		std::cout << "Total number of points: " << pointCount << '\n';

		// stop the timer
		auto endTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = endTime - startTime;
		std::cout << "Elapsed time: " << elapsed.count() << " s\n";
		std::cout << "-------------------------------------------------\n";

    return 0;
}