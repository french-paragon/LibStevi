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
    for (auto& att : header->attributeList()) {
        std::cout << '"' << att << "\" ";
    }
    std::cout << "\n\n";

    std::cout << "Header size: " << std::get<uint32_t>(header->getAttributeByName("headerSize").value_or(0)) << '\n';
    std::cout << "Major version: " << std::get<uint16_t>(header->getAttributeByName("majorVersion").value_or(0)) << '\n';
    std::cout << "Minor version: " << std::get<uint16_t>(header->getAttributeByName("minorVersion").value_or(0)) << '\n';
    std::cout << std::get<std::string>(header->getAttributeByName("headerInformation").value_or("")) << '\n';

    std::cout << '\n';

    std::cout << "Point cloud attributes: ";
    for (auto& att : cloudpoint->attributeList()) {
        std::cout << att << ' ';
    }

    std::cout << "\n\n";

    for (int i = 0; i < 10; i++)
    {
        std::cout << "--------------- point " << i << " ---------------\n";
        std::cout << "time: " << std::get<double>(cloudpoint->getAttributeByName("time").value_or(double{})) << '\n';
        std::cout << "range: " << std::get<float>(cloudpoint->getAttributeByName("range").value_or(float{})) << '\n';
        std::cout << "theta: " << std::get<float>(cloudpoint->getAttributeByName("theta").value_or(float{})) << '\n';
        std::cout << "x: " << std::get<float>(cloudpoint->getAttributeByName("x").value_or(float{})) << '\n';
        std::cout << "y: " << std::get<float>(cloudpoint->getAttributeByName("y").value_or(float{})) << '\n';
        std::cout << "z: " << std::get<float>(cloudpoint->getAttributeByName("z").value_or(float{})) << '\n';
        std::cout << "amplitude: " << std::get<uint16_t>(cloudpoint->getAttributeByName("amplitude").value_or(uint16_t{})) << '\n';
        std::cout << "width: " << std::get<uint16_t>(cloudpoint->getAttributeByName("width").value_or(uint16_t{})) << '\n';
        std::cout << "targettype: " << std::get<uint8_t>(cloudpoint->getAttributeByName("targettype").value_or(uint8_t{})) << '\n';
        std::cout << "target: " << std::get<uint8_t>(cloudpoint->getAttributeByName("target").value_or(uint8_t{})) << '\n';
        std::cout << "numtarget: " << std::get<uint8_t>(cloudpoint->getAttributeByName("numtarget").value_or(uint8_t{})) << '\n';
        std::cout << "rgindex: " << std::get<uint16_t>(cloudpoint->getAttributeByName("rgindex").value_or(uint16_t{})) << '\n';
        std::cout << "channeldesc: " << std::get<uint8_t>(cloudpoint->getAttributeByName("channeldesc").value_or(uint8_t{})) << '\n';
        std::cout << "classid: " << std::get<uint8_t>(cloudpoint->getAttributeByName("classid").value_or(uint8_t{})) << '\n';
        std::cout << "rho: " << std::get<float>(cloudpoint->getAttributeByName("rho").value_or(float{})) << '\n';
        std::cout << "reflectance: " << std::get<int16_t>(cloudpoint->getAttributeByName("reflectance").value_or(int16_t{})) << '\n';
        if (!cloudpoint->gotoNext()) break;
    }
        std::cout << "-------------------------------------------------\n";

    return 0;
}