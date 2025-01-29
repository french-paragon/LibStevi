#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include "io/pcd_pointcloud_io.h"

// function that converts any std::vector to a string
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    for (const auto& item : vec) {
        os << item << " ";
    }
    return os;
}

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

    auto fullAccessOpt = StereoVision::IO::openPointCloud(pcdFilePath);

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
            std::cout << "point color: " << pointColor->r << ' ' << pointColor->g << ' ' << pointColor->b << ' ' << pointColor->a << '\n';
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

    // write the point cloud to a pcd file

    startTime = std::chrono::high_resolution_clock::now();
    // reopen the file
    auto fullAccessWriteOpt = StereoVision::IO::openPointCloud(pcdFilePath);
    if (!fullAccessWriteOpt) {
        std::cout << "Could not open the pcd file, check the path" << std::endl;
        return 1;
    }
    auto& fullAccessWrite = *fullAccessWriteOpt;
    std::filesystem::path pcdFilePathOut = pcdFilePath;
    pcdFilePathOut.replace_extension("out.pcd");
    StereoVision::IO::writePointCloudPcd(pcdFilePathOut, fullAccessWrite, StereoVision::IO::PcdDataStorageType::binary);

    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;
    std::cout << "Elapsed time for writing to binary: " << elapsed.count() << " s\n";

    startTime = std::chrono::high_resolution_clock::now();
    // reopen the file and write it to ascii
    auto fullAccessWriteOpt2 = StereoVision::IO::openPointCloud(pcdFilePathOut);
    if (!fullAccessWriteOpt2) {
        std::cout << "Could not open the pcd file, check the path" << std::endl;
        return 1;
    }
    auto& fullAccessWrite2 = *fullAccessWriteOpt2;
    pcdFilePathOut.replace_extension("out2.pcd");
    StereoVision::IO::writePointCloudPcd(pcdFilePathOut, fullAccessWrite2, StereoVision::IO::PcdDataStorageType::ascii);

    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;
    std::cout << "Elapsed time for writing from binary to ascii: " << elapsed.count() << " s\n";

    return 0;
}