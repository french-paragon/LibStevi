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

    std::filesystem::path pcdFilePath;

    if (argc > 1) {
        pcdFilePath = argv[1];
    }
	else {
		std::cout << "No pcd file provided" << std::endl;
		return 1;
	}

    //reader
    std::ifstream reader(pcdFilePath, std::ios::binary);
	// create a Las header
    auto lasPointCloudHeader = StereoVision::IO::LasPointCloudHeader::readHeader(reader);

    std::cout << std::endl;


    return 0;
}