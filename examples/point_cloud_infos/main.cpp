#include <iostream>

#include <tclap/CmdLine.h>

#include "io/pointcloud_io.h"

int main(int argc, char** argv) {

    std::string pointCloudFile;

    try {
        TCLAP::CmdLine cmd("List the information about a point cloud", '=', "0.0");

        TCLAP::UnlabeledValueArg<std::string> cloudFilePathArg("ptCloudFilePath", "Path where the points cloud is stored", true, "", "local path to point cloud file");

        cmd.add(cloudFilePathArg);

        cmd.parse(argc, argv);

        pointCloudFile = cloudFilePathArg.getValue();

    } catch (TCLAP::ArgException &e) {
        std::cerr << "Argument error:" << e.error().c_str() << " for arg " << e.argId().c_str() << std::endl;
    }

    std::optional<StereoVision::IO::FullPointCloudAccessInterface> optPointCloud =
        StereoVision::IO::openPointCloud(pointCloudFile);

    if (!optPointCloud.has_value()) {
        std::cerr << "Could not open point cloud \"" << pointCloudFile << "\"" << std::endl;
        return -1;
    }

    StereoVision::IO::FullPointCloudAccessInterface& pointCloud = optPointCloud.value();

    std::cout << "Information about point cloud in: \"" << pointCloudFile << "\"" << std::endl;
    std::cout << std::endl;

    std::vector<std::string> headerAttrs = pointCloud.headerAccess->attributeList();

    if (headerAttrs.empty()) {
        std::cout << "\t" << "Empty header attributes!" << std::endl;
    } else {
        std::cout << "\t" << "Header attributes:" << std::endl;
        std::cout << "\t";

        for (std::string const& attrName : headerAttrs) {
            std::cout << attrName << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    std::vector<std::string> pointsAttrs = pointCloud.pointAccess->attributeList();

    if (pointsAttrs.empty()) {
        std::cout << "\t" << "Empty points attributes!" << std::endl;
    } else {
        std::cout << "\t" << "Points attributes:" << std::endl;
        std::cout << "\t";

        for (std::string const& attrName : pointsAttrs) {
            std::cout << attrName << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    return 0;
}
