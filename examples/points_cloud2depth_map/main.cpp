#include <iostream>
#include <vector>
#include <random>

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QVector>
#include <QString>

#include <MultidimArrays/MultidimArrays.h>

#include <imageProcessing/inpainting.h>
#include <imageProcessing/morphologicalOperators.h>
#include <geometry/rotations.h>
#include <geometry/genericbinarypartitioningtree.h>
#include <io/image_io.h>
#include <io/pointcloud_io.h>

#include <Eigen/Core>

#include <tclap/CmdLine.h>

#include <omp.h>

using PointType = Eigen::Vector4d;
using BinaryTree = StereoVision::Geometry::GenericBSP<PointType, 2, StereoVision::Geometry::BSPObjectWrapper<PointType,double>>;

BinaryTree::ContainerT loadPoints(QString const& fileName) {
    BinaryTree::ContainerT ret;

    std::optional<StereoVision::IO::FullPointCloudAccessInterface> optPointCloud =
            StereoVision::IO::openPointCloud(fileName.toStdString());

    if (!optPointCloud.has_value()) {
        return ret;
    }

    StereoVision::IO::FullPointCloudAccessInterface& ptCloud = optPointCloud.value();

    bool hasMore = true;

    do {

        PointType point;
        auto geom = ptCloud.pointAccess->castedPointGeometry<double>();

        point.x() = geom.x;
        point.y() = geom.y;
        point.z() = geom.z;

        double intensity = StereoVision::IO::castedPointCloudAttribute<double>(
                    ptCloud.pointAccess->getAttributeByName("intensity").value_or(0.0));

        point[3] = intensity;

        ret.push_back(point);

        hasMore = ptCloud.pointAccess->gotoNext();
    } while (hasMore);

    return ret;
}

QVector<QString> getFileList(QString const& fileListFileName) {
    QVector<QString> ret;

    QFile inFile(fileListFileName);

    if(!inFile.open(QFile::ReadOnly)) {
        return ret;
    }

    int count = 0;
    QTextStream in(&inFile);

    in.seek(0);

    while (!in.atEnd()) {
        QString line = in.readLine();
        ret.push_back(line);
    }

    return ret;

}

Multidim::Array<float,3> generateDepthMapWithBinaryTree(std::array<int,2> shape,
                                                        BinaryTree::ContainerT & points,
                                                        StereoVision::Geometry::AffineTransform<double> const& img2ptcloud,
                                                        StereoVision::Geometry::AffineTransform<double> const& ptcloud2img,
                                                        float minZ,
                                                        QTextStream & out) {

    Multidim::Array<float,3> raster(shape[0], shape[1], 2);

    out << "Start building binary space partitioning tree!" << Qt::endl;

    BinaryTree tree(std::move(points));

    out << "Start creating raster!" << Qt::endl;

    constexpr float distTol = 8;
    constexpr float bgVal = -1;

    #pragma omp parallel for
    for (int i = 0; i < shape[0]; i++) {

        for (int j = 0; j < shape[1]; j++) {
            Eigen::Vector3d imgCoord(i,j,0);
            Eigen::Vector3d minImg(i-distTol,j-distTol,0);
            Eigen::Vector3d maxImg(i+distTol,j+distTol,0);

            Eigen::Vector4d ptCoords;
            ptCoords.block<3,1>(0,0) = img2ptcloud*imgCoord;
            Eigen::Vector4d min;
            min.block<3,1>(0,0) = img2ptcloud*minImg;
            Eigen::Vector4d max;
            max.block<3,1>(0,0) = img2ptcloud*maxImg;

            for (int d = 0; d < 2; d++) {
                if (min[d] > max[d]) {
                    float tmp = min[d];
                    min[d] = max[d];
                    max[d] = tmp;
                }
            }

            int pointIdx = tree.closestInRange(ptCoords, min, max);

            if (pointIdx < 0) {
                raster.atUnchecked(i,j,0) = -1;
                raster.atUnchecked(i,j,1) = 0;
                continue;
            }

            PointType& point = tree[pointIdx];

            Eigen::Vector3d imgPoint = ptcloud2img*point.block<3,1>(0,0);
            imgPoint.z() = 0;

            float dist = (imgPoint - imgCoord).norm();

            if (dist <= distTol) {
                raster.atUnchecked(i,j,0) = point[2] - minZ;
                raster.atUnchecked(i,j,1) = point[3];
            } else {
                raster.atUnchecked(i,j,0) = bgVal;
                raster.atUnchecked(i,j,1) = 0;
            }
        }
    }

    return raster;

}

std::tuple<Multidim::Array<float,3>,Multidim::Array<float,2>> generateDepthMapDirect(std::array<int,2> shape,
                                                BinaryTree::ContainerT & points,
                                                StereoVision::Geometry::AffineTransform<double> const& img2ptcloud,
                                                StereoVision::Geometry::AffineTransform<double> const& ptcloud2img,
                                                float minZ,
                                                int bufferDepth,
                                                QTextStream & out) {

    Multidim::Array<float,3> raster(shape[0], shape[1], bufferDepth);
    Multidim::Array<float,2> intensity(shape[0], shape[1]);

    out << "Start creating raster!" << Qt::endl;

    constexpr float distTol = 2;
    constexpr float bgVal = -1;

    #pragma omp parallel for
    for (int i = 0; i < shape[0]; i++) {

        for (int j = 0; j < shape[1]; j++) {

            intensity.atUnchecked(i,j) = 0;

            for (int d = 0; d < bufferDepth; d++) {
                raster.atUnchecked(i,j,d) = bgVal;
            }
        }
    }

    #pragma omp parallel for
    for (int idx = 0; idx < points.size(); idx++) {

        Eigen::Vector4d const& ptCoords = points[idx];

        Eigen::Vector3d imgCoord = ptcloud2img*ptCoords.block<3,1>(0,0);

        size_t i = std::round(imgCoord.x());
        size_t j = std::round(imgCoord.y());

        if (i < 0 or j < 0) {
            continue;
        }

        if (i >= shape[0] or j >= shape[1]) {
            continue;
        }

        double z = ptCoords.z() - minZ;

        double currentZ = z;

        for (int d = 0; d < bufferDepth; d++) {

            double currentDepth = raster.atUnchecked(i,j,d);

            if (raster.atUnchecked(i,j,d) < currentZ) {
                raster.atUnchecked(i,j,d) = currentZ;
            }

            if (currentDepth > bgVal) {
                currentZ = currentDepth;
            }
        }

        intensity.atUnchecked(i,j) = ptCoords[3];
    }

    return std::make_tuple(std::move(raster), std::move(intensity));

}

Multidim::Array<uint8_t,3> getCoverMask(std::array<int,2> shape,
                                        QVector<QString> const& clustersFiles,
                                        StereoVision::Geometry::AffineTransform<double> const& ptcloud2img,
                                        QTextStream & out) {

    Multidim::Array<uint8_t,3> raster(shape[0],shape[1], 3);

    out << "Start creating cover mask!" << Qt::endl;

    #pragma omp parallel for
    for (int i = 0; i < shape[0]; i++) {

        for (int j = 0; j < shape[1]; j++) {
            for (int c = 0; c < 3; c++) {
                raster.atUnchecked(i,j,c) = 0;
            }
        }
    }

    std::default_random_engine re;
    re.seed(shape[0]+shape[1]);

    std::uniform_int_distribution<uint8_t> colorGen(25,255);

    for (QString const& filename : clustersFiles) {
        BinaryTree::ContainerT clusterPoints = loadPoints(filename);

        uint8_t red = colorGen(re);
        uint8_t green = colorGen(re);
        uint8_t blue = colorGen(re);

        #pragma omp parallel for
        for (PointType const& pt : clusterPoints) {

            Eigen::Vector3d imgPoint = ptcloud2img*pt.block<3,1>(0,0);

            int i = std::round(imgPoint.x());
            int j = std::round(imgPoint.y());

            if (i < 0 or j < 0) {
                continue;
            }

            if (i >= shape[0] or j >= shape[1]) {
                continue;
            }

            raster.atUnchecked(i,j,0) = red;
            raster.atUnchecked(i,j,1) = green;
            raster.atUnchecked(i,j,2) = blue;
        }
    }

    return raster;

}

Multidim::Array<uint32_t,2> getClustersMask(std::array<int,2> shape,
                                            QVector<QString> const& clustersFiles,
                                            StereoVision::Geometry::AffineTransform<double> const& ptcloud2img,
                                            QTextStream & out) {

    Multidim::Array<uint32_t,2> raster(shape);

    out << "Start creating cluster mask!" << Qt::endl;

    #pragma omp parallel for
    for (int i = 0; i < shape[0]; i++) {

        for (int j = 0; j < shape[1]; j++) {
            raster.atUnchecked(i,j) = 0;
        }
    }

    for (int c = 0; c < clustersFiles.size(); c++) {

        QString const& filename = clustersFiles[c];

        BinaryTree::ContainerT clusterPoints = loadPoints(filename);

        #pragma omp parallel for
        for (PointType const& pt : clusterPoints) {

            Eigen::Vector3d imgPoint = ptcloud2img*pt.block<3,1>(0,0);

            int i = std::round(imgPoint.x());
            int j = std::round(imgPoint.y());

            if (i < 0 or j < 0) {
                continue;
            }

            if (i >= shape[0] or j >= shape[1]) {
                continue;
            }

            raster.atUnchecked(i,j) = c+1;
        }
    }

    return raster;

}

int main(int argc, char** argv) {

    QTextStream out(stdout);
    QTextStream err(stderr);

    QString pointCloudFile;
    int resolution;
    int bufferDepth = 1;
    int inPaintingRadius = 5;

    QVector<QString> clustersFileList;
    QVector<QString> validClustersFileList;

    try {
        TCLAP::CmdLine cmd("Export a point cloud to a depth map", '=', "0.0");

        TCLAP::UnlabeledValueArg<std::string> cloudFilePathArg("ptCloudFilePath", "Path where the points cloud is stored", true, "", "local path to point cloud file");

        TCLAP::ValueArg<int> resArg("r","resolution", "resolution to use for export", true, 1, "int");
        TCLAP::ValueArg<int> bufferDepthArg("b","bufferDepth", "buffer depth of the depth map", false, bufferDepth, "int");
        TCLAP::ValueArg<std::string> clustersArg("", "clusters", "clusters definition file", false, "", "filepath");
        TCLAP::ValueArg<std::string> vclustersArg("", "vclusters", "valid clusters definition file", false, "", "filepath");

        cmd.add(cloudFilePathArg);
        cmd.add(resArg);
        cmd.add(bufferDepthArg);
        cmd.add(clustersArg);
        cmd.add(vclustersArg);

        cmd.parse(argc, argv);

        pointCloudFile = QString::fromStdString(cloudFilePathArg.getValue());
        resolution = resArg.getValue();

        if (bufferDepthArg.isSet()) {
            int tmp = bufferDepthArg.getValue();
            bufferDepth = std::max(tmp, 1);
        }

        if (clustersArg.isSet()) {
            clustersFileList = getFileList(clustersArg.getValue().c_str());
        }

        if (vclustersArg.isSet()) {
            validClustersFileList = getFileList(vclustersArg.getValue().c_str());
        }


    } catch (TCLAP::ArgException &e) {
        err << "Argument error:" << e.error().c_str() << " for arg " << e.argId().c_str() << Qt::endl;
    }

    out << "Start loading points data!" << Qt::endl;

    BinaryTree::ContainerT points = loadPoints(pointCloudFile);

    if (points.empty()) {
        err << "Could not load point data" << Qt::endl;
        return 1;
    }

    float minX = points[0].x();
    float maxX = points[0].x();
    float minY = points[0].y();
    float maxY = points[0].y();
    float minZ = points[0].z();

    for (int i = 1; i < points.size(); i++) {

        if (points[i].x() < minX) {
            minX = points[i].x();
        }
        if (points[i].x() > maxX) {
            maxX = points[i].x();
        }

        if (points[i].y() < minY) {
            minY = points[i].y();
        }
        if (points[i].y() > maxY) {
            maxY = points[i].y();
        }

        if (points[i].z() < minZ) {
            minZ = points[i].z();
        }
    }

    float rangeX = maxX - minX;
    float rangeY = maxY - minY;

    if (!std::isfinite(rangeX) or !std::isfinite(rangeY)) {
        err << "Invalid point data" << Qt::endl;
        return 1;
    }

    float rangeMax = std::max(rangeX, rangeY);
    float scale = resolution/rangeMax;
    float invScale = rangeMax/resolution;

    StereoVision::Geometry::AffineTransform<double> img2ptcloud;
    img2ptcloud.t.x() = minX;
    img2ptcloud.t.y() = maxY;
    img2ptcloud.t.z() = minZ;

    img2ptcloud.R = Eigen::Matrix3d::Identity();
    img2ptcloud.R(0,0) = 0;
    img2ptcloud.R(1,1) = 0;
    img2ptcloud.R(1,0) = -invScale;
    img2ptcloud.R(0,1) = invScale;

    StereoVision::Geometry::AffineTransform<double> ptcloud2img;

    ptcloud2img.R = Eigen::Matrix3d::Identity();
    ptcloud2img.R(0,0) = 0;
    ptcloud2img.R(1,1) = 0;
    ptcloud2img.R(1,0) = scale;
    ptcloud2img.R(0,1) = -scale;

    ptcloud2img.t = -ptcloud2img.R*img2ptcloud.t;

    StereoVision::Geometry::AffineTransform<float> check = ptcloud2img.cast<float>()*img2ptcloud.cast<float>();

    auto oldNotation = out.realNumberNotation();
    out.setRealNumberNotation(QTextStream::RealNumberNotation::FixedNotation);

    out << "Image to point cloud coordinates transform: " << Qt::endl;
    out << img2ptcloud.R(0,0) << " " << img2ptcloud.R(0,1) << " " << img2ptcloud.R(0,2) << " " << img2ptcloud.t[0] << "\n";
    out << img2ptcloud.R(1,0) << " " << img2ptcloud.R(1,1) << " " << img2ptcloud.R(1,2) << " " << img2ptcloud.t[1] << "\n";
    out << img2ptcloud.R(2,0) << " " << img2ptcloud.R(2,1) << " " << img2ptcloud.R(2,2) << " " << img2ptcloud.t[2] << Qt::endl;

    out << "Point cloud to Image coordinates transform: " << Qt::endl;
    out << ptcloud2img.R(0,0) << " " << ptcloud2img.R(0,1) << " " << ptcloud2img.R(0,2) << " " << ptcloud2img.t[0] << "\n";
    out << ptcloud2img.R(1,0) << " " << ptcloud2img.R(1,1) << " " << ptcloud2img.R(1,2) << " " << ptcloud2img.t[1] << "\n";
    out << ptcloud2img.R(2,0) << " " << ptcloud2img.R(2,1) << " " << ptcloud2img.R(2,2) << " " << ptcloud2img.t[2] << Qt::endl;

    out << "Check: " << Qt::endl;
    out << check.R(0,0) << " " << check.R(0,1) << " " << check.R(0,2) << " " << check.t[0] << "\n";
    out << check.R(1,0) << " " << check.R(1,1) << " " << check.R(1,2) << " " << check.t[1] << "\n";
    out << check.R(2,0) << " " << check.R(2,1) << " " << check.R(2,2) << " " << check.t[2] << Qt::endl;

    out.setRealNumberNotation(oldNotation);

    int sX = std::floor(rangeX*scale);
    int sY = std::floor(rangeY*scale);

    double nPixels = static_cast<size_t>(sX)*static_cast<size_t>(sY);

    std::array<int,2> shape = {sY, sX};


    double nPts = points.size();

    bool useBinaryTree = true;

    Multidim::Array<float,3> raster;
    Multidim::Array<float,2> intensityRaster;

    if (useBinaryTree) {
        Multidim::Array<float,3> data = generateDepthMapWithBinaryTree(shape,
                                                     points,
                                                     img2ptcloud,
                                                     ptcloud2img,
                                                     minZ,
                                                     out);
        raster = Multidim::Array<float,3>(data.shape()[0], data.shape()[1],1);
        intensityRaster = Multidim::Array<float,2>(data.shape()[0], data.shape()[1]);

        for (int i = 0; i < data.shape()[0]; i++) {
            for (int j = 0; j < data.shape()[1]; j++) {
                raster.atUnchecked(i,j,0) = data.valueUnchecked(i,j,0);
                intensityRaster.atUnchecked(i,j) = data.valueUnchecked(i,j,1);
            }
        }

    } else {
        std::tie(raster, intensityRaster) = generateDepthMapDirect(shape,
                                     points,
                                     img2ptcloud,
                                     ptcloud2img,
                                     minZ,
                                     bufferDepth,
                                     out);

        Multidim::Array<bool,2> coverMask(shape[0], shape[1]);

        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                coverMask.atUnchecked(i,j) = raster.valueUnchecked(i,j,0) >= 0; //pixels that do not need to be inpainted
            }
        }

        Multidim::Array<bool,2> inpaintingMask =
                StereoVision::ImageProcessing::erosion(inPaintingRadius+2,inPaintingRadius+2,
                    StereoVision::ImageProcessing::dilation(inPaintingRadius,inPaintingRadius,coverMask)
                );

        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                inpaintingMask.atUnchecked(i,j) = inpaintingMask.valueUnchecked(i,j) and !coverMask.valueUnchecked(i,j);
            }
        }

        raster = StereoVision::ImageProcessing::nearestInPaintingBatched<float,3,1>(raster,inpaintingMask,{2});
        intensityRaster = StereoVision::ImageProcessing::nearestInPaintingMonochannel(intensityRaster, inpaintingMask);
    }


    out << "Start writing image!" << Qt::endl;

    QString outFile = pointCloudFile + ".tiff";
    QString outIntensityFile = pointCloudFile + "intensity.tiff";

    bool ok = StereoVision::IO::writeImage<float, float>(outFile.toStdString(), raster);

    if (!ok) {
        err << "Error while writing image" << Qt::endl;
        return 1;
    }

    if (!intensityRaster.empty()) {
        ok = StereoVision::IO::writeImage<float, float>(outIntensityFile.toStdString(), intensityRaster);

        if (!ok) {
            err << "Error while writing intensity image" << Qt::endl;
            return 1;
        }
    }

    out << "Image written!" << Qt::endl;

    out.setRealNumberNotation(QTextStream::FixedNotation);
    out.setRealNumberPrecision(4);

    if (!clustersFileList.isEmpty()) {

        Multidim::Array<uint8_t,3> coverMask = getCoverMask(shape, clustersFileList, ptcloud2img, out);

        out << "Start writing cover mask!" << Qt::endl;

        QString outFile = pointCloudFile + "_cover_mask.tiff";

        ok = StereoVision::IO::writeImage<uint8_t, uint8_t>(outFile.toStdString(), coverMask);

        if (!ok) {
            err << "Error while writing cover mask" << Qt::endl;
            return 1;
        }

        out << "Cover mask written!" << Qt::endl;
    }

    if (!validClustersFileList.isEmpty()) {

        Multidim::Array<uint32_t,2> clustersMask = getClustersMask(shape, validClustersFileList, ptcloud2img, out);

        out << "Start writing clusters mask!" << Qt::endl;

        QString outFile = pointCloudFile + "_clusters_mask.tiff";

        ok = StereoVision::IO::writeImage<float, uint32_t>(outFile.toStdString(), clustersMask);

        if (!ok) {
            err << "Error while writing clusters mask" << Qt::endl;
            return 1;
        }

        out << "Clusters mask written!" << Qt::endl;
    }

    return 0;

}
