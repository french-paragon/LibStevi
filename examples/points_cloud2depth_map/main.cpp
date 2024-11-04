#include <iostream>
#include <vector>
#include <random>

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QVector>
#include <QString>

#include <MultidimArrays/MultidimArrays.h>

#include <geometry/rotations.h>
#include <geometry/genericbinarypartitioningtree.h>
#include <io/image_io.h>

#include <Eigen/Core>

#include <omp.h>

using PointType = Eigen::Vector4d;
using BinaryTree = StereoVision::Geometry::GenericBSP<PointType, 2, StereoVision::Geometry::BSPObjectWrapper<PointType,double>>;

BinaryTree::ContainerT loadPoints(QString const& fileName) {
    BinaryTree::ContainerT ret;

    QFile inFile(fileName);

    if(!inFile.open(QFile::ReadOnly)) {
        return ret;
    }

    int count = 0;
    QTextStream in(&inFile);

    in.seek(0);

    while (!in.atEnd()) {

        QString line = in.readLine();

        if (line.startsWith("VERSION")) {
            continue;
        }

        if (line.startsWith("FIELDS")) {
            continue;
        }

        if (line.startsWith("SIZE")) {
            continue;
        }

        if (line.startsWith("TYPE")) {
            continue;
        }

        if (line.startsWith("WIDTH")) {
            continue;
        }

        if (line.startsWith("HEIGHT")) {
            continue;
        }

        if (line.startsWith("VIEWPOINT")) {
            continue;
        }

        if (line.startsWith("POINTS")) {
            QStringList points = line.split(" ", Qt::SkipEmptyParts);
            if (points.size() == 2) {
                bool ok = true;
                size_t nPoints = points[1].toInt(&ok);

                if (ok) {
                    ret.reserve(nPoints);
                }
            }
            continue;
        }

        if (line.startsWith("DATA")) {
            continue;
        }

        QStringList splitted = line.split(" ", Qt::SkipEmptyParts);

        if (splitted.size() < 4) {
            continue;
        }

        PointType xyz;

        bool ok = true;

        for (int i = 0; i < 4; i++) {
            xyz[i] = splitted[i].toFloat(&ok);

            if (!ok) {
                break;
            }
        }

        if (!ok) {
            continue;
        }

        ret.push_back(xyz);
    }

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
                raster.atUnchecked(i,j,0) = -1;
                raster.atUnchecked(i,j,1) = 0;
            }
        }
    }

    return raster;

}

Multidim::Array<float,2> generateDepthMapDirect(std::array<int,2> shape,
                                                BinaryTree::ContainerT & points,
                                                StereoVision::Geometry::AffineTransform<double> const& img2ptcloud,
                                                StereoVision::Geometry::AffineTransform<double> const& ptcloud2img,
                                                float minZ,
                                                QTextStream & out) {

    Multidim::Array<float,2> raster(shape);

    out << "Start creating raster!" << Qt::endl;

    constexpr float distTol = 2;

    #pragma omp parallel for
    for (int i = 0; i < shape[0]; i++) {

        for (int j = 0; j < shape[1]; j++) {
            raster.atUnchecked(i,j) = -1;
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

        if (raster.atUnchecked(i,j) < z) {
            raster.atUnchecked(i,j) = z;
        }
    }

    return raster;

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

    if (argc < 3) {
        err << "Invalid number of arguments provided" << Qt::endl;
        return 1;
    }

    bool ok = true;
    QString pointCloudFile = argv[1];
    int resolution = QString(argv[2]).toInt(&ok);

    if (!ok or resolution <= 0) {
        err << "Invalid resolution provided" << Qt::endl;
        return 1;
    }

    QVector<QString> clustersFileList;
    QVector<QString> validClustersFileList;

    if (argc >= 4) {
        clustersFileList = getFileList(argv[3]);
    }

    if (argc >= 5) {
        validClustersFileList = getFileList(argv[4]);
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

    Multidim::Array<float,2> raster;
    Multidim::Array<float,2> intensityRaster;

    if (useBinaryTree) {
        Multidim::Array<float,3> data = generateDepthMapWithBinaryTree(shape,
                                                     points,
                                                     img2ptcloud,
                                                     ptcloud2img,
                                                     minZ,
                                                     out);
        raster = Multidim::Array<float,2>(data.shape()[0], data.shape()[1]);
        intensityRaster = Multidim::Array<float,2>(data.shape()[0], data.shape()[1]);

        for (int i = 0; i < data.shape()[0]; i++) {
            for (int j = 0; j < data.shape()[1]; j++) {
                raster.atUnchecked(i,j) = data.valueUnchecked(i,j,0);
                intensityRaster.atUnchecked(i,j) = data.valueUnchecked(i,j,1);
            }
        }
    } else {
        raster = generateDepthMapDirect(shape,
                                     points,
                                     img2ptcloud,
                                     ptcloud2img,
                                     minZ,
                                     out);
    }


    out << "Start writing image!" << Qt::endl;

    QString outFile = pointCloudFile + ".tiff";
    QString outIntensityFile = pointCloudFile + "intensity.tiff";

    ok = StereoVision::IO::writeImage<float, float>(outFile.toStdString(), raster);

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
