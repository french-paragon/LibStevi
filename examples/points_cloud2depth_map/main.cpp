#include <iostream>
#include <vector>

#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <MultidimArrays/MultidimArrays.h>

#include <geometry/rotations.h>
#include <geometry/genericbinarypartitioningtree.h>
#include <io/image_io.h>

#include <Eigen/Core>

#include <omp.h>

using PointType = Eigen::Vector3f;
using BinaryTree = StereoVision::Geometry::GenericBSP<PointType, 2, StereoVision::Geometry::BSPObjectWrapper<PointType,float>>;

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

        if (splitted.size() < 3) {
            continue;
        }

        Eigen::Vector3f xyz;

        bool ok = true;

        xyz.x() = splitted[0].toFloat(&ok);

        if (!ok) {
            continue;
        }

        xyz.y() = splitted[1].toFloat(&ok);

        if (!ok) {
            continue;
        }

        xyz.z() = splitted[2].toFloat(&ok);

        if (!ok) {
            continue;
        }

        ret.push_back(xyz);
    }

    return ret;
}

Multidim::Array<float,2> processPointsWithBinaryTree(std::array<int,2> shape,
                                                     BinaryTree::ContainerT & points,
                                                     StereoVision::Geometry::AffineTransform<float> const& img2ptcloud,
                                                     StereoVision::Geometry::AffineTransform<float> const& ptcloud2img,
                                                     float minZ,
                                                     QTextStream & out) {

    Multidim::Array<float,2> raster(shape);

    out << "Start building binary space partitioning tree!" << Qt::endl;

    BinaryTree tree(std::move(points));

    out << "Start creating raster!" << Qt::endl;

    constexpr float distTol = 2;

    #pragma omp parallel for
    for (int i = 0; i < shape[0]; i++) {

        for (int j = 0; j < shape[1]; j++) {
            Eigen::Vector3f imgCoord(i,j,0);
            Eigen::Vector3f minImg(i-distTol,j-distTol,0);
            Eigen::Vector3f maxImg(i+distTol,j+distTol,0);

            Eigen::Vector3f ptCoords = img2ptcloud*imgCoord;
            Eigen::Vector3f min = img2ptcloud*minImg;
            Eigen::Vector3f max = img2ptcloud*maxImg;

            for (int d = 0; d < 2; d++) {
                if (min[d] > max[d]) {
                    float tmp = min[d];
                    min[d] = max[d];
                    max[d] = tmp;
                }
            }

            int pointIdx = tree.closestInRange(ptCoords, min, max);

            if (pointIdx < 0) {
                raster.atUnchecked(i,j) = -1;
                continue;
            }

            Eigen::Vector3f& point = tree[pointIdx];

            Eigen::Vector3f imgPoint = ptcloud2img*point;
            imgPoint.z() = 0;

            float dist = (imgPoint - imgCoord).norm();

            if (dist <= distTol) {
                raster.atUnchecked(i,j) = point.z() - minZ;
            } else {
                raster.atUnchecked(i,j) = -1;
            }
        }
    }

    return raster;

}

Multidim::Array<float,2> processPointsDirect(std::array<int,2> shape,
                                             BinaryTree::ContainerT & points,
                                             StereoVision::Geometry::AffineTransform<float> const& img2ptcloud,
                                             StereoVision::Geometry::AffineTransform<float> const& ptcloud2img,
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

        Eigen::Vector3f const& ptCoords = points[idx];

        Eigen::Vector3f imgCoord = ptcloud2img*ptCoords;

        size_t i = std::round(imgCoord.x());
        size_t j = std::round(imgCoord.y());

        if (i < 0 or j < 0) {
            continue;
        }

        if (i >= shape[0] or j >= shape[1]) {
            continue;
        }

        float z = ptCoords.z() - minZ;

        if (raster.atUnchecked(i,j) < z) {
            raster.atUnchecked(i,j) = z;
        }
    }

    return raster;

}

int main(int argc, char** argv) {

    QTextStream out(stdout);
    QTextStream err(stderr);

    if (argc != 3) {
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

    StereoVision::Geometry::AffineTransform<float> img2ptcloud;
    img2ptcloud.t.x() = minX;
    img2ptcloud.t.y() = maxY;
    img2ptcloud.t.z() = 0;

    img2ptcloud.R = Eigen::Matrix3f::Identity();
    img2ptcloud.R(0,0) = 0;
    img2ptcloud.R(1,1) = 0;
    img2ptcloud.R(1,0) = -invScale;
    img2ptcloud.R(0,1) = invScale;

    StereoVision::Geometry::AffineTransform<float> ptcloud2img;

    ptcloud2img.R = Eigen::Matrix3f::Identity();
    ptcloud2img.R(0,0) = 0;
    ptcloud2img.R(1,1) = 0;
    ptcloud2img.R(1,0) = scale;
    ptcloud2img.R(0,1) = -scale;

    ptcloud2img.t = -ptcloud2img.R*img2ptcloud.t;

    int sX = std::floor(rangeX*scale);
    int sY = std::floor(rangeY*scale);

    double nPixels = static_cast<size_t>(sX)*static_cast<size_t>(sY);

    std::array<int,2> shape = {sY, sX};


    double nPts = points.size();

    bool useBinaryTree = true;

    Multidim::Array<float,2> raster;

    if (useBinaryTree) {
        raster = processPointsWithBinaryTree(shape,
                                             points,
                                             img2ptcloud,
                                             ptcloud2img,
                                             minZ,
                                             out);
    } else {
        raster = processPointsDirect(shape,
                                     points,
                                     img2ptcloud,
                                     ptcloud2img,
                                     minZ,
                                     out);
    }


    out << "Start writing image!" << Qt::endl;

    QString outFile = pointCloudFile + ".tiff";

    ok = StereoVision::IO::writeImage<float, float>(outFile.toStdString(), raster);

    if (!ok) {
        err << "Error while wrinting image" << Qt::endl;
        return 1;
    }

    out << "Image written!" << Qt::endl;

    out.setRealNumberNotation(QTextStream::FixedNotation);
    out.setRealNumberPrecision(4);

    out << "Image to point cloud coordinates transform: " << Qt::endl;
    out << img2ptcloud.R(0,0) << " " << img2ptcloud.R(0,1) << " " << img2ptcloud.R(0,2) << " " << img2ptcloud.t[0] << "\n";
    out << img2ptcloud.R(1,0) << " " << img2ptcloud.R(1,1) << " " << img2ptcloud.R(1,2) << " " << img2ptcloud.t[1] << "\n";
    out << img2ptcloud.R(2,0) << " " << img2ptcloud.R(2,1) << " " << img2ptcloud.R(2,2) << " " << img2ptcloud.t[2] << Qt::endl;

    out << "Point cloud to Image coordinates transform: " << Qt::endl;
    out << ptcloud2img.R(0,0) << " " << ptcloud2img.R(0,1) << " " << ptcloud2img.R(0,2) << " " << ptcloud2img.t[0] << "\n";
    out << ptcloud2img.R(1,0) << " " << ptcloud2img.R(1,1) << " " << ptcloud2img.R(1,2) << " " << ptcloud2img.t[1] << "\n";
    out << ptcloud2img.R(2,0) << " " << ptcloud2img.R(2,1) << " " << ptcloud2img.R(2,2) << " " << ptcloud2img.t[2] << Qt::endl;

    return 0;

}
