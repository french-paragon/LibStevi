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
#include <interpolation/interpolation.h>
#include <geometry/rotations.h>
#include <geometry/genericbinarypartitioningtree.h>
#include <io/image_io.h>
#include <io/pointcloud_io.h>
#include <io/las_pointcloud_io.h>
#include <io/pcd_pointcloud_io.h>

#include <Eigen/Core>

#include <tclap/CmdLine.h>

using PointType = Eigen::Vector3f;
using PointsContainer = std::vector<PointType>;

PointsContainer loadPoints(QString const& fileName) {

    PointsContainer ret;

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

        ret.push_back(point);

        hasMore = ptCloud.pointAccess->gotoNext();
    } while (hasMore);

    return ret;
}

std::tuple<Multidim::Array<float,2>, Multidim::Array<float,3>> computeMinInTiles(PointsContainer const& points,
                                                                                 StereoVision::Geometry::AffineTransform<float> const& ptcloud2img,
                                                                                 std::array<int,2> const& shape) {

    Multidim::Array<float,2> zMap(shape[0], shape[1]);
    Multidim::Array<float,3> shiftMap(shape[0], shape[1],2);

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            zMap.atUnchecked(i,j) = std::numeric_limits<float>::infinity();
            shiftMap.atUnchecked(i,j,0) = 0;
            shiftMap.atUnchecked(i,j,1) = 0;
        }
    }

    for (PointType const& point : points) {

        float z = point.z();

        PointType imgCoord = ptcloud2img*point;

        int i = std::round(imgCoord[0]);
        int j = std::round(imgCoord[1]);

        if (zMap.valueUnchecked(i,j) > z) {
            zMap.atUnchecked(i,j) = z;
            shiftMap.atUnchecked(i,j, 0) = i - imgCoord.x();
            shiftMap.atUnchecked(i,j, 1) = j - imgCoord.y();
        }
    }

    return std::make_tuple(zMap, shiftMap);

}


class ShiftedPointCloudPointAccessInterface : public StereoVision::IO::PointCloudPointAccessInterface {
public:

    inline ShiftedPointCloudPointAccessInterface(std::unique_ptr<StereoVision::IO::PointCloudPointAccessInterface> && sourceInterface,
                                                 StereoVision::Geometry::AffineTransform<float> const& ptcloud2img,
                                                 Multidim::Array<float,2> const& minMask,
                                                 Multidim::Array<float,3> const& shiftMask) :
        PointCloudPointAccessInterface(),
        _point_cloud(std::move(sourceInterface)),
        _ptcloud2img(ptcloud2img),
        _minMask(minMask),
        _shiftMask(shiftMask)
    {
    };
    inline virtual ~ShiftedPointCloudPointAccessInterface() {};

    virtual StereoVision::IO::PtGeometry<StereoVision::IO::PointCloudGenericAttribute> getPointPosition() const override {

        StereoVision::IO::PtGeometry<float> ret = _point_cloud->castedPointGeometry<float>();


        constexpr float(*kernel)(std::array<float,2> const&) = &StereoVision::Interpolation::bicubicKernel<float,2>;
        constexpr int kernelRadius = 2;

        PointType newPt;
        newPt.x() = ret.x;
        newPt.y() = ret.y;
        newPt.z() = ret.z;

        PointType imgCoord = _ptcloud2img*newPt;

        std::array<float,2> coord = {imgCoord.x(), imgCoord.y()};

        float shiftX = StereoVision::Interpolation::interpolateValue<2, float, kernel, kernelRadius>(_shiftMask.sliceView(2,0), coord);
        float shiftY = StereoVision::Interpolation::interpolateValue<2, float, kernel, kernelRadius>(_shiftMask.sliceView(2,1), coord);

        coord[0] += shiftX;
        coord[1] += shiftY;

        float terrain = StereoVision::Interpolation::interpolateValue<2, float, kernel, kernelRadius>(_minMask, coord);

        ret.z -= terrain;

        return StereoVision::IO::PtGeometry<StereoVision::IO::PointCloudGenericAttribute>{ret.x, ret.y, ret.z};
    }

    virtual std::optional<StereoVision::IO::PtColor<StereoVision::IO::PointCloudGenericAttribute>> getPointColor() const override {
        return _point_cloud->getPointColor();
    }

    virtual std::optional<StereoVision::IO::PointCloudGenericAttribute> getAttributeById(int id) const override {
        return _point_cloud->getAttributeById(id);
    }
    virtual std::optional<StereoVision::IO::PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override {
        return _point_cloud->getAttributeByName(attributeName);
    }

    virtual std::vector<std::string> attributeList() const override {
        return _point_cloud->attributeList();
    }

    virtual bool gotoNext() override {
        return _point_cloud->gotoNext();
    }

    virtual bool hasData() const override {
        return _point_cloud->hasData();
    }

protected:

    std::unique_ptr<StereoVision::IO::PointCloudPointAccessInterface> _point_cloud;
    StereoVision::Geometry::AffineTransform<float> const& _ptcloud2img;
    Multidim::Array<float,2> const& _minMask;
    Multidim::Array<float,3> const& _shiftMask;
};

int main(int argc, char** argv) {

    QTextStream out(stdout);
    QTextStream err(stderr);

    QString pointCloudFile;
    QString outCloudFile;
    double tilesize;

    try {
        TCLAP::CmdLine cmd("Make a rough estimation of terrain model and subtract it from source point cloud", '=', "0.0");

        TCLAP::UnlabeledValueArg<std::string> cloudFilePathArg("ptCloudFilePath", "Path where the points cloud is stored", true, "", "local path to point cloud file");

        TCLAP::ValueArg<double> resArg("s","tilesize", "Size of the tile to estimate the terrain model from", true, 10, "double > 0");
        TCLAP::ValueArg<std::string> outArg("o","out", "Out file where to write the resulting cloud to", true, "", "local path to point cloud file");

        cmd.add(cloudFilePathArg);
        cmd.add(resArg);
        cmd.add(outArg);

        cmd.parse(argc, argv);

        pointCloudFile = QString::fromStdString(cloudFilePathArg.getValue());
        outCloudFile = QString::fromStdString(outArg.getValue());
        tilesize = resArg.getValue();

    } catch (TCLAP::ArgException &e) {
        err << "Argument error:" << e.error().c_str() << " for arg " << e.argId().c_str() << Qt::endl;
    }

    out << "Start loading points data!" << Qt::endl;

    PointsContainer points = loadPoints(pointCloudFile);

    if (points.empty()) {
        err << "Could not load point data" << Qt::endl;
        return 1;
    }

    out << "Point data loaded!" << Qt::endl;

    float minX = points[0].x();
    float maxX = points[0].x();
    float minY = points[0].y();
    float maxY = points[0].y();

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
    }

    float rangeX = maxX - minX;
    float rangeY = maxY - minY;

    if (!std::isfinite(rangeX) or !std::isfinite(rangeY)) {
        err << "Invalid point data" << Qt::endl;
        return 1;
    }

    int nTilesX = std::ceil(rangeX/tilesize) + 1;
    int nTilesY = std::ceil(rangeY/tilesize) + 1;

    float scale = 1/tilesize;
    float invScale = tilesize;

    StereoVision::Geometry::AffineTransform<float> img2ptcloud;
    img2ptcloud.t.x() = minX - tilesize/2;
    img2ptcloud.t.y() = minY - tilesize/2;
    img2ptcloud.t.z() = 0;

    img2ptcloud.R = Eigen::Matrix3f::Identity();
    img2ptcloud.R(0,0) = invScale;
    img2ptcloud.R(1,1) = invScale;
    img2ptcloud.R(1,0) = 0;
    img2ptcloud.R(0,1) = 0;

    StereoVision::Geometry::AffineTransform<float> ptcloud2img;

    ptcloud2img.R = Eigen::Matrix3f::Identity();
    ptcloud2img.R(0,0) = scale;
    ptcloud2img.R(1,1) = scale;
    ptcloud2img.R(1,0) = 0;
    ptcloud2img.R(0,1) = 0;

    ptcloud2img.t = -ptcloud2img.R*img2ptcloud.t;

    double nPixels = static_cast<size_t>(rangeX)*static_cast<size_t>(rangeY);

    std::array<int,2> shape = {nTilesX, nTilesY};

    std::tuple<Multidim::Array<float,2>, Multidim::Array<float,3>> tileMinInfos = computeMinInTiles(points, ptcloud2img, shape);

    Multidim::Array<float,2>& minMask = std::get<0>(tileMinInfos);
    Multidim::Array<bool ,2> area2Fill(minMask.shape()[0], minMask.shape()[1]);

    for (int i = 0; i < minMask.shape()[0]; i++) {
        for (int j = 0; j < minMask.shape()[1]; j++) {
            area2Fill.atUnchecked(i,j) = !std::isfinite(minMask.valueUnchecked(i,j));
        }
    }

    Multidim::Array<float,2> inpaintedMinMask = StereoVision::ImageProcessing::nearestInPaintingMonochannel(minMask, area2Fill);

    out << "Depth map produced!" << Qt::endl;

    std::optional<StereoVision::IO::FullPointCloudAccessInterface> optPointCloud =
            StereoVision::IO::openPointCloud(pointCloudFile.toStdString());

    if (!optPointCloud.has_value()) {
        err << "Could not open input point cloud! aborting!" << Qt::endl;
        return 1;
    }

    StereoVision::IO::FullPointCloudAccessInterface& ptCloud = optPointCloud.value();

    ptCloud.pointAccess = std::make_unique<ShiftedPointCloudPointAccessInterface>(std::move(ptCloud.pointAccess),
                                                                                  ptcloud2img,
                                                                                  inpaintedMinMask,
                                                                                  std::get<1>(tileMinInfos));



    //write file

    std::string outFormat = "las";

    if (outCloudFile.toLower().endsWith("pcd")) {
        outFormat = "pcd";
    }

    if (outFormat == "las") {
        bool ok = StereoVision::IO::writePointCloudLas(std::filesystem::path(outCloudFile.toStdString()), ptCloud);

        if (!ok) {
            err << "Error writing point cloud data to " << outCloudFile << "!" << Qt::endl;
            return 1;
        }
    } else if (outFormat == "pcd") {

        StereoVision::IO::PcdDataStorageType dataStorageType = StereoVision::IO::PcdDataStorageType::ascii;

        bool ok = StereoVision::IO::writePointCloudPcd(std::filesystem::path(outCloudFile.toStdString()), ptCloud, dataStorageType);

        if (!ok) {
            err << "Error writing point cloud data to " << outCloudFile << "!" << Qt::endl;
            return 1;
        }
    }

    return 0;
}
