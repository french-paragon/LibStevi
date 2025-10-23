#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>

#include <cstdint>

#include <tclap/CmdLine.h>

#include "io/pointcloud_io.h"
#include "io/pcd_pointcloud_io.h"

#include "geometry/genericbinarypartitioningtree.h"

struct Point {
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
    uint8_t inCluster;

    float& operator[](int dim) {
        switch(dim) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return r;
        case 4:
            return g;
        case 5:
            return b;
        }
        return x;
    }

    float const& operator[](int dim) const {
        switch(dim) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return r;
        case 4:
            return g;
        case 5:
            return b;
        }
        return x;
    }
};

struct PointSet {


    using BVHStruct = StereoVision::Geometry::GenericBVH<size_t,3,float,std::vector<size_t>>;

    ~PointSet() {
        if (bvhAccelerator != nullptr) {
            delete bvhAccelerator;
        }
    }

    Point operator[](int i) {
        return points[i];
    }

    void insertPoint(Point const& point) {

        if (!point.inCluster) {
            context.push_back(point);
            return;
        }

        if (points.empty()) {
            minX = point.x;
            maxX = point.x;
            minY = point.y;
            maxY = point.y;
            minZ = point.z;
            maxZ = point.z;
        } else {
            minX = std::min(minX,point.x);
            maxX = std::max(maxX,point.x);
            minY = std::min(minY,point.y);
            maxY = std::max(maxY,point.y);
            minZ = std::min(minZ,point.z);
            maxZ = std::max(maxZ,point.z);
        }
        points.push_back(point);
    }

    bool pointIsInContext(Point const& pt, float distance) {
        if (bvhAccelerator != nullptr) {
            return bvhAccelerator->pointIntersectAnItem(pt);
        }

        //if within the bounding box search for closest match
        float dist2 = distance + 1;
        dist2 = dist2*dist2;

        for (int i = 0; i < points.size(); i++) {
            float dx = points[i].x - pt.x;
            float dy = points[i].y - pt.y;
            float dz = points[i].z - pt.z;

            float d2 = dx*dx + dy*dy + dz*dz;

            dist2 = std::min(dist2, d2);
        }

        return dist2 <= (distance*distance);
    }

    void buildBVHAccelerator(float distance) {
        std::vector<size_t> pointsPointers(points.size());

        for (size_t i = 0; i < points.size(); i++) {
            pointsPointers[i] = i;
        }

        BVHStruct::RangeFunc rangeFunc = [this, distance] (int i, int dim) {
            //read the range with distance tolerance for correct intersection computation
            switch (dim) {
            case 0:
                return BVHStruct::Range{points[i].x-distance, points[i].x+distance};
            case 1:
                return BVHStruct::Range{points[i].y-distance, points[i].y+distance};
            case 2:
                return BVHStruct::Range{points[i].z-distance, points[i].z+distance};
            }
            return BVHStruct::Range{points[i].x-distance, points[i].x+distance};
        };

        double d2 = distance*distance;
        BVHStruct::ContainPointFunc containPointFunc = [this, d2] (int ptId, BVHStruct::GenericPoint point) -> bool {
            double dist2 = 0;

            for (int i = 0; i < 3; i++) {
                float delta = point[i] - points[ptId][i];
                dist2 += delta*delta;
            }

            return dist2 <= d2;
        };

        bvhAccelerator = new BVHStruct(pointsPointers, rangeFunc, containPointFunc);
    }

    long id;

    float minX;
    float maxX;
    float minY;
    float maxY;
    float minZ;
    float maxZ;

    std::vector<Point> points;
    std::vector<Point> context;

    BVHStruct* bvhAccelerator = nullptr;
};

class PointSetPointAccessInterface : public StereoVision::IO::PointCloudPointAccessInterface {
public:

    inline PointSetPointAccessInterface(PointSet const& point_cloud) :
        PointCloudPointAccessInterface(),
        _point_cloud(&point_cloud)
    {
        _itPos = 0;
    };    inline PointSetPointAccessInterface(PointSet const* point_cloud) :
        PointCloudPointAccessInterface(),
        _point_cloud(point_cloud)
    {
        _itPos = 0;
    };
    inline virtual ~PointSetPointAccessInterface() {};

    virtual StereoVision::IO::PtGeometry<StereoVision::IO::PointCloudGenericAttribute> getPointPosition() const override {

        StereoVision::IO::PtGeometry<StereoVision::IO::PointCloudGenericAttribute> ret{std::nanf(""), std::nanf(""), std::nanf("")};

        if (_itPos < _point_cloud->points.size()) {
            ret.x = _point_cloud->points[_itPos].x;
            ret.y = _point_cloud->points[_itPos].y;
            ret.z = _point_cloud->points[_itPos].z;

        } else if (_itPos - _point_cloud->points.size() < _point_cloud->context.size()) {
            int i = _itPos - _point_cloud->points.size();
            ret.x = _point_cloud->context[i].x;
            ret.y = _point_cloud->context[i].y;
            ret.z = _point_cloud->context[i].z;
        }

        return ret;
    }
    virtual std::optional<StereoVision::IO::PtColor<StereoVision::IO::PointCloudGenericAttribute>> getPointColor() const override {

        StereoVision::IO::PtColor<StereoVision::IO::PointCloudGenericAttribute> ret{std::nanf(""), std::nanf(""), std::nanf("")};

        if (_itPos < _point_cloud->points.size()) {
            ret.r = _point_cloud->points[_itPos].r;
            ret.g = _point_cloud->points[_itPos].g;
            ret.b = _point_cloud->points[_itPos].b;

        } else if (_itPos - _point_cloud->points.size() < _point_cloud->context.size()) {
            int i = _itPos - _point_cloud->points.size();
            ret.r = _point_cloud->context[i].r;
            ret.g = _point_cloud->context[i].g;
            ret.b = _point_cloud->context[i].b;
        } else {
            return std::nullopt;
        }

        return ret;
    }

    virtual std::optional<StereoVision::IO::PointCloudGenericAttribute> getAttributeById(int id) const override {
        //we have one attribute, and this is if the point is in the cluster
        if (id == 0) {
            if (_itPos < _point_cloud->points.size()) {
                return _point_cloud->points[_itPos].inCluster;
            } else if (_itPos - _point_cloud->points.size() < _point_cloud->context.size()) {
                int i = _itPos - _point_cloud->points.size();
                return _point_cloud->context[i].inCluster;
            } else {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }
    virtual std::optional<StereoVision::IO::PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override {

        if (std::string(attributeName) == "inCluster") {
            return getAttributeById(0);
        }

        return std::nullopt;
    }

    virtual std::vector<std::string> attributeList() const override {
        return std::vector<std::string>{"inCluster"};
    }

    void reset() {
        _itPos = 0;
    }

    virtual bool gotoNext() override {

        size_t end = _point_cloud->points.size() + _point_cloud->context.size();

        _itPos++;

        if (_itPos >= end) {
            return false;
        }

        return true;
    }

    virtual bool hasData() const override {
        size_t end = _point_cloud->points.size() + _point_cloud->context.size();
        return _itPos < end;
    }

protected:

    int _itPos;
    const PointSet* _point_cloud;
};

class PointSetHeaderInterface : public StereoVision::IO::PointCloudHeaderInterface {
public:

    inline PointSetHeaderInterface(PointSet const& point_cloud) :
        PointCloudHeaderInterface(),
        _point_cloud(&point_cloud)
    {
    };
    inline PointSetHeaderInterface(PointSet const* point_cloud) :
        PointCloudHeaderInterface(),
        _point_cloud(point_cloud)
    {
    };
    inline virtual ~PointSetHeaderInterface() {};



    virtual std::optional<StereoVision::IO::PointCloudGenericAttribute> getAttributeById(int id) const {
        return std::nullopt;
    }
    virtual std::optional<StereoVision::IO::PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const {
        return std::nullopt;
    }

    virtual std::vector<std::string> attributeList() const {
        return {};
    }

protected:

    const PointSet* _point_cloud;
};

int main(int argc, char** argv) {

    std::string pointCloudFile;
    std::filesystem::path outDir;
    std::string clustersAttribute;
    float distance;

    try {
        TCLAP::CmdLine cmd("Export split the clusters into a point cloud, keeping some context around each cluster", '=', "0.0");

        TCLAP::UnlabeledValueArg<std::string> cloudFilePathArg("ptCloudFilePath", "Path where the points cloud is stored", true, "", "local path to point cloud file");
        TCLAP::ValueArg<std::string> clustersArg("c", "clusters", "point cloud attribute storing the clusters (assumed to be castable to long)", true, "", "string");
        TCLAP::ValueArg<std::string> outDirArg("o", "out", "output directory were the output point clouds will be stored", true, "", "directory");
        TCLAP::ValueArg<float> contextDistanceArg("d", "distance", "distance tolerance to build the context", false, 1.0, "positive float");

        cmd.add(cloudFilePathArg);
        cmd.add(clustersArg);
        cmd.add(outDirArg);
        cmd.add(contextDistanceArg);

        cmd.parse(argc, argv);

        pointCloudFile = cloudFilePathArg.getValue();
        clustersAttribute = clustersArg.getValue();
        outDir = outDirArg.getValue();
        distance = contextDistanceArg.getValue();

    } catch (TCLAP::ArgException &e) {
        std::cerr << "Argument error:" << e.error().c_str() << " for arg " << e.argId().c_str() << std::endl;
        return 1;
    }

    std::map<long,long> setMatch;
    std::vector<std::unique_ptr<PointSet>> sets;

    size_t nPoints = 0;

    //scope 1, first read
    {
        StatusOptional<StereoVision::IO::FullPointCloudAccessInterface> optPointCloud =
            StereoVision::IO::openPointCloud(pointCloudFile);

        if (!optPointCloud.has_value()) {
            std::cerr << "Could not open point cloud: \"" << pointCloudFile << "\"\n\t Error message is: \"" << optPointCloud.message() << "\"" << std::endl;
            return 1;
        }

        StereoVision::IO::FullPointCloudAccessInterface& ptCloud = optPointCloud.value();

        bool hasMore = true;

        //build the sets;
        do {

            Point point;
            auto geom = ptCloud.pointAccess->castedPointGeometry<double>();
            auto color = ptCloud.pointAccess->castedPointColor<double>();

            point.x = geom.x;
            point.y = geom.y;
            point.z = geom.z;

            if (color.has_value()) {
                point.r = color.value().r;
                point.g = color.value().b;
                point.b = color.value().b;
            } else {
                point.r = 0;
                point.g = 0;
                point.b = 0;
            }

            point.inCluster = true; //first we load the points within the cluster

            long cluster = StereoVision::IO::castedPointCloudAttribute<long>(
                ptCloud.pointAccess->getAttributeByName(clustersAttribute.c_str()).value_or(0));
            hasMore = ptCloud.pointAccess->gotoNext();

            nPoints++;

            if (cluster <= 0) {
                continue; //cluster 0 is assumed to be ground
            }

            if (setMatch.count(cluster) <= 0) {
                setMatch[cluster] = sets.size();
                sets.push_back(std::make_unique<PointSet>());
                sets.back()->id = cluster;
            }

            sets[setMatch[cluster]]->insertPoint(point);

        } while (hasMore);
    }

    std::cout << "Points data loaded (" << nPoints << " points)!" << std::endl;

    //scope 2, second read

    for (std::unique_ptr<PointSet>& ptSet : sets) {
        ptSet->buildBVHAccelerator(distance);
        ptSet->context.reserve(ptSet->points.size()/2); //reserve memory for speedups
    }

    std::cout << "Internal BVH built!" << std::endl;

    int nProcessed = 0;

    {

        using BVHStruct = StereoVision::Geometry::GenericBVH<int,3,float,std::vector<int>>;

        std::vector<int> clustersIdxs(sets.size());

        for (int i = 0; i < sets.size(); i++) {
            clustersIdxs[i] = i;
        }

        BVHStruct::RangeFunc rangeRead = [&sets, &distance] (int i, int dim) {
            //read the range with distance tolerance for correct intersection computation
            switch (dim) {
            case 0:
                return BVHStruct::Range{sets[i]->minX-distance, sets[i]->maxX+distance};
            case 1:
                return BVHStruct::Range{sets[i]->minY-distance, sets[i]->maxY+distance};
            case 2:
                return BVHStruct::Range{sets[i]->minZ-distance, sets[i]->maxZ+distance};
            }
            return BVHStruct::Range{sets[i]->minX, sets[i]->maxX};
        };

        BVHStruct bvhStruct(clustersIdxs, rangeRead);

        StatusOptional<StereoVision::IO::FullPointCloudAccessInterface> optPointCloud =
            StereoVision::IO::openPointCloud(pointCloudFile);

        if (!optPointCloud.has_value()) {
            std::cerr << "Could not open point cloud: \"" << pointCloudFile << "\"\n\t Error message is: \"" << optPointCloud.message() << "\"" << std::endl;
            return 1;
        }

        StereoVision::IO::FullPointCloudAccessInterface& ptCloud = optPointCloud.value();

        bool hasMore = true;

        //build the sets;
        do {

            Point point;
            auto geom = ptCloud.pointAccess->castedPointGeometry<double>();
            auto color = ptCloud.pointAccess->castedPointColor<double>();

            point.x = geom.x;
            point.y = geom.y;
            point.z = geom.z;

            if (color.has_value()) {
                point.r = color.value().r;
                point.g = color.value().b;
                point.b = color.value().b;
            } else {
                point.r = 0;
                point.g = 0;
                point.b = 0;
            }

            point.inCluster = false; //first we load the points within the cluster

            long cluster = StereoVision::IO::castedPointCloudAttribute<long>(
                ptCloud.pointAccess->getAttributeByName(clustersAttribute.c_str()).value_or(0));

            hasMore = ptCloud.pointAccess->gotoNext();

            std::vector<int> clustersIntersecting = bvhStruct.itemsContainingPoint(point);

            for (int i : clustersIntersecting) {
                if (sets[i]->id == cluster) {
                    continue; //do not insert the point into its own cluster
                }

                if (sets[i]->pointIsInContext(point, distance)) {
                    sets[i]->insertPoint(point);
                }
            }

            nProcessed++;

            if (nProcessed%16 == 0) {
                std::cout << "\r" << "Processed point " << nProcessed << "/" << nPoints << std::flush;
            }

        } while (hasMore);

        std::cout << "\r" << "Processed point " << nPoints << "/" << nPoints << std::endl;

        std::cout << "Start writing the sets!" << std::endl;

        //writing the sets
        for (int i = 0; i < sets.size(); i++) {
            std::stringstream fnamestream;
            fnamestream << sets[i]->id << ".pcd";
            std::filesystem::path outFile = outDir / fnamestream.str();

            PointSetPointAccessInterface* writingInterface = new PointSetPointAccessInterface(sets[i].get());
            PointSetHeaderInterface* headerInterface = new PointSetHeaderInterface(sets[i].get());

            StereoVision::IO::FullPointCloudAccessInterface outInterface(headerInterface, writingInterface);

            StereoVision::IO::writePointCloudPcd(outFile, outInterface);
        }
    }

    return 0;
}
