#include <QtTest/QtTest>

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024  Paragon<french.paragon@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "io/pointcloud_io.h"

#include <random>

class TestPointCloudIO: public QObject
{

    Q_OBJECT

private Q_SLOTS:

    void initTestCase();

    void testPointCloudInterfaces();

private:
    std::default_random_engine re;
};

void TestPointCloudIO::initTestCase() {
    std::random_device rd;
    re.seed(rd());
}

void TestPointCloudIO::testPointCloudInterfaces() {
    using PointCloudT = StereoVision::IO::GenericPointCloud<float, void>;

    PointCloudT testPointCloud;

    constexpr int nPoints = 10;

    std::string nPointsAttrName = "nPoints";
    testPointCloud.globalAttribute(nPointsAttrName) = nPoints;

    std::uniform_real_distribution<float> pt_dist(-10,10);

    for (int i = 0; i < nPoints; i++) {
        PointCloudT::Point point;
        point.xyz.x = pt_dist(re);
        point.xyz.y = pt_dist(re);
        point.xyz.z = pt_dist(re);
        testPointCloud.addPoint(point);
    }

    int count = 0;

    StereoVision::IO::FullPointCloudAccessInterface interface(new StereoVision::IO::GenericPointCloudHeaderInterface<float, void>(testPointCloud),
                                                              new StereoVision::IO::GenericPointCloudPointAccessInterface<float, void>(testPointCloud));

    bool running;

    do {

        StereoVision::IO::PointCloudGenericAttribute x = interface.pointAccess->getPointPosition().x;
        StereoVision::IO::PointCloudGenericAttribute y = interface.pointAccess->getPointPosition().y;
        StereoVision::IO::PointCloudGenericAttribute z = interface.pointAccess->getPointPosition().z;

        QVERIFY(std::holds_alternative<float>(x));
        QVERIFY(std::holds_alternative<float>(y));
        QVERIFY(std::holds_alternative<float>(z));

        QCOMPARE(testPointCloud[count].xyz.x, std::get<float>(x));
        QCOMPARE(testPointCloud[count].xyz.y, std::get<float>(y));
        QCOMPARE(testPointCloud[count].xyz.z, std::get<float>(z));

        running = interface.pointAccess->gotoNext();
        count++;

        if (count > nPoints) {
            break;
        }

    } while (running);

    QCOMPARE(count, nPoints);

    std::vector<std::string> attributes = interface.headerAccess->attributeList();

    QCOMPARE(attributes.size(), 1);
    QCOMPARE(attributes[0], nPointsAttrName);
}

QTEST_MAIN(TestPointCloudIO)
#include "testPointCloudIO.moc"
