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
#include "io/las_pointcloud_io.h"
#include "io/pcd_pointcloud_io.h"
#include "io/bit_manipulations.h"

#include <QtTest/QtTest>
#include <QtCore/QTemporaryFile>
#include <random>
#include <iostream>

Q_DECLARE_METATYPE(StereoVision::IO::PointCloudGenericAttribute)
Q_DECLARE_METATYPE(std::string)
Q_DECLARE_METATYPE(std::vector<uint8_t>)
Q_DECLARE_METATYPE(std::vector<int16_t>)
Q_DECLARE_METATYPE(std::vector<uint16_t>)
Q_DECLARE_METATYPE(std::vector<int32_t>)
Q_DECLARE_METATYPE(std::vector<uint32_t>)
Q_DECLARE_METATYPE(std::vector<int64_t>)
Q_DECLARE_METATYPE(std::vector<uint64_t>)
Q_DECLARE_METATYPE(std::vector<float>)
Q_DECLARE_METATYPE(std::vector<double>)
Q_DECLARE_METATYPE(std::vector<std::string>)
Q_DECLARE_METATYPE(std::vector<std::byte>)

class TestPointCloudIO: public QObject
{

    Q_OBJECT

private Q_SLOTS:

    void initTestCase();

    void testPointCloudInterfaces();

    void testCastedPointCloudAttribute();
    void testCastedPointCloudAttribute_data();

    //* *************** LAS ***********************
    void testLasExtraBytes();

    void testLasPointCloud();

    //* *************** PCD ***********************
    void testPcdPointCloud();

    //* *************** SDC ***********************
    // void testSdcPointCloud();

    //* *************** MetaCloud ***********************
    // void testMetacloudPointCloud();
    
private:
    std::default_random_engine re;

    QStringList lasFiles;
    QStringList pcdFiles;
    QStringList sdcFiles;
    QStringList metacloudFiles;

    // helper functions

    // make sure that the attribute value is correct

    void verifyLasHeaderAttributeValue(std::string attrName, StereoVision::IO::PointCloudGenericAttribute attr);
    
    void verifyLasPointAttributeValue(std::string attrName, StereoVision::IO::PointCloudGenericAttribute attr);

    void verifyPcdHeaderAttributeValue(std::string attrName, StereoVision::IO::PointCloudGenericAttribute attr);
};

void TestPointCloudIO::initTestCase() {
    std::random_device rd;
    re.seed(rd());

    QString las_path = "@CMAKE_SOURCE_DIR@/test/pointcloud_samples/las";
    QString pcd_path = "@CMAKE_SOURCE_DIR@/test/pointcloud_samples/pcd";
    QString sdc_path = "@CMAKE_SOURCE_DIR@/test/pointcloud_samples/sdc";
    QString metacloud_path = "@CMAKE_SOURCE_DIR@/test/pointcloud_samples/metacloud";
    auto las_dir = QDir(las_path);
    auto pcd_dir = QDir(pcd_path);
    auto sdc_dir = QDir(sdc_path);
    auto metacloud_dir = QDir(metacloud_path);

    lasFiles = las_dir.entryList(QDir::Files);
    for (QString &file : lasFiles) {
        file = las_dir.absoluteFilePath(file);
    }
    
    pcdFiles = pcd_dir.entryList(QDir::Files);
    for (QString &file : pcdFiles) {
        file = pcd_dir.absoluteFilePath(file);
    }
    
    sdcFiles = sdc_dir.entryList(QDir::Files);
    for (QString &file : sdcFiles) {
        file = sdc_dir.absoluteFilePath(file);
    }
    
    metacloudFiles = metacloud_dir.entryList(QDir::Files);
    for (QString &file : metacloudFiles) {
        file = metacloud_dir.absoluteFilePath(file);
    }

    if (lasFiles.isEmpty()) {
        QFAIL(qPrintable(QString("No LAS files found in the test directory: %1").arg(las_dir.absolutePath())));
    }

    if (pcdFiles.isEmpty()) {
        qWarning() << "No PCD files found in the test directory: " << pcd_dir.absolutePath();
    }

    if (sdcFiles.isEmpty()) {
        qWarning() << "No SDC files found in the test directory: " << sdc_dir.absolutePath();
    }

    if (metacloudFiles.isEmpty()) {
        qWarning() << "No Metacloud files found in the test directory: " << metacloud_dir.absolutePath();
    }
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

void TestPointCloudIO::testCastedPointCloudAttribute_data() {
    QTest::addColumn<StereoVision::IO::PointCloudGenericAttribute>("input");

    QTest::addColumn<int8_t>("result_int8_t");
    QTest::addColumn<uint8_t>("result_uint8_t");
    QTest::addColumn<int16_t>("result_int16_t");
    QTest::addColumn<uint16_t>("result_uint16_t");
    QTest::addColumn<int32_t>("result_int32_t");
    QTest::addColumn<uint32_t>("result_uint32_t");
    QTest::addColumn<int64_t>("result_int64_t");
    QTest::addColumn<uint64_t>("result_uint64_t");
    QTest::addColumn<float>("result_float");
    QTest::addColumn<double>("result_double");
    QTest::addColumn<std::string>("result_string");
    QTest::addColumn<std::vector<int8_t>>("result_vector_int8_t");
    QTest::addColumn<std::vector<uint8_t>>("result_vector_uint8_t");
    QTest::addColumn<std::vector<int16_t>>("result_vector_int16_t");
    QTest::addColumn<std::vector<uint16_t>>("result_vector_uint16_t");
    QTest::addColumn<std::vector<int32_t>>("result_vector_int32_t");
    QTest::addColumn<std::vector<uint32_t>>("result_vector_uint32_t");
    QTest::addColumn<std::vector<int64_t>>("result_vector_int64_t");
    QTest::addColumn<std::vector<uint64_t>>("result_vector_uint64_t");
    QTest::addColumn<std::vector<float>>("result_vector_float");
    QTest::addColumn<std::vector<double>>("result_vector_double");
    QTest::addColumn<std::vector<std::string>>("result_vector_string");
    QTest::addColumn<std::vector<std::byte>>("result_vector_byte");

    // Row 1: Single scalar value
    QTest::newRow("integer_value")
        << StereoVision::IO::PointCloudGenericAttribute(int32_t{42}) 
        << int8_t(42) << uint8_t(42)
        << int16_t(42) << uint16_t(42)
        << int32_t(42) << uint32_t(42)
        << int64_t(42) << uint64_t(42)
        << float(42.0f) << double(42.0)
        << std::string("42") 
        << std::vector<int8_t>{42}        
        << std::vector<uint8_t>{42}
        << std::vector<int16_t>{42}
        << std::vector<uint16_t>{42}
        << std::vector<int32_t>{42}
        << std::vector<uint32_t>{42}
        << std::vector<int64_t>{42}
        << std::vector<uint64_t>{42}
        << std::vector<float>{42.0f}
        << std::vector<double>{42.0}
        << std::vector<std::string>{"42"}
        << std::vector<std::byte>{};

    QTest::newRow("integer_8bit_value")
        << StereoVision::IO::PointCloudGenericAttribute(uint8_t{6}) 
        << int8_t(6) << uint8_t(6)
        << int16_t(6) << uint16_t(6)
        << int32_t(6) << uint32_t(6)
        << int64_t(6) << uint64_t(6)
        << float(6.0f) << double(6.0)
        << std::string("6") 
        << std::vector<int8_t>{6}        
        << std::vector<uint8_t>{6}
        << std::vector<int16_t>{6}
        << std::vector<uint16_t>{6}
        << std::vector<int32_t>{6}
        << std::vector<uint32_t>{6}
        << std::vector<int64_t>{6}
        << std::vector<uint64_t>{6}
        << std::vector<float>{6.0f}
        << std::vector<double>{6.0}
        << std::vector<std::string>{"6"}
        << std::vector<std::byte>{};

    QTest::newRow("floating_point_value")
        << StereoVision::IO::PointCloudGenericAttribute(double{14.4}) 
        << int8_t(14) << uint8_t(14)
        << int16_t(14) << uint16_t(14)
        << int32_t(14) << uint32_t(14)
        << int64_t(14) << uint64_t(14)
        << float(14.4f) << double(14.4)
        << std::string("14.4")   
        << std::vector<int8_t>{14}    
        << std::vector<uint8_t>{14}
        << std::vector<int16_t>{14}
        << std::vector<uint16_t>{14}
        << std::vector<int32_t>{14}
        << std::vector<uint32_t>{14}
        << std::vector<int64_t>{14}
        << std::vector<uint64_t>{14}
        << std::vector<float>{14.4f}
        << std::vector<double>{14.4}
        << std::vector<std::string>{"14.4"}
        << std::vector<std::byte>{};

    QTest::newRow("string_single_value")
        << StereoVision::IO::PointCloudGenericAttribute(std::string{"3.9"}) 
        << int8_t(3) << uint8_t(3)
        << int16_t(3) << uint16_t(3)
        << int32_t(3) << uint32_t(3)
        << int64_t(3) << uint64_t(3)
        << float(3.9f) << double(3.9)
        << std::string("3.9")   
        << std::vector<int8_t>{3}    
        << std::vector<uint8_t>{3}
        << std::vector<int16_t>{3}
        << std::vector<uint16_t>{3}
        << std::vector<int32_t>{3}
        << std::vector<uint32_t>{3}
        << std::vector<int64_t>{3}
        << std::vector<uint64_t>{3}
        << std::vector<float>{3.9f}
        << std::vector<double>{3.9}
        << std::vector<std::string>{"3.9"}
        << std::vector<std::byte>{};

    QTest::newRow("string_multiple_values")
        << StereoVision::IO::PointCloudGenericAttribute(std::string{"1 3.55 1e1"}) 
        << int8_t{1} << uint8_t{1}
        << int16_t{1} << uint16_t{1}
        << int32_t{1} << uint32_t{1}
        << int64_t{1} << uint64_t{1}
        << float{1.0f} << double{1}
        << std::string("1 3.55 1e1")   
        << std::vector<int8_t>{1, 3, 10}    
        << std::vector<uint8_t>{1, 3, 10}
        << std::vector<int16_t>{1, 3, 10}
        << std::vector<uint16_t>{1, 3, 10}
        << std::vector<int32_t>{1, 3, 10}
        << std::vector<uint32_t>{1, 3, 10}
        << std::vector<int64_t>{1, 3, 10}
        << std::vector<uint64_t>{1, 3, 10}
        << std::vector<float>{1.0f, 3.55f, 10.0f}
        << std::vector<double>{1.0, 3.55, 10.0}
        << std::vector<std::string>{"1", "3.55", "1e1"}
        << std::vector<std::byte>{};

    // Row 2: Vector of values
    QTest::newRow("vector_multiple_values")
        << StereoVision::IO::PointCloudGenericAttribute(std::vector<int32_t>{10, 20, 30})
        << int8_t{10} << uint8_t{10}
        << int16_t{10} << uint16_t{10}
        << int32_t{10} << uint32_t{10}
        << int64_t{10} << uint64_t{10}
        << float{10} << double{10}
        << std::string("10 20 30")                  
        << std::vector<int8_t>{10, 20, 30}    
        << std::vector<uint8_t>{10, 20, 30}
        << std::vector<int16_t>{10, 20, 30}
        << std::vector<uint16_t>{10, 20, 30}
        << std::vector<int32_t>{10, 20, 30}
        << std::vector<uint32_t>{10, 20, 30}
        << std::vector<int64_t>{10, 20, 30}
        << std::vector<uint64_t>{10, 20, 30}
        << std::vector<float>{10.0f, 20.0f, 30.0f}
        << std::vector<double>{10.0, 20.0, 30.0}
        << std::vector<std::string>{"10", "20", "30"}
        << std::vector<std::byte>{};

    QTest::newRow("vector_single_value")
        << StereoVision::IO::PointCloudGenericAttribute(std::vector<float>{10.2f})
        << int8_t{10} << uint8_t{10}
        << int16_t{10} << uint16_t{10}
        << int32_t{10} << uint32_t{10}
        << int64_t{10} << uint64_t{10}
        << float{10.2f} << double{10.2f}
        << std::string("10.2")                  
        << std::vector<int8_t>{10}    
        << std::vector<uint8_t>{10}
        << std::vector<int16_t>{10}
        << std::vector<uint16_t>{10}
        << std::vector<int32_t>{10}
        << std::vector<uint32_t>{10}
        << std::vector<int64_t>{10}
        << std::vector<uint64_t>{10}
        << std::vector<float>{10.2f}
        << std::vector<double>{10.2f}
        << std::vector<std::string>{"10.2"}
        << std::vector<std::byte>{};

    std::vector<std::byte> bytes{std::byte{0x3F}, std::byte{0xB}, std::byte{0x0}, std::byte{0xAD}};
    QTest::newRow("vector_bytes")
    << StereoVision::IO::PointCloudGenericAttribute(bytes)
    << int8_t{} << uint8_t{}
    << int16_t{} << uint16_t{} // undefied
    << int32_t{} << uint32_t{} // undefied
    << int64_t{} << uint64_t{} // undefied
    << float{} << double{} // undefied
    << std::string("0x3f0b00ad")                  
    << std::vector<int8_t>{
            StereoVision::IO::bit_cast<int8_t>(bytes[0]),
            StereoVision::IO::bit_cast<int8_t>(bytes[1]),
            StereoVision::IO::bit_cast<int8_t>(bytes[2]),
            StereoVision::IO::bit_cast<int8_t>(bytes[3])}    
    << std::vector<uint8_t>{
            StereoVision::IO::bit_cast<uint8_t>(bytes[0]),
            StereoVision::IO::bit_cast<uint8_t>(bytes[1]),    
            StereoVision::IO::bit_cast<uint8_t>(bytes[2]),
            StereoVision::IO::bit_cast<uint8_t>(bytes[3])}
    << std::vector<int16_t>{} // undefied
    << std::vector<uint16_t>{} // undefied
    << std::vector<int32_t>{} // undefied
    << std::vector<uint32_t>{} // undefied
    << std::vector<int64_t>{} // undefied
    << std::vector<uint64_t>{} // undefied
    << std::vector<float>{} // undefied
    << std::vector<double>{} // undefied
    << std::vector<std::string>{"0x3f", "0x0b", "0x00", "0xad"}
    << bytes;
}

void TestPointCloudIO::testCastedPointCloudAttribute() {
    QFETCH(StereoVision::IO::PointCloudGenericAttribute, input);
    QFETCH(int8_t, result_int8_t);
    QFETCH(uint8_t, result_uint8_t);
    QFETCH(int16_t, result_int16_t);
    QFETCH(uint16_t, result_uint16_t);
    QFETCH(int32_t, result_int32_t);
    QFETCH(uint32_t, result_uint32_t);
    QFETCH(int64_t, result_int64_t);
    QFETCH(uint64_t, result_uint64_t);
    QFETCH(float, result_float);
    QFETCH(double, result_double);
    QFETCH(std::string, result_string);
    QFETCH(std::vector<int8_t>, result_vector_int8_t);
    QFETCH(std::vector<uint8_t>, result_vector_uint8_t);
    QFETCH(std::vector<int16_t>, result_vector_int16_t);
    QFETCH(std::vector<uint16_t>, result_vector_uint16_t);
    QFETCH(std::vector<int32_t>, result_vector_int32_t);
    QFETCH(std::vector<uint32_t>, result_vector_uint32_t);
    QFETCH(std::vector<int64_t>, result_vector_int64_t);
    QFETCH(std::vector<uint64_t>, result_vector_uint64_t);
    QFETCH(std::vector<float>, result_vector_float);
    QFETCH(std::vector<double>, result_vector_double);
    QFETCH(std::vector<std::string>, result_vector_string);
    QFETCH(std::vector<std::byte>, result_vector_byte);

    // Test each conversion
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<int8_t>(input), result_int8_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<uint8_t>(input), result_uint8_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<int16_t>(input), result_int16_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<uint16_t>(input), result_uint16_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<int32_t>(input), result_int32_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<uint32_t>(input), result_uint32_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<int64_t>(input), result_int64_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<uint64_t>(input), result_uint64_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<float>(input), result_float);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<double>(input), result_double);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::string>(input), result_string);

    // Test vector conversions
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<int8_t>>(input), result_vector_int8_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<uint8_t>>(input), result_vector_uint8_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<int16_t>>(input), result_vector_int16_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<uint16_t>>(input), result_vector_uint16_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<int32_t>>(input), result_vector_int32_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<uint32_t>>(input), result_vector_uint32_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<int64_t>>(input), result_vector_int64_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<uint64_t>>(input), result_vector_uint64_t);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<float>>(input), result_vector_float);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<double>>(input), result_vector_double);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<std::string>>(input), result_vector_string);
    QCOMPARE(StereoVision::IO::castedPointCloudAttribute<std::vector<std::byte>>(input), result_vector_byte);
}

void TestPointCloudIO::testLasExtraBytes() {
    std::array<uint8_t, 2> reserved{};
    uint8_t data_type = 1;
    uint8_t options = 2;
    std::array<char, 32> name{"sample"};
    std::array<uint8_t, 4> unused{0, 1, 2, 3};
    std::array<std::byte, 8> no_data{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
    std::array<uint8_t, 16> deprecated1{"asdf"};
    std::array<std::byte, 8> min {std::byte{43}, std::byte{44}, std::byte{45}, std::byte{46}, std::byte{47}, std::byte{48}, std::byte{49}, std::byte{50}};
    std::array<uint8_t, 16> deprecated2{"5678"};
    std::array<std::byte, 8> max{std::byte{51}, std::byte{52}, std::byte{53}, std::byte{54}, std::byte{55}, std::byte{56}, std::byte{57}, std::byte{58}};
    std::array<uint8_t, 16> deprecated3{"1234"};
    double scale = 345.678;
    std::array<uint8_t, 16> deprecated4{"*+*"};
    double offset = 123.456;
    std::array<uint8_t, 16> deprecated5{"deprecated"};
    std::array<char, 32> description{"test description"};
    StereoVision::IO::LasExtraBytesDescriptor extraBytes{reserved, data_type, options, name, unused, no_data, deprecated1, min, deprecated2, max, deprecated3, scale, deprecated4, offset, deprecated5, description};
    // convert to a raw byte array
    auto rawBytes = extraBytes.toBytes();
    // reconvert it
    StereoVision::IO::LasExtraBytesDescriptor extraBytes2{reinterpret_cast<char*>(rawBytes.data())};
    // display
    auto rawBytes2 = extraBytes2.toBytes();

    QCOMPARE(rawBytes, rawBytes2);

    // test other constructors
    uint8_t data_type_3 = 9; // float
    std::string name_3 = "testFloat";
    std::string description_3 = "descriptionTest";
    double no_data_3 = 6;
    double min_3 = -324;
    double max_3 = 324;
    char* min_3_rawPtr = reinterpret_cast<char*>(&min_3);
    char* max_3_rawPtr = reinterpret_cast<char*>(&max_3);
    char* no_data_3_rawPtr = reinterpret_cast<char*>(&no_data_3);
    auto min_3_raw = StereoVision::IO::arrayFromBytes<std::byte, 8>(min_3_rawPtr);
    auto max_3_raw = StereoVision::IO::arrayFromBytes<std::byte, 8>(max_3_rawPtr);
    auto no_data_3_raw = StereoVision::IO::arrayFromBytes<std::byte, 8>(no_data_3_rawPtr);
    double scale_3 = 3;
    double offset_3 = 5;

    StereoVision::IO::LasExtraBytesDescriptor extraBytes3{data_type_3, name_3, 0, description_3, no_data_3, min_3, max_3, scale_3, offset_3};
    // convert to raw bytes
    auto rawBytes3 = extraBytes3.toBytes();
    // expected extra byte 
    StereoVision::IO::LasExtraBytesDescriptor expectedExtraBytes3 {std::array<uint8_t, 2>{}, uint8_t{9}, uint8_t{0b00011111}, std::array<char, 32>{"testFloat"},
        std::array<uint8_t, 4>{}, no_data_3_raw, std::array<uint8_t, 16>{}, min_3_raw, std::array<uint8_t, 16>{},
        max_3_raw, std::array<uint8_t, 16>{}, scale_3, std::array<uint8_t, 16>{}, offset_3, std::array<uint8_t, 16>{}, std::array<char, 32>{"descriptionTest"}};
    // raw bytes
    auto expectedRawBytes3 = expectedExtraBytes3.toBytes();

    // compare them
    QCOMPARE(rawBytes3, expectedRawBytes3);
    
    // again, read + convert
    StereoVision::IO::LasExtraBytesDescriptor extraBytes4{reinterpret_cast<char*>(rawBytes3.data())};
    auto rawBytes4 = extraBytes4.toBytes();
    QCOMPARE(rawBytes4, expectedRawBytes3);

    // same thing with only the data_type and name
    StereoVision::IO::LasExtraBytesDescriptor extraBytes5{data_type_3, name_3};
    // convert to raw bytes
    auto rawBytes5 = extraBytes5.toBytes();
    // expected extra byte 
    StereoVision::IO::LasExtraBytesDescriptor expectedExtraBytes5 {std::array<uint8_t, 2>{}, uint8_t{9}, uint8_t{}, std::array<char, 32>{"testFloat"},
        std::array<uint8_t, 4>{}, std::array<std::byte, 8>{}, std::array<uint8_t, 16>{}, std::array<std::byte, 8>{}, std::array<uint8_t, 16>{},
        std::array<std::byte, 8>{}, std::array<uint8_t, 16>{}, 0, std::array<uint8_t, 16>{}, 0, std::array<uint8_t, 16>{}, std::array<char, 32>{}};
    // raw bytes
    auto expectedRawBytes5 = expectedExtraBytes5.toBytes();

    // compare them
    QCOMPARE(rawBytes5, expectedRawBytes5);
    
    // again, read + convert
    StereoVision::IO::LasExtraBytesDescriptor extraBytes6{reinterpret_cast<char*>(rawBytes5.data())};
    auto rawBytes6 = extraBytes6.toBytes();
    QCOMPARE(rawBytes6, expectedRawBytes5);
}

void TestPointCloudIO::testLasPointCloud() {
    /*  
        Open las test files + open pointcloud files in various formats, write them to lass, and read them back. 
        verify that the interface is not nullopt and that the header and the point cloud are not nullptr.
        verify the attributes of the header based on version.
        Get the format number and verify the points attributes based on this (The position and the color attributes
        should also be hidden).
        verify that the point position is "correct" (not NaN) and that the color is either nullopt if there is no color
        for this format or a valid color if there is color for this format.
        Verify that the number of points is the same as the number of points in the header.
        Verify that the number of VLRs/EVLRs is the same as the number of VLRs/EVLRs in the header.
        Verify that the extra bytes attributes, if any, are present in the point cloud and that their type is correct.
    */
    const std::vector<std::string> expectedHeaderAttributes_v1_4 = 
        {"fileSignature", "fileSourceID", "globalEncoding", "projectID_GUID_Data1", "projectID_GUID_Data2",
        "projectID_GUID_Data3", "projectID_GUID_Data4", "versionMajor", "versionMinor", "systemIdentifier",
        "generatingSoftware", "fileCreationDayOfYear", "fileCreationYear", "headerSize", "offsetToPointData",
        "numberOfVariableLengthRecords", "pointDataRecordFormat", "pointDataRecordLength",
        "legacyNumberOfPointRecords", "legacyNumberOfPointsByReturn", "xScaleFactor", "yScaleFactor",
        "zScaleFactor", "xOffset", "yOffset", "zOffset", "maxX", "minX", "maxY", "minY", "maxZ", "minZ",
        "startOfWaveformDataPacketRecord", "startOfFirstExtendedVariableLengthRecord",
        "numberOfExtendedVariableLengthRecords", "numberOfPointRecords", "numberOfPointsByReturn"};

    const std::vector<std::vector<std::string>> expectedPointAttributes = {
        // format 0
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "syntheticFlag",
        "keyPointFlag", "withheldFlag", "scanAngleRank", "userData", "pointSourceID"},
        // format 1
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "syntheticFlag",
        "keyPointFlag", "withheldFlag", "scanAngleRank", "userData", "pointSourceID", "GPSTime"},
        // format 2
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "syntheticFlag",
        "keyPointFlag", "withheldFlag", "scanAngleRank", "userData", "pointSourceID", "red", "green",
        "blue"},
        // format 3
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "syntheticFlag",
        "keyPointFlag", "withheldFlag", "scanAngleRank", "userData", "pointSourceID", "GPSTime", "red",
        "green", "blue"},
        // format 4
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "syntheticFlag",
        "keyPointFlag", "withheldFlag", "scanAngleRank", "userData", "pointSourceID", "GPSTime",
        "wavePacketDescriptorIndex", "byteOffsetToWaveformData", "waveformPacketSizeInBytes",
        "returnPointWaveformLocation", "parametricDx", "parametricDy", "parametricDz"},
        // format 5
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "syntheticFlag",
        "keyPointFlag", "withheldFlag", "scanAngleRank", "userData", "pointSourceID", "GPSTime", "red",
        "green", "blue", "wavePacketDescriptorIndex", "byteOffsetToWaveformData", "waveformPacketSizeInBytes",
        "returnPointWaveformLocation", "parametricDx", "parametricDy", "parametricDz"},
        // format 6
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "syntheticFlag", "keyPointFlag", "withheldFlag", "overlapFlag", "scannerChannel",
        "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "userData", "scanAngle",
        "pointSourceID", "GPSTime"},
        // format 7
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "syntheticFlag", "keyPointFlag", "withheldFlag", "overlapFlag", "scannerChannel",
        "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "userData", "scanAngle",
        "pointSourceID", "GPSTime", "red", "green", "blue"},
        // format 8
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "syntheticFlag", "keyPointFlag", "withheldFlag", "overlapFlag", "scannerChannel",
        "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "userData", "scanAngle",
        "pointSourceID", "GPSTime", "red", "green", "blue", "NIR"},
        // format 9
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "syntheticFlag", "keyPointFlag", "withheldFlag", "overlapFlag", "scannerChannel",
        "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "userData", "scanAngle",
        "pointSourceID", "GPSTime", "wavePacketDescriptorIndex",
        "byteOffsetToWaveformData", "waveformPacketSizeInBytes", "returnPointWaveformLocation", "parametricDx",
        "parametricDy", "parametricDz"},
        // format 10
        {"x", "y", "z", "intensity", "returnNumber",
        "numberOfReturns", "syntheticFlag", "keyPointFlag", "withheldFlag", "overlapFlag", "scannerChannel",
        "scanDirectionFlag", "edgeOfFlightLineFlag", "classification", "userData", "scanAngle",
        "pointSourceID", "GPSTime", "red", "green", "blue", "NIR", "wavePacketDescriptorIndex",
        "byteOffsetToWaveformData", "waveformPacketSizeInBytes", "returnPointWaveformLocation", "parametricDx",
        "parametricDy", "parametricDz"}
    };

    auto originalFiles = lasFiles + pcdFiles + sdcFiles + metacloudFiles;
    QStringList rewrittenFiles;
    std::vector<QTemporaryFile> tempFiles(originalFiles.size()); // temp files that will be auto removed
    for (size_t i = 0; i < originalFiles.size(); i++) {
        // test if the file is open and get its path
        QVERIFY2(tempFiles[i].open(), qPrintable(QString("Failed to open temporary file: %1")
            .arg(tempFiles[i].fileName())));
        QString tempFilePath = tempFiles[i].fileName();
        // save the path
        rewrittenFiles.append(tempFilePath);
        // close the file
        tempFiles[i].close();
        // open the original point cloud and write it to the temporary file
        auto fullPointCloud = StereoVision::IO::openPointCloud(originalFiles[i].toStdString());
        QVERIFY2(fullPointCloud.has_value(),
            qPrintable(QString("Failed to open point cloud: %1").arg(originalFiles[i])));
        
        auto& pointAccess = fullPointCloud->pointAccess;
        auto& headerAccess = fullPointCloud->headerAccess;
        QVERIFY2(pointAccess != nullptr, qPrintable(QString("The point cloud is null for file: %1")
            .arg(originalFiles[i])));
        QVERIFY2(headerAccess != nullptr, qPrintable(QString("The header is null for file: %1").arg(originalFiles[i])));

        // write the point cloud to the temporary file
        QVERIFY2(StereoVision::IO::writePointCloudLas(tempFilePath.toStdString(), fullPointCloud.value()),
            qPrintable(QString("Failed to write point cloud: %1 to file: %2").arg(originalFiles[i]).arg(tempFilePath)));
    }

    auto allFiles = lasFiles + rewrittenFiles;
    for (size_t i = 0; i < allFiles.size(); i++) {
        auto filePath = allFiles[i];
        bool isRewrittenFile = i >= lasFiles.size();
        auto originalFileIndex = i - lasFiles.size();
        // open the las file
        auto fullPointCloud = StereoVision::IO::openPointCloudLas(filePath.toStdString());
        // verify that the interface is not nullopt and that the header and the point cloud are not nullptr
        QVERIFY2(fullPointCloud != std::nullopt, qPrintable(QString("Failed to open las file: %1").arg(filePath)));
        auto& pointAccess = fullPointCloud->pointAccess;
        auto& headerAccess = fullPointCloud->headerAccess;
        QVERIFY2(pointAccess != nullptr, qPrintable(QString("The point cloud is null for file: %1").arg(filePath)));
        QVERIFY2(headerAccess != nullptr, qPrintable(QString("The header is null for file: %1").arg(filePath)));

        // try to get the version
        auto versionMajorOpt = headerAccess->getAttributeByName("versionMajor");
        auto versionMinorOpt = headerAccess->getAttributeByName("versionMinor");
        QVERIFY2(versionMajorOpt.has_value(),
            qPrintable(QString("Failed to get versionMajor for file: %1").arg(filePath)));
        QVERIFY2(versionMinorOpt.has_value(),
            qPrintable(QString("Failed to get versionMinor for file: %1").arg(filePath)));

        auto versionMajor = StereoVision::IO::castedPointCloudAttribute<uint8_t>(versionMajorOpt.value());
        auto versionMinor = StereoVision::IO::castedPointCloudAttribute<uint8_t>(versionMinorOpt.value());
        // verify the attributes of the header based on version
        auto expectedHeaderAttributes = expectedHeaderAttributes_v1_4;
        if (versionMajor == 1 && versionMinor <= 4) {
            expectedHeaderAttributes.resize(33);
        }
        
        for (auto& expectedAttrName: expectedHeaderAttributes) {
            // attribute should exist
            auto attrOpt = headerAccess->getAttributeByName(expectedAttrName.c_str());
            QVERIFY2(attrOpt.has_value(),
                qPrintable(QString("Failed to get the header attribute: %1 for file: %2")
                    .arg(expectedAttrName.c_str()).arg(filePath)));
            auto attr = attrOpt.value();

            verifyLasHeaderAttributeValue(expectedAttrName, attr);
        }

        // get the format number
        auto formatNumberOpt = headerAccess->getAttributeByName("pointDataRecordFormat");
        QVERIFY2(formatNumberOpt.has_value(),
            qPrintable(QString("Failed to get formatNumber for file: %1").arg(filePath)));
        auto formatNumber = StereoVision::IO::castedPointCloudAttribute<uint16_t>(formatNumberOpt.value());
        
        size_t nbPoints = 1;
        // iterate on all the points
        do {
            for (auto& expectedAttrName: expectedPointAttributes[formatNumber]) {
                // attribute should exist
                auto attrOpt = pointAccess->getAttributeByName(expectedAttrName.c_str());
                // if attribute is color or x,y,z then it should be hidden BUT the color/xyz attributes should exist.
                // the position always exists and should not be NaN unless we have a no data point.
                if (expectedAttrName == "red" || expectedAttrName == "green" || expectedAttrName == "blue") {
                    // the color is hidden
                    QVERIFY2(!attrOpt.has_value(),
                        qPrintable(QString("The color attribute should be hidden for file: %1").arg(filePath)));
                    // the colorPoint should exist
                    auto pointColorOpt = pointAccess->getPointColor();
                    QVERIFY2(pointColorOpt.has_value(),
                        qPrintable(QString("Failed to get the point color for file: %1").arg(filePath)));
                    auto pointColor = pointColorOpt.value();
                    // verify the type
                    std::holds_alternative<uint8_t>(pointColor.r);
                    std::holds_alternative<uint8_t>(pointColor.g);
                    std::holds_alternative<uint8_t>(pointColor.b);
                    std::holds_alternative<StereoVision::IO::EmptyParam>(pointColor.a);
                } else if (expectedAttrName == "x" || expectedAttrName == "y" || expectedAttrName == "z") {
                    // the position is hidden
                    QVERIFY2(!attrOpt.has_value(),
                        qPrintable(QString("The position attribute should be hidden for file: %1").arg(filePath)));
                    auto pointPosition = pointAccess->getPointPosition();
                    // verify the type
                    std::holds_alternative<double>(pointPosition.x);
                    std::holds_alternative<double>(pointPosition.y);
                    std::holds_alternative<double>(pointPosition.z);
                } else {
                    QVERIFY2(attrOpt.has_value(),
                        qPrintable(QString("Failed to get the point attribute: %1 for file: %2")
                            .arg(expectedAttrName.c_str()).arg(filePath)));
                    auto attr = attrOpt.value();

                    verifyLasPointAttributeValue(expectedAttrName, attr);
                }
            }
            nbPoints++;
        } while (pointAccess->gotoNext());
        // TODO: test the number of points and the number of points by return

        // compare with the original point cloud if it is a rewritten point cloud
        if (isRewrittenFile) {
            auto originalFilePath = originalFiles[originalFileIndex];
            auto originalFullPointCloud = StereoVision::IO::openPointCloud(originalFilePath.toStdString());
            auto rewrittenFullPointCloud = StereoVision::IO::openPointCloudLas(filePath.toStdString());
            QVERIFY2(originalFullPointCloud.has_value(),
                qPrintable(QString("Failed to open the original point cloud: %1").arg(originalFilePath)));
            QVERIFY2(rewrittenFullPointCloud.has_value(),
                qPrintable(QString("Failed to open the rewritten point cloud: %1").arg(filePath)));
            auto& originalPointAccess = originalFullPointCloud.value().pointAccess;
            auto& rewrittenPointAccess = rewrittenFullPointCloud.value().pointAccess;
            auto* castedLasRewrittenPointAccess
                = dynamic_cast<StereoVision::IO::LasPointCloudPoint*>(rewrittenPointAccess.get());
            QVERIFY2(castedLasRewrittenPointAccess != nullptr,
                qPrintable(QString("Failed to cast the point cloud to a las point cloud for file: %1").arg(filePath)));
            // iterate on all the points
            while (true) {
                for (auto& originalAttrName: originalPointAccess->attributeList()) {
                    auto originalAttrOpt = originalPointAccess->getAttributeByName(originalAttrName.c_str());
                    QVERIFY2(originalAttrOpt.has_value(),
                        qPrintable(QString("Failed to get the original point attribute: %1 for file: %2")
                            .arg(originalAttrName.c_str()).arg(originalFilePath)));
                    auto originalAttr = originalAttrOpt.value();
                    
                    // test if it is a valid las type. If not, the attribute will not be present
                    if (std::holds_alternative<uint8_t>(originalAttr) ||
                        std::holds_alternative<uint16_t>(originalAttr) ||
                        std::holds_alternative<uint32_t>(originalAttr) ||
                        std::holds_alternative<uint64_t>(originalAttr) ||
                        std::holds_alternative<int8_t>(originalAttr) ||
                        std::holds_alternative<int16_t>(originalAttr) ||
                        std::holds_alternative<int32_t>(originalAttr) ||
                        std::holds_alternative<int64_t>(originalAttr) ||
                        std::holds_alternative<float>(originalAttr) ||
                        std::holds_alternative<double>(originalAttr)) {

                        auto rewrittenAttrOpt = rewrittenPointAccess->getAttributeByName(originalAttrName.c_str());
                        QVERIFY2(rewrittenAttrOpt.has_value(),
                            qPrintable(QString("Failed to get the rewritten point attribute: %1 for file: %2")
                                .arg(originalAttrName.c_str()).arg(filePath)));
                        
                        // visit the attribute
                        std::visit([&](auto& rewrittenAttr) {
                            using rewritten_type = std::decay_t<decltype(rewrittenAttr)>;
                            if constexpr (std::is_integral_v<rewritten_type> ||
                                          std::is_floating_point_v<rewritten_type>) {
                                // cast the original type to the rewritten type
                                auto castedAttribute
                                    = StereoVision::IO::castedPointCloudAttribute<rewritten_type>(originalAttr);
                                // compare the values
                                QCOMPARE(castedAttribute, rewrittenAttr);
                            } else {
                                QFAIL(qPrintable(
                                    QString("Unsupported type for attribute: %1").arg(originalAttrName.c_str())));
                            }
                        }, rewrittenAttrOpt.value());
                    }
                }
                auto comparePositionComponent = [&](auto originalComponent, auto rewrittenComponent, auto scaleFactor,
                    auto offset) {
                    
                        // visit the attributes
                    std::visit([&](auto& newComponent) {
                        using new_type = std::decay_t<decltype(newComponent)>;
                        if constexpr (std::is_integral_v<new_type> ||
                                    std::is_floating_point_v<new_type>) {
                            auto castedDoubleOriginalComponent
                                = StereoVision::IO::castedPointCloudAttribute<double>(originalComponent);
                            // both values should be nan
                            if (std::isnan(castedDoubleOriginalComponent)) {
                                QVERIFY2(std::isnan(static_cast<double>(newComponent)),
                                    qPrintable(
                                        QString("Original position component should be nan for file: %1")
                                        .arg(originalFilePath)));
                            } else {
                                // precision depend on the scale factor and the offset
                                // rescale + to int
                                int64_t originalScaledInt = 
                                    static_cast<int64_t>(
                                        std::round(
                                            (StereoVision::IO::castedPointCloudAttribute<double>(originalComponent)
                                            - offset) / scaleFactor));

                                int64_t newScaledInt = 
                                    static_cast<int64_t>(
                                        std::round(
                                            (static_cast<double>(newComponent)
                                            - offset) / scaleFactor));
                                // compare the values
                                // QCOMPARE(originalScaledInt, newScaledInt);
                                QVERIFY2(originalScaledInt == newScaledInt,
                                    qPrintable(
                                        QString("Original scaled position: %1, rewritten scaled position: %2 for file: %3")
                                        .arg(originalScaledInt).arg(newScaledInt).arg(originalFilePath)));
                            }
                        } else {
                            QFAIL(qPrintable(
                                QString("Unsupported type for the position component for file: %1").arg(originalFilePath)));
                        }
                    }, rewrittenComponent);
                };

                auto compareColorComponent = [&](auto originalComponent, auto rewrittenComponent) {
                    // visit the attributes
                    std::visit([&](auto& newComponent) {
                        using new_type = std::decay_t<decltype(newComponent)>;
                        if constexpr (std::is_integral_v<new_type> ||
                                    std::is_floating_point_v<new_type>) {
                            // cast the original type to the rewritten type
                            auto castedComponent
                                = StereoVision::IO::castedPointCloudAttribute<new_type>(originalComponent);
                            // compare the values
                            QCOMPARE(castedComponent, newComponent);
                        } else if constexpr (std::is_same_v<new_type, StereoVision::IO::EmptyParam>) {
                            QVERIFY2(std::holds_alternative<StereoVision::IO::EmptyParam>(originalComponent),
                                qPrintable(QString("Both color components should be empty. Original file: %1")
                                    .arg(originalFilePath)));
                        } else {
                            QFAIL(qPrintable(
                                QString("Unsupported type for the color component for file: %1").arg(originalFilePath)));
                        }
                    }, rewrittenComponent);
                };
                auto xOffset = castedLasRewrittenPointAccess->getXOffset();
                auto yOffset = castedLasRewrittenPointAccess->getYOffset();
                auto zOffset = castedLasRewrittenPointAccess->getZOffset();

                auto xScaleFactor = castedLasRewrittenPointAccess->getXScaleFactor();
                auto yScaleFactor = castedLasRewrittenPointAccess->getYScaleFactor();
                auto zScaleFactor = castedLasRewrittenPointAccess->getZScaleFactor();

                // compare the position and the color
                auto originalPointPosition = originalPointAccess->getPointPosition();
                auto rewrittenPointPosition = rewrittenPointAccess->getPointPosition();
                comparePositionComponent(originalPointPosition.x, rewrittenPointPosition.x, xScaleFactor, xOffset);
                comparePositionComponent(originalPointPosition.y, rewrittenPointPosition.y, yScaleFactor, yOffset);
                comparePositionComponent(originalPointPosition.z, rewrittenPointPosition.z, zScaleFactor, zOffset);

                auto originalPointColorOpt = originalPointAccess->getPointColor();
                auto rewrittenPointColorOpt = rewrittenPointAccess->getPointColor();
                QCOMPARE(originalPointColorOpt.has_value(), rewrittenPointColorOpt.has_value());
                if (originalPointColorOpt.has_value() && rewrittenPointColorOpt.has_value()) {
                    auto originalPointColor = originalPointColorOpt.value();
                    auto rewrittenPointColor = rewrittenPointColorOpt.value();

                    compareColorComponent(originalPointColor.r, rewrittenPointColor.r);
                    compareColorComponent(originalPointColor.g, rewrittenPointColor.g);
                    compareColorComponent(originalPointColor.b, rewrittenPointColor.b);
                    // las format does not contains color alpha...
                    // compareColorComponent(originalPointColor.a, rewrittenPointColor.a);
                }
                
                // Go to the next point
                auto isNextPointOriginal = originalPointAccess->gotoNext();
                auto isNextPointRewritten = rewrittenPointAccess->gotoNext();
                
                if (isNextPointOriginal != isNextPointRewritten) {
                    QVERIFY2(isNextPointOriginal == isNextPointRewritten,
                        qPrintable(QString("The original point cloud : %1 and the rewritten point cloud: \
                            %2 have different number of points").arg(originalFilePath).arg(filePath)));
                }

                if (!isNextPointOriginal || !isNextPointRewritten) {
                    break;
                }
            }
        }
    }
}

void TestPointCloudIO::verifyLasHeaderAttributeValue(std::string attrName,
    StereoVision::IO::PointCloudGenericAttribute attr) {
    bool isTypeCorrect = false;
    
    std::string expectedTypeName;
    
    if (attrName == "fileSignature" || attrName == "systemIdentifier" || attrName == "generatingSoftware") {
        isTypeCorrect = std::holds_alternative<std::string>(attr);
        expectedTypeName = "std::string";
    } else if (attrName == "fileSourceID" || attrName == "globalEncoding" || attrName == "projectID_GUID_Data2" ||
               attrName == "projectID_GUID_Data3" || attrName == "fileCreationDayOfYear" ||
               attrName == "fileCreationYear" || attrName == "headerSize" || attrName == "pointDataRecordLength") {
        isTypeCorrect = std::holds_alternative<uint16_t>(attr);
        expectedTypeName = "uint16_t";
    } else if (attrName == "projectID_GUID_Data1" || attrName == "offsetToPointData" ||
               attrName == "numberOfVariableLengthRecords" || attrName == "legacyNumberOfPointRecords" ||
               attrName == "numberOfExtendedVariableLengthRecords") {
        isTypeCorrect = std::holds_alternative<uint32_t>(attr);
        expectedTypeName = "uint32_t";
    } else if (attrName == "projectID_GUID_Data4") {
        isTypeCorrect = std::holds_alternative<std::vector<uint8_t>>(attr);
        expectedTypeName = "std::vector<uint8_t>";
    } else if (attrName == "versionMajor" || attrName == "versionMinor" || attrName == "pointDataRecordFormat") {
        isTypeCorrect = std::holds_alternative<uint8_t>(attr);
        expectedTypeName = "uint8_t";
    } else if (attrName == "xScaleFactor" || attrName == "yScaleFactor" || attrName == "zScaleFactor" ||
               attrName == "xOffset" || attrName == "yOffset" || attrName == "zOffset" ||
               attrName == "maxX" || attrName == "minX" || attrName == "maxY" || attrName == "minY" ||
               attrName == "maxZ" || attrName == "minZ") {
        isTypeCorrect = std::holds_alternative<double>(attr);
        expectedTypeName = "double";
    } else if (attrName == "startOfWaveformDataPacketRecord" ||
               attrName == "startOfFirstExtendedVariableLengthRecord" || attrName == "numberOfPointRecords") {
        isTypeCorrect = std::holds_alternative<uint64_t>(attr);
        expectedTypeName = "uint64_t";
    } else if (attrName == "legacyNumberOfPointsByReturn") {
        isTypeCorrect = std::holds_alternative<std::vector<uint32_t>>(attr);
        expectedTypeName = "std::vector<uint32_t>";
    } else if (attrName == "numberOfPointsByReturn") {
        isTypeCorrect = std::holds_alternative<std::vector<uint64_t>>(attr);
        expectedTypeName = "std::vector<uint64_t>";
    } else {
        // unexpected attribute
        QFAIL(("Unexpected attribute: " + attrName).c_str());
    }
    
    QVERIFY2(isTypeCorrect,
        ("Attribute \"" + attrName + "\" has incorrect type. Expected: " + expectedTypeName).c_str());
    
    if (attrName == "fileSignature") {
        auto castedValue = StereoVision::IO::castedPointCloudAttribute<std::string>(attr);
        QVERIFY2(castedValue == "LASF", ("Attribute \"" + attrName + "\" has invalid value: " + castedValue).c_str());
    }
}

void TestPointCloudIO::verifyLasPointAttributeValue(std::string attrName, StereoVision::IO::PointCloudGenericAttribute attr) {

    bool isTypeCorrect = false;
    
    std::string expectedTypeName;
    
    if (attrName == "returnNumber" ||  attrName == "returnNumber" ||  attrName == "numberOfReturns" || 
        attrName == "scanDirectionFlag" ||  attrName == "edgeOfFlightLineFlag" ||
        attrName == "classification" ||  attrName == "syntheticFlag" ||  attrName == "keyPointFlag" ||
        attrName == "withheldFlag" || attrName == "overlapFlag" || attrName == "scannerChannel" ||
        attrName == "userData" || attrName == "wavePacketDescriptorIndex") {

        isTypeCorrect = std::holds_alternative<uint8_t>(attr);
        expectedTypeName = "uint8_t";
    } else if (attrName == "intensity" || attrName == "pointSourceID" || attrName == "red" || attrName == "green" ||
               attrName == "blue" || attrName == "NIR") {
        isTypeCorrect = std::holds_alternative<uint16_t>(attr);
        expectedTypeName = "uint16_t";
    } else if (attrName == "waveformPacketSizeInBytes") {
        isTypeCorrect = std::holds_alternative<uint32_t>(attr);
        expectedTypeName = "uint32_t";  
    } else if (attrName == "byteOffsetToWaveformData") {
        isTypeCorrect = std::holds_alternative<uint64_t>(attr);
        expectedTypeName = "uint64_t";  
    } else if (attrName == "scanAngleRank") {
        isTypeCorrect = std::holds_alternative<int8_t>(attr);
        expectedTypeName = "int8_t";
    } else if (attrName == "scanAngle") {
        isTypeCorrect = std::holds_alternative<int16_t>(attr);
        expectedTypeName = "int16_t";
    } else if (attrName == "x" || attrName == "y" || attrName == "z") {
        isTypeCorrect = std::holds_alternative<int32_t>(attr);
        expectedTypeName = "int32_t";
    } else if (attrName == "returnPointWaveformLocation" || attrName == "parametricDx" || attrName == "parametricDy" ||
             attrName == "parametricDz") {
        isTypeCorrect = std::holds_alternative<float>(attr);
        expectedTypeName = "float"; 
    } else if (attrName == "GPSTime") {
        isTypeCorrect = std::holds_alternative<double>(attr);
        expectedTypeName = "double"; 
    } else {
        // unexpected attribute
        QFAIL(("Unexpected attribute: " + attrName).c_str());
    }
    
    QVERIFY2(isTypeCorrect,
        ("Attribute \"" + attrName + "\" has incorrect type. Expected: " + expectedTypeName).c_str());
}

void TestPointCloudIO::testPcdPointCloud() {

    const std::vector<std::string> expectedHeaderAttributes
        = {"version", "fields", "size", "type", "count", "width", "height", "viewpoint", "points", "data"};

    auto originalFiles = lasFiles + pcdFiles + sdcFiles + metacloudFiles;
    QStringList rewrittenFiles;
    std::vector<QTemporaryFile> tempFiles(originalFiles.size()); // temp files that will be auto removed
    for (size_t i = 0; i < originalFiles.size(); i++) {
        // test if the file is open and get its path
        QVERIFY2(tempFiles[i].open(), qPrintable(QString("Failed to open temporary file: %1")
            .arg(tempFiles[i].fileName())));
        QString tempFilePath = tempFiles[i].fileName();
        // save the path
        rewrittenFiles.append(tempFilePath);
        // close the file
        tempFiles[i].close();
        // open the original point cloud and write it to the temporary file
        auto fullPointCloud = StereoVision::IO::openPointCloud(originalFiles[i].toStdString());
        QVERIFY2(fullPointCloud.has_value(),
            qPrintable(QString("Failed to open point cloud: %1").arg(originalFiles[i])));
        
        auto& pointAccess = fullPointCloud->pointAccess;
        auto& headerAccess = fullPointCloud->headerAccess;
        QVERIFY2(pointAccess != nullptr, qPrintable(QString("The point cloud is null for file: %1")
            .arg(originalFiles[i])));
        QVERIFY2(headerAccess != nullptr, qPrintable(QString("The header is null for file: %1").arg(originalFiles[i])));

        // write the point cloud to the temporary file
        QVERIFY2(StereoVision::IO::writePointCloudPcd(tempFilePath.toStdString(), fullPointCloud.value()),
            qPrintable(QString("Failed to write point cloud: %1 to file: %2").arg(originalFiles[i]).arg(tempFilePath)));
    }

    auto allFiles = pcdFiles + rewrittenFiles;
    for (size_t i = 0; i < allFiles.size(); i++) {
        auto filePath = allFiles[i];
        bool isRewrittenFile = i >= pcdFiles.size();
        auto originalFileIndex = i - pcdFiles.size();
        // open the pcd file
        auto fullPointCloud = StereoVision::IO::openPointCloudPcd(filePath.toStdString());
        // verify that the interface is not nullopt and that the header and the point cloud are not nullptr
        QVERIFY2(fullPointCloud != std::nullopt, qPrintable(QString("Failed to open pcd file: %1").arg(filePath)));
        auto& pointAccess = fullPointCloud->pointAccess;
        auto& headerAccess = fullPointCloud->headerAccess;
        QVERIFY2(pointAccess != nullptr, qPrintable(QString("The point cloud is null for file: %1").arg(filePath)));
        QVERIFY2(headerAccess != nullptr, qPrintable(QString("The header is null for file: %1").arg(filePath)));

        for (auto& expectedAttrName: expectedHeaderAttributes) {
            // attribute should exist
            auto attrOpt = headerAccess->getAttributeByName(expectedAttrName.c_str());
            QVERIFY2(attrOpt.has_value(),
                qPrintable(QString("Failed to get the header attribute: %1 for file: %2")
                    .arg(expectedAttrName.c_str()).arg(filePath)));
            auto attr = attrOpt.value();

            verifyPcdHeaderAttributeValue(expectedAttrName, attr);
        }
    }
}

void TestPointCloudIO::verifyPcdHeaderAttributeValue(std::string attrName,
    StereoVision::IO::PointCloudGenericAttribute attr) {
    bool isTypeCorrect = false;
    
    std::string expectedTypeName;
    
    if (attrName == "version") {
        isTypeCorrect = std::holds_alternative<double>(attr);
        expectedTypeName = "double";
    } else if (attrName == "fields") {
        isTypeCorrect = std::holds_alternative<std::vector<std::string>>(attr);
        expectedTypeName = "std::vector<std::string>";
    } else if (attrName == "size" || attrName == "count") {
        isTypeCorrect = std::holds_alternative<std::vector<uint64_t>>(attr);
        expectedTypeName = "std::vector<uint64_t>";
    } else if (attrName == "type") {
        isTypeCorrect = std::holds_alternative<std::vector<uint8_t>>(attr);
        expectedTypeName = "std::vector<uint8_t>";
    } else if (attrName == "width" || attrName == "height" || attrName == "points") {
        isTypeCorrect = std::holds_alternative<uint64_t>(attr);
        expectedTypeName = "uint64_t";
    } else if (attrName == "viewpoint") {
        isTypeCorrect = std::holds_alternative<std::vector<double>>(attr);
        expectedTypeName = "std::vector<double>";
    } else if (attrName == "data") {
        isTypeCorrect = std::holds_alternative<std::string>(attr);
        expectedTypeName = "std::string";  
    } else {
        // unexpected attribute
        QFAIL(("Unexpected attribute: " + attrName).c_str());
    }
    
    QVERIFY2(isTypeCorrect,
        ("Attribute \"" + attrName + "\" has incorrect type. Expected: " + expectedTypeName).c_str());

    // special case: viewpoint size is 7 and data storage is either "ascii", "binary" or "binary_compressed"
    if (attrName == "viewpoint") {
        auto castedAttr = StereoVision::IO::castedPointCloudAttribute<std::vector<double>>(attr);
        QVERIFY2(castedAttr.size() == 7,
            qPrintable(QString("Attribute \"viewpoint\" has incorrect size. Expected: 7. Actual: %1")
                .arg(castedAttr.size())));
    } else if (attrName == "data") {
        auto castedAttr = StereoVision::IO::castedPointCloudAttribute<std::string>(attr);
        QVERIFY2(castedAttr == "ascii" || castedAttr == "binary" || castedAttr == "binary_compressed",
            qPrintable(QString("Attribute \"data\" has incorrect value. Expected: \"ascii\", \"binary\" or \
                \"binary_compressed\". Actual: %1")
                .arg(castedAttr.c_str())));
    }
}

QTEST_MAIN(TestPointCloudIO)
#include "testPointCloudIO.moc"
