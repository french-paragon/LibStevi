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
#include "io/bit_manipulations.h"

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

void TestPointCloudIO::testCastedPointCloudAttribute_data()
{
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

void TestPointCloudIO::testCastedPointCloudAttribute()
{
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

QTEST_MAIN(TestPointCloudIO)
#include "testPointCloudIO.moc"
