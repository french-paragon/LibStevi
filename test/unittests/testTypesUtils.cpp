#include <QtTest/QtTest>

#include "utils/types_manipulations.h"

using namespace StereoVision::TypesManipulations;

class testTypesUtils: public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void testCheckFloat32Precision();

};

void testTypesUtils::testCheckFloat32Precision() {

    if (sizeof (float) <= 4) {
        //float is float32 or smaller
        QVERIFY2(!typeExceedFloat32Precision<float>(), "Float should not exceed float32 precision when sizeof(float) <= 4");
    } else {
        //float is more than float32
        QVERIFY2(typeExceedFloat32Precision<float>(), "Float should exceed float32 precision when sizeof(float) > 4");
    }

    if (sizeof (double) <= 4) {
        //double is float32 or smaller
        QVERIFY2(!typeExceedFloat32Precision<double>(), "Double should not exceed float32 precision when sizeof(double) <= 4");
    } else {
        //double is more than float32
        QVERIFY2(typeExceedFloat32Precision<double>(), "Double should exceed float32 precision when sizeof(double) > 4");
    }

}

QTEST_MAIN(testTypesUtils)
#include "testTypesUtils.moc"
