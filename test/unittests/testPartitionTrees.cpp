#include <QtTest/QtTest>

#include "geometry/genericbinarypartitioningtree.h"

#include <QVector>
#include <vector>

#include <algorithm>

Q_DECLARE_METATYPE(std::vector<float>);

using VecType = QVector<float>;

template<typename VT>
using ContainerType = QVector<VT>;

template<int nDim>
using FixedSizeVecType = std::array<float, nDim>;

template<int nDim, bool fixedSize>
using Vec = std::conditional_t<fixedSize,FixedSizeVecType<nDim>,VecType>;

Q_DECLARE_METATYPE(FixedSizeVecType<1>);
Q_DECLARE_METATYPE(FixedSizeVecType<2>);
Q_DECLARE_METATYPE(FixedSizeVecType<3>);
Q_DECLARE_METATYPE(FixedSizeVecType<4>);
Q_DECLARE_METATYPE(FixedSizeVecType<5>);
Q_DECLARE_METATYPE(FixedSizeVecType<6>);
Q_DECLARE_METATYPE(FixedSizeVecType<7>);
Q_DECLARE_METATYPE(FixedSizeVecType<8>);
Q_DECLARE_METATYPE(FixedSizeVecType<9>);

inline QVector<int> getIndexFromPseudoFlatId(int index, QVector<int> const& shape) {

    int nDim = shape.size();
    QVector<int> out(nDim);

    QVector<int> pseudoStride(nDim);

    int stride = 1;

    for (int i = nDim-1; i >= 0; i--) {
        pseudoStride[i] = stride;
        stride *= shape[i];
    }

    int leftOver = index;

    for (int i = 0; i < nDim; i++) {

        out[i] = leftOver/pseudoStride[i];
        leftOver %= pseudoStride[i];
    }

    return out;
}

template<int nDim, bool fixedSize>
class pointsBuilder {
public:

    inline static VecType buildPoint() {
        return VecType(nDim);
    }

};

template<int nDim>
class pointsBuilder<nDim, true> {
public:

    inline static FixedSizeVecType<nDim> buildPoint() {
        return FixedSizeVecType<nDim>();
    }
};

template <int nDim, bool fixedSize>
ContainerType<Vec<nDim, fixedSize>> buildPoints(QVector<int> const& shape, std::default_random_engine & re) {

        if(nDim != shape.size()) {
            return ContainerType<Vec<nDim, fixedSize>>();
        }

        int n = 1;

        for (int i = 0; i < nDim; i++) {
            n *= shape[i];
        }

        ContainerType<Vec<nDim, fixedSize>> points = ContainerType<Vec<nDim, fixedSize>>(n);

        std::uniform_real_distribution<float> shiftDistribution(-0.5,0.5);

        for (int i = 0; i < n; i++) {
            QVector<int> idx = getIndexFromPseudoFlatId(i, shape);
            Vec<nDim, fixedSize> pt = pointsBuilder<nDim, fixedSize>::buildPoint();

            for (int j = 0; j < nDim; j++) {
                pt[j] = idx[j] + shiftDistribution(re);
            }

            points[i] = pt;
        }

        return points;

    }

template <int nDim, bool fixedSize>
ContainerType<Vec<nDim, fixedSize>> buildPointsWithHoles(QVector<int> const& shape, std::default_random_engine & re) {
        ContainerType<Vec<nDim, fixedSize>> basePoints = buildPoints<nDim, fixedSize>(shape, re);
        if (basePoints.empty()) {
            return basePoints;
        }

        ContainerType<Vec<nDim, fixedSize>> selectedPoints;
        selectedPoints.reserve(basePoints.size());

        double radius = shape[0];

        for (int i = 0; i < shape.size(); i++) {
            if (shape[i] < radius) {
                radius = shape[i];
            }
        }

        radius /= 2;

        double smallerRadius = radius/10;
        double smallerRadiusSq = smallerRadius*smallerRadius;

        std::shuffle(basePoints.begin(), basePoints.end(), re);

        std::array<Vec<nDim, fixedSize>, 5> ptExcl;

        ptExcl[0] = pointsBuilder<nDim, fixedSize>::buildPoint();

        for (int i = 0; i < nDim; i++) {
            ptExcl[0][i] = (shape[i]/2.);
        }

        ptExcl[1] = basePoints[0];
        ptExcl[2] = basePoints[std::min(1,basePoints.size()-1)];
        ptExcl[3] = basePoints[std::min(2,basePoints.size()-1)];
        ptExcl[4] = basePoints[std::min(3,basePoints.size()-1)];

        for (Vec<nDim, fixedSize> const& vev : basePoints) {

            bool keepPoint = true;

            for (int i = 0; i < ptExcl.size(); i++) {
                double radiusSq = 0;

                for (int d = 0; d < nDim; d++) {
                    double tmp = vev[d] - ptExcl[i][d];
                    radiusSq += tmp*tmp;
                }

                if (radiusSq < smallerRadiusSq and i != 0) {
                    keepPoint = false;
                    break;
                }

                if (i == 0 and radiusSq > radius*radius) {
                    keepPoint = false;
                    break;
                }
            }

            if (!keepPoint) {
                continue;
            }

            selectedPoints.push_back(vev);

        }

        return selectedPoints;

    }

class TestPartitionTrees: public QObject
{
    Q_OBJECT
private Q_SLOTS:


    void initTestCase_data();
    void initTestCase();

    void testGenericBSPClosest();
    void testGenericBSPClosestInRange();

private:

    template<int nD, bool fixedSize, bool withHoles>
    void testGenericBSPClosestImpl(QVector<int> const& shape) {
        using VT = std::conditional_t<fixedSize, FixedSizeVecType<nD>, VecType>;
        using BSP = StereoVision::Geometry::GenericBSP<VT, nD, StereoVision::Geometry::BSPObjectWrapper<VT,float>, ContainerType<VT>>;

        int nDim = shape.size();

        if (nDim < nD) {
            QSKIP("Points have lower dimensions than search space!");
        }


        ContainerType<VT> points;

        if (withHoles) {
            points = buildPointsWithHoles<nD, fixedSize>(shape, _re);
        } else {
            points = buildPoints<nD, fixedSize>(shape, _re);
        }

        float max_error = 0;
        size_t n = 1;

        for (int s : shape) {
            n *= s;
            if (s > max_error) {
                max_error = s;
            }
        }

        if (!withHoles) {
            max_error = std::sqrt(nDim)*0.5; //if no hole we know a point has to be closed.
            //if with hole with just keep a basic distance test, mostly to ensure the code is not optimized out.
        }

        BSP tree(points);

        QBENCHMARK {
            for (int i = 0; i < n; i++) {
                QVector<int> idx = getIndexFromPseudoFlatId(i, shape);
                VT pt = pointsBuilder<nD, fixedSize>::buildPoint();

                for (int j = 0; j < nDim; j++) {
                    pt[j] = idx[j];
                }

                VT& closest = tree.closest(pt);
                for (int j = 0; j < nDim; j++) {
                    float error = closest[j] - pt[j];
                    QVERIFY2(std::fabs(error) <= max_error, qPrintable(QString("unexpected closest point found! (index %1, error = %2)").arg(i).arg(error)));
                }
            }
        }
    }

    template<int nD, bool fixedSize, bool withHoles>
    void testGenericBSPClosestInRangeImpl(QVector<int> const& shape) {
        using VT = std::conditional_t<fixedSize, FixedSizeVecType<nD>, VecType>;
        using BSP = StereoVision::Geometry::GenericBSP<VT, nD, StereoVision::Geometry::BSPObjectWrapper<VT,float>, ContainerType<VT>>;

        int nDim = shape.size();

        if (nDim < nD) {
            QSKIP("Points have lower dimensions than search space!");
        }


        ContainerType<VT> points;

        if (withHoles) {
            points = buildPointsWithHoles<nD, fixedSize>(shape, _re);
        } else {
            points = buildPoints<nD, fixedSize>(shape, _re);
        }

        float max_error = 0;
        size_t n = 1;

        for (int s : shape) {
            n *= s;
            if (s > max_error) {
                max_error = s;
            }
        }

        if (!withHoles) {
            max_error = std::sqrt(nDim)*0.5; //if no hole we know a point has to be closed.
            //if with hole with just keep a basic distance test, mostly to ensure the code is not optimized out.
        }

        BSP tree(points);

        QBENCHMARK {
            for (int i = 0; i < n; i++) {
                QVector<int> idx = getIndexFromPseudoFlatId(i, shape);
                VT pt = pointsBuilder<nD, fixedSize>::buildPoint();
                VT min = pointsBuilder<nD, fixedSize>::buildPoint();
                VT max= pointsBuilder<nD, fixedSize>::buildPoint();

                for (int j = 0; j < nDim; j++) {
                    pt[j] = idx[j];
                    min[j] = idx[j]-0.5;
                    max[j] = idx[j]+0.5;
                }

                int closestIdx = tree.closestInRange(pt, min, max);

                if (!withHoles) {
                    QVERIFY(closestIdx >= 0);
                } else {
                    continue;
                }

                VT& closest = tree[closestIdx];

                for (int j = 0; j < nDim; j++) {
                    float error = closest[j] - pt[j];
                    QVERIFY2(std::fabs(error) <= max_error, qPrintable(QString("unexpected closest point found! (index %1, error = %2)").arg(i).arg(error)));
                }
            }
        }
    }

    std::default_random_engine _re;

};

void TestPartitionTrees::initTestCase_data() {

    QTest::addColumn<QVector<int>>("shape");
    QTest::addColumn<bool>("fixedSize");
    QTest::addColumn<bool>("withHoles");


    QTest::newRow("2D space (5x5) (dynamic)") << QVector<int>{5, 5} << false << false;
    QTest::newRow("2D space (25x25) (dynamic)") << QVector<int>{25, 25} << false << false;
    QTest::newRow("2D space (125x125) (dynamic)") << QVector<int>{125, 125} << false << false;
    QTest::newRow("2D space (250x250) (dynamic)") << QVector<int>{250, 250} << false << false;
    QTest::newRow("2D space (375x375) (dynamic)") << QVector<int>{375, 375} << false << false;
    QTest::newRow("2D space (500x500) (dynamic)") << QVector<int>{500, 500} << false << false;
    QTest::newRow("2D space (625x625) (dynamic)") << QVector<int>{625, 625} << false << false;
    QTest::newRow("3D space (3x3x3) (dynamic)") << QVector<int>{3,3,3} << false << false;
    QTest::newRow("3D space (9x9x9) (dynamic)") << QVector<int>{9,9,9} << false << false;
    QTest::newRow("3D space (27x27x27) (dynamic)") << QVector<int>{27,27,27} << false << false;
    QTest::newRow("3D space (50x50x50) (dynamic)") << QVector<int>{50,50,50} << false << false;


    QTest::newRow("2D space (5x5) (fixed)") << QVector<int>{5, 5} << true << false;
    QTest::newRow("2D space (25x25) (fixed)") << QVector<int>{25, 25} << true << false;
    QTest::newRow("2D space (125x125) (fixed)") << QVector<int>{125, 125} << true << false;
    QTest::newRow("2D space (250x250) (fixed)") << QVector<int>{250, 250} << true << false;
    QTest::newRow("2D space (375x375) (fixed)") << QVector<int>{375, 375} << true << false;
    QTest::newRow("2D space (500x500) (fixed)") << QVector<int>{500, 500} << true << false;
    QTest::newRow("2D space (625x625) (fixed)") << QVector<int>{625, 625} << true << false;
    QTest::newRow("3D space (3x3x3) (fixed)") << QVector<int>{3,3,3} << true << false;
    QTest::newRow("3D space (9x9x9) (fixed)") << QVector<int>{9,9,9} << true << false;
    QTest::newRow("3D space (27x27x27) (fixed)") << QVector<int>{27,27,27} << true << false;
    QTest::newRow("3D space (50x50x50) (fixed)") << QVector<int>{50,50,50} << true << false;


    QTest::newRow("2D space (5x5) (fixed, with holes)") << QVector<int>{5, 5} << true << true;
    QTest::newRow("2D space (25x25) (fixed, with holes)") << QVector<int>{25, 25} << true << true;
    QTest::newRow("2D space (125x125) (fixed, with holes)") << QVector<int>{125, 125} << true << true;
    QTest::newRow("2D space (250x250) (fixed, with holes)") << QVector<int>{250, 250} << true << true;
    QTest::newRow("2D space (375x375) (fixed, with holes)") << QVector<int>{375, 375} << true << true;
    QTest::newRow("2D space (500x500) (fixed, with holes)") << QVector<int>{500, 500} << true << true;
    QTest::newRow("2D space (625x625) (fixed, with holes)") << QVector<int>{625, 625} << true << true;
    QTest::newRow("3D space (3x3x3) (fixed, with holes)") << QVector<int>{3,3,3} << true << true;
    QTest::newRow("3D space (9x9x9) (fixed, with holes)") << QVector<int>{9,9,9} << true << true;
    QTest::newRow("3D space (27x27x27) (fixed, with holes)") << QVector<int>{27,27,27} << true << true;
    QTest::newRow("3D space (50x50x50) (fixed, with holes)") << QVector<int>{50,50,50} << true << true;
}

void TestPartitionTrees::initTestCase() {

    std::random_device rd;
    _re.seed(rd());

}

void TestPartitionTrees::testGenericBSPClosest() {
    QFETCH_GLOBAL(QVector<int>, shape);
    QFETCH_GLOBAL(bool, fixedSize);
    QFETCH_GLOBAL(bool, withHoles);


    int nDim = shape.size();

    switch (nDim) {
    case 1:
        if (withHoles) {
            if (fixedSize) {
                testGenericBSPClosestImpl<1, true, true>(shape);
            } else {
                testGenericBSPClosestImpl<1, false, true>(shape);
            }
        } else {
            if (fixedSize) {
                testGenericBSPClosestImpl<1, true, false>(shape);
            } else {
                testGenericBSPClosestImpl<1, false, false>(shape);
            }
        }
        break;
    case 2:
        if (withHoles) {
            if (fixedSize) {
                testGenericBSPClosestImpl<2, true, true>(shape);
            } else {
                testGenericBSPClosestImpl<2, false, true>(shape);
            }
        } else {
            if (fixedSize) {
                testGenericBSPClosestImpl<2, true, false>(shape);
            } else {
                testGenericBSPClosestImpl<2, false, false>(shape);
            }
        }
        break;
    case 3:
        if (withHoles) {
            if (fixedSize) {
                testGenericBSPClosestImpl<3, true, true>(shape);
            } else {
                testGenericBSPClosestImpl<3, false, true>(shape);
            }
        } else {
            if (fixedSize) {
                testGenericBSPClosestImpl<3, true, false>(shape);
            } else {
                testGenericBSPClosestImpl<3, false, false>(shape);
            }
        }
        break;
    case 4:
        if (withHoles) {
            if (fixedSize) {
                testGenericBSPClosestImpl<4, true, true>(shape);
            } else {
                testGenericBSPClosestImpl<4, false, true>(shape);
            }
        } else {
            if (fixedSize) {
                testGenericBSPClosestImpl<4, true, false>(shape);
            } else {
                testGenericBSPClosestImpl<4, false, false>(shape);
            }
        }
        break;
    default:
        QSKIP("Unsupported number of dimensions requested!");
    }

}

void TestPartitionTrees::testGenericBSPClosestInRange() {
    QFETCH_GLOBAL(QVector<int>, shape);
    QFETCH_GLOBAL(bool, fixedSize);
    QFETCH_GLOBAL(bool, withHoles);

    int nDim = shape.size();

    switch (nDim) {
    case 1:
        if (withHoles) {
            if (fixedSize) {
                testGenericBSPClosestInRangeImpl<1, true, true>(shape);
            } else {
                testGenericBSPClosestInRangeImpl<1, false, true>(shape);
            }
        } else {
            if (fixedSize) {
                testGenericBSPClosestInRangeImpl<1, true, false>(shape);
            } else {
                testGenericBSPClosestInRangeImpl<1, false, false>(shape);
            }
        }
        break;
    case 2:
        if (withHoles) {
            if (fixedSize) {
                testGenericBSPClosestInRangeImpl<2, true, true>(shape);
            } else {
                testGenericBSPClosestInRangeImpl<2, false, true>(shape);
            }
        } else {
            if (fixedSize) {
                testGenericBSPClosestInRangeImpl<2, true, false>(shape);
            } else {
                testGenericBSPClosestInRangeImpl<2, false, false>(shape);
            }
        }
        break;
    case 3:
        if (withHoles) {
            if (fixedSize) {
                testGenericBSPClosestInRangeImpl<3, true, true>(shape);
            } else {
                testGenericBSPClosestInRangeImpl<3, false, true>(shape);
            }
        } else {
            if (fixedSize) {
                testGenericBSPClosestInRangeImpl<3, true, false>(shape);
            } else {
                testGenericBSPClosestInRangeImpl<3, false, false>(shape);
            }
        }
        break;
    case 4:
        if (withHoles) {
            if (fixedSize) {
                testGenericBSPClosestInRangeImpl<4, true, true>(shape);
            } else {
                testGenericBSPClosestInRangeImpl<4, false, true>(shape);
            }
        } else {
            if (fixedSize) {
                testGenericBSPClosestInRangeImpl<4, true, false>(shape);
            } else {
                testGenericBSPClosestInRangeImpl<4, false, false>(shape);
            }
        }
        break;
    default:
        QSKIP("Unsupported number of dimensions requested!");
    }
}

QTEST_MAIN(TestPartitionTrees);
#include "testPartitionTrees.moc"
