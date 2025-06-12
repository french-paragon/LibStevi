#include <QtTest/QtTest>

#include "geometry/genericraysalignement.h"

#include <random>

using namespace StereoVision;
using namespace StereoVision::Geometry;

template <typename T>
struct AxisRayAlignementProblem {
    std::vector<RayPairInfos<T>> rays;
    Eigen::Matrix<T, 3, 3> boresight; //ground truth boresight;
};

template <typename T, typename RngT>
AxisRayAlignementProblem<T> generateRandomAxisRayAlignementProblem(int nCorresp, Eigen::Matrix<T, 3, 3> boresight, RngT & rngGenerator) {

    std::uniform_real_distribution<double> uniform(-1,1);

    AxisRayAlignementProblem<T> ret;
    ret.rays = std::vector<RayPairInfos<T>>(nCorresp);
    ret.boresight = boresight;

    std::uniform_real_distribution<T> directions(-1,1);
    std::uniform_real_distribution<T> translation(-10,10);
    std::uniform_real_distribution<T> distance(0,10);

    for (int i = 0; i < nCorresp; i++) {

        Eigen::Matrix<T, 3, 1> direction1(uniform(rngGenerator), uniform(rngGenerator), uniform(rngGenerator));
        direction1.normalize();

        Eigen::Matrix<T, 3, 3> R1_to_2 = rodriguezFormula<double>(0.3*Eigen::Vector3d(uniform(rngGenerator), uniform(rngGenerator), uniform(rngGenerator)));
        double dist = distance(rngGenerator);
        Eigen::Matrix<T, 3, 1> t(translation(rngGenerator), translation(rngGenerator), translation(rngGenerator));

        Eigen::Matrix<T, 3, 1> direction2 = dist*direction1 - t;
        direction2.normalize();

        ret.rays[i] = RayPairInfos<T>{boresight.transpose()*R1_to_2.transpose()*direction1, boresight.transpose()*direction2, R1_to_2, t};

    }

    return ret;

}

class BenchmarkRaysAlignementAlgorithms: public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void initTestCase();

    void benchmarkAxisAlign_data();
    void benchmarkAxisAlign();

private:
    std::default_random_engine re;

};

void BenchmarkRaysAlignementAlgorithms::initTestCase() {
    std::random_device rd;
    re.seed(rd());
}


void BenchmarkRaysAlignementAlgorithms::benchmarkAxisAlign_data() {

    QTest::addColumn<int>("nObs");


    QTest::newRow("underdetermined (3)") << 3;
    QTest::newRow("determined (34)") << 34;
    QTest::newRow("overdetermined (50)") << 50;
    QTest::newRow("overdetermined (500)") << 500;

}
void BenchmarkRaysAlignementAlgorithms::benchmarkAxisAlign() {

    QFETCH(int, nObs);

    std::uniform_real_distribution<double> uniform(-1,1);

    Eigen::Matrix3d boresight = rodriguezFormula<double>(0.3*Eigen::Vector3d(uniform(re), uniform(re), uniform(re)));

    AxisRayAlignementProblem<double> problem = generateRandomAxisRayAlignementProblem<double>(nObs, boresight, re);

    std::optional<Eigen::Matrix3d> result;

    QBENCHMARK { //run only once to avoid the code being optimized out
        result = relaxedAxisAlignRaysSets<double>(problem.rays);
    }

    //sides effects, so the computation is not optimized out

    QVERIFY(result.has_value());

    Eigen::Matrix3d res = result.value().transpose()*problem.boresight - Eigen::Matrix3d::Identity();


    QVERIFY(result.value().array().isFinite().all());

    if (nObs >= 34) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                QVERIFY(std::abs(res(i,j)) < 1e-3);
            }
        }
    }

}

QTEST_MAIN(BenchmarkRaysAlignementAlgorithms)
#include "benchmarkRaysAlignementAlgorithms.moc"
