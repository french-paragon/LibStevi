#include <QtTest/QtTest>

#include "optimization/assignement_problems.h"

#include <iostream>
#include <random>

using namespace StereoVision::Optimization;

using CostT = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using PairsT = QVector<std::array<int,2>>;

Q_DECLARE_METATYPE(CostT)
Q_DECLARE_METATYPE(PairsT)

class TestAssignementProblem: public QObject{

    Q_OBJECT
private Q_SLOTS:

    void initTestCase();

    void testOptimalAssignement_data();
    void testOptimalAssignement();

    void testCostExtensionWithNonMatched();

private:
    std::default_random_engine re;
};


void TestAssignementProblem::initTestCase() {
    //srand((unsigned int) time(nullptr));
    std::random_device rd;
    re.seed(rd());
}

void TestAssignementProblem::testOptimalAssignement_data() {

    QTest::addColumn<CostT>("cost");
    QTest::addColumn<PairsT>("gt_matches");

    CostT problem1cost;
    problem1cost.resize(2,2);

    problem1cost << 2, 1,
                    1, 2;

    QVector<std::array<int,2>> problem1matches = {{0,1},{1,0}};

    QTest::newRow("Very basic") << problem1cost << problem1matches;

    CostT problem2cost;
    problem2cost.resize(2,2);

    problem2cost << 1, 2,
                    3, 5;

    QVector<std::array<int,2>> problem2matches = {{0,1},{1,0}};

    QTest::newRow("Less basic") << problem2cost << problem2matches;

    CostT problem3cost;
    problem3cost.resize(2,3);

    problem3cost << 1, 2, 7,
                    3, 5, 8;

    QVector<std::array<int,2>> problem3matches = {{0,1},{1,0}};

    QTest::newRow("Non-square basic") << problem3cost << problem3matches;

    CostT problem4cost;
    problem4cost.resize(3,2);

    problem4cost << 1, 2,
                    3, 5,
                    7, 8;

    QVector<std::array<int,2>> problem4matches = {{0,1},{1,0}};

    QTest::newRow("Non-square basic vertical") << problem4cost << problem4matches;


    CostT problem5cost;
    problem5cost.resize(4,4);

    problem5cost << 5, 4, 1, 6,
                    3, 5, 2, 6,
                    6, 6, 3, 4,
                    4, 3, 2, 5;

    QVector<std::array<int,2>> problem5matches = {{0,2},{1,0},{2,3},{3,1}};

    QTest::newRow("Larger") << problem5cost << problem5matches;



}
void TestAssignementProblem::testOptimalAssignement() {

    QFETCH(CostT, cost);
    QFETCH(PairsT, gt_matches);

    std::vector<std::array<int,2>> matches = optimalAssignement(cost);

    QSet<int> found;

    QCOMPARE(matches.size(), gt_matches.size());

    for (std::array<int,2> const& pair : matches) {

        int idx = gt_matches.indexOf(pair);
        QVERIFY2(idx >= 0, "could not find match in ground truth");
        QVERIFY2(!found.contains(idx), "duplicate pair returned");

        found.insert(idx);

    }

}



void TestAssignementProblem::testCostExtensionWithNonMatched() {

    CostT cost;

    cost.resize(6,4);

    std::uniform_int_distribution<int> dist(-27,33);

    std::vector<int> idxs(cost.cols());

    for (int i = 0; i < cost.rows(); i++) {
        for (int j = 0; j < cost.cols(); j++) {
            cost(i,j) = dist(re);
        }

        idxs[i] = i;
    }

    int nBest = 3;
    int costDist = 4;
    int best = -69;
    int worst = 42;

    for (int i = 0; i < cost.rows(); i++) {

        std::shuffle(idxs.begin(), idxs.end(), re);

        for (int j = 0; j < nBest; j++) {
            cost(i,j) = best+j;
        }
        cost(i,cost.cols()-1) = worst;

        for (int j = 0; j < cost.cols(); j++) {
            int tmp = cost(i,j);
            cost(i,j) = cost(i,idxs[j]);
            cost(i,idxs[j]) = tmp;
        }
    }

    CostT costExtendedNBest = extendCostForNBestCosts(cost,nBest);
    CostT costExtendedScale = extendCostForDistFromBestCost(cost,costDist);

    QCOMPARE(costExtendedNBest.rows(), cost.rows());
    QCOMPARE(costExtendedScale.rows(), cost.rows());

    QCOMPARE(costExtendedNBest.cols(), cost.cols() + cost.rows());
    QCOMPARE(costExtendedScale.cols(), cost.cols() + cost.rows());

    for (int i = 0; i < cost.rows(); i++) {

        for (int j = 0; j < cost.cols(); j++) {
            if (j == i) {
                QVERIFY(costExtendedNBest(i,cost.cols()+j) <= best+2);
                QVERIFY(costExtendedScale(i,cost.cols()+j) <= best+costDist);

            } else {
                QVERIFY(costExtendedNBest(i,cost.cols()+j) >= worst);
                QVERIFY(costExtendedScale(i,cost.cols()+j) >= worst);
            }
        }

    }

}

QTEST_MAIN(TestAssignementProblem)
#include "testAssignementProblems.moc"
