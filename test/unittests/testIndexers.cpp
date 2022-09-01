#include <QtTest/QtTest>

#include "utils/indexers.h"

#include <random>

class TestIndexers: public QObject
{
	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void testIndexPairMap();
	void testFixedSizeDisjointSetForest();

private:

	std::default_random_engine re;

};


void TestIndexers::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestIndexers::testIndexPairMap() {

	int nElements = 100;
	int mapInitialSize = 1000;
	int value = 1;

	QVector<QPair<int, int>> pairs(nElements);

	auto dist1 = std::uniform_int_distribution(0, nElements-1);
	auto dist2 = std::uniform_int_distribution(0, nElements-2);

	for (int i = 0; i < nElements; i++) {
		int random1 = dist1(re);
		int random2 = dist2(re);

		if (random2 >= random1) {
			random2 += 1;
		}

		pairs[i] = {random1, random2};
	}

	StereoVision::Indexers::IndexPairMap map(mapInitialSize);

	for (auto pair : pairs) {
		QVERIFY2(!map.hasElement(pair.first, pair.second), "Map not starting empty!");
		QVERIFY2(!map.getElement(pair.first, pair.second).has_value(), "Map not starting empty!");
		QVERIFY2(map.getElementOrDefault(pair.first, pair.second, 0) == 0, "Map not starting empty!");
	}

	for (auto pair : pairs) {
		map.setElement(pair.first, pair.second, value);
	}

	for (auto pair : pairs) {
		QVERIFY2(map.hasElement(pair.first, pair.second), "Missing inserted element");
		QVERIFY2(map.getElement(pair.first, pair.second).has_value(), "Map not starting empty!");
		QVERIFY2(map.getElementOrDefault(pair.first, pair.second, 0) == value, "Map not starting empty!");
	}
}

void TestIndexers::testFixedSizeDisjointSetForest() {

	int nElements = 100;
	int nGroups = 3;

	QVector<int> groupsIds(nGroups);

	for(int i = 0; i < nGroups; i++) {
		int randIndex = std::uniform_int_distribution(i, nElements)(re);

		for (int j = 0; j < i; j++) {
			if (groupsIds[j] == randIndex) {
				groupsIds[j] = j;
				break;
			}
		}

		groupsIds[i] = randIndex;
	}

	QString groups = "[";
	for (int i = 0; i < groupsIds.size(); ++i)
	{
		if (i > 0)
			groups += " ";
		groups += QString::number(groupsIds[i]);
	}
	groups += "]";

	auto dist = std::uniform_int_distribution(0, nElements-2);

	StereoVision::Indexers::FixedSizeDisjointSetForest f(nElements);

	for (int i = 0; i < nElements; i++) {
		if (groupsIds.contains(i)) {
			continue;
		}

		int other = dist(re);

		if (other >= i) {
			other += 1;
		}

		f.joinNode(i, other);
	}

	auto gdist = std::uniform_int_distribution(0, nGroups-1);

	for (int i = 0; i < nElements; i++) {
		if (groupsIds.contains(i)) {
			continue;
		}

		if (f.getGroup(i) == i) {
			int rg = gdist(re);

			f.joinNode(i, groupsIds[rg]);
		}

	}


	for (int i = 0; i < nElements; i++) {
		QVERIFY2(groupsIds.contains(f.getGroup(i)),
				 qPrintable(QString("Element %1 is not a part of the expected groups (is %2, expected: %3)").arg(i).arg(f.getGroup(i)).arg(groups)));
	}
}

QTEST_MAIN(TestIndexers);

#include "testIndexers.moc"
