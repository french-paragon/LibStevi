#include <QtTest/QtTest>
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2026  Paragon<french.paragon@gmail.com>

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

#include "utils/combinatorial.h"

#include <set>
#include <vector>

class TestCombinatorial : public QObject
{
	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testNChooseK();
	void testNChooseKIndexer_data();
	void testNChooseKIndexer();
};

void TestCombinatorial::initTestCase() {

}

void TestCombinatorial::testNChooseK() {

	QCOMPARE(StereoVision::Combinatorial::ChooseInSetIndexer::nChooseK(3,2),3);
	QCOMPARE(StereoVision::Combinatorial::ChooseInSetIndexer::nChooseK(5,3),10);
	QCOMPARE(StereoVision::Combinatorial::ChooseInSetIndexer::nChooseK(11,6),462);
	QCOMPARE(StereoVision::Combinatorial::ChooseInSetIndexer::nChooseK(7,3),35);
	QCOMPARE(StereoVision::Combinatorial::ChooseInSetIndexer::nChooseK(13,7),1716);

}

void TestCombinatorial::testNChooseKIndexer_data() {

	QTest::addColumn<int>("n");
	QTest::addColumn<int>("k");

	QTest::newRow("3 - 2") << 3 << 2;
	QTest::newRow("5 - 3") << 5 << 3;
	QTest::newRow("7 - 3") << 7 << 3;
	QTest::newRow("11 - 6") << 11 << 6;
	QTest::newRow("13 - 5") << 13 << 5;
	QTest::newRow("13 - 7") << 13 << 7;
}

void TestCombinatorial::testNChooseKIndexer() {

	QFETCH(int, n);
	QFETCH(int, k);

	StereoVision::Combinatorial::ChooseInSetIndexer indexer(n,k);

	int nChoices = indexer.nChoices();
	QCOMPARE(nChoices, StereoVision::Combinatorial::ChooseInSetIndexer::nChooseK(n,k));

	std::set<std::vector<int>> founds;

	for (int i = 0; i < nChoices; i++) {
		std::vector<int> choice = indexer.idx2set(i);
		QCOMPARE(choice.size(),k);
		for (int i = 0; i < choice.size(); i++) {
			QVERIFY(choice[i] >= 0);
			QVERIFY(choice[i] < n);

			if (i+1 < choice.size()) {
				QVERIFY(choice[i] < choice[i+1]);
			}
		}
		QVERIFY(!founds.count(choice));
		founds.insert(choice);
		QVERIFY(founds.count(choice));
		int idx = indexer.set2idx(choice);
		QCOMPARE(idx, i);
	}


}

QTEST_MAIN(TestCombinatorial)
#include "testCombinatorial.moc"
