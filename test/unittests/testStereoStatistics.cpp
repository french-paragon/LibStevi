#include <QtTest/QtTest>

#include "statistics/stereo_covering.h"

class TestStereoStatistics: public QObject
{
	Q_OBJECT

private Q_SLOTS:
	void testStereoCoveringStat();

};

void TestStereoStatistics::testStereoCoveringStat() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	constexpr int imgSize = 100;
	constexpr int squareSize = 20;

	Multidim::Array<int, 2> dispL(imgSize, imgSize);
	Multidim::Array<int, 2> dispR(imgSize, imgSize);

	for (int i = 0; i < imgSize; i++) {
		for (int j = 0; j < imgSize; j++) {

			dispL.at<Nc>(i,j) = squareSize *
								(i-imgSize/2 < squareSize/2 and i-imgSize/2 >= -squareSize/2)*
								(j-imgSize/2 < squareSize/2 and j-imgSize/2 >= -squareSize/2);

			dispR.at<Nc>(i,j) = squareSize *
								(i+squareSize-imgSize/2 < squareSize/2 and i+squareSize-imgSize/2 >= -squareSize/2)*
								(j-imgSize/2 < squareSize/2 and j-imgSize/2 >= -squareSize/2);
		}
	}

	float expProp = 1 - float(squareSize*squareSize)/float(imgSize*imgSize);
	float found = StereoVision::Statistics::computeCoveringProportion<StereoVision::Correlation::dispDirection::RightToLeft>(dispL, dispR);

	QCOMPARE(found, expProp);
}

QTEST_MAIN(TestStereoStatistics)
#include "testStereoStatistics.moc"
