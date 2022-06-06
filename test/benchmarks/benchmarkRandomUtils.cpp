#include <QtTest/QtTest>

#include "utils/randomcache.h"

#include <iostream>
#include <random>

class BenchmarkRandomUtils: public QObject{

	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void benchmarkStdRandomInts();
	void benchmarkStdRandomFloats();

	void benchmarkCachedRandomInts();
	void benchmarkCachedRandomFloats();

protected:
	std::default_random_engine re;

};

void BenchmarkRandomUtils::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void BenchmarkRandomUtils::benchmarkStdRandomInts() {

	std::uniform_int_distribution<int> dist(-42,69); //nothing suspicious about those numbers, they are just nice !

	volatile int n;

	QBENCHMARK {
		for (int i = 0; i < 500; i++) {
			n = dist(re);
		}
	}

}
void BenchmarkRandomUtils::benchmarkStdRandomFloats() {

	std::uniform_real_distribution<float> dist(-42,69); //nothing suspicious about those numbers, they are just nice !

	volatile float x;

	QBENCHMARK {
		for (int i = 0; i < 500; i++) {
			x = dist(re);
		}
	}

}

void BenchmarkRandomUtils::benchmarkCachedRandomInts() {

	std::uniform_int_distribution<int> dist(-42,69); //nothing suspicious about those numbers, they are just nice !
	auto lambda = [&dist, this] () {return dist(re); };
	StereoVision::Random::NumbersCache<int> cache(4096, lambda);

	volatile int n;

	QBENCHMARK {
		for (int i = 0; i < 500; i++) {
			n = cache();
		}
	}

}
void BenchmarkRandomUtils::benchmarkCachedRandomFloats() {

	std::uniform_real_distribution<float> dist(-42,69); //nothing suspicious about those numbers, they are just nice !
	auto lambda = [&dist, this] () {return dist(re); };
	StereoVision::Random::NumbersCache<float> cache(4096, lambda);

	volatile float x;

	QBENCHMARK {
		for (int i = 0; i < 500; i++) {
			x = cache();
		}
	}

}


QTEST_MAIN(BenchmarkRandomUtils)
#include "benchmarkRandomUtils.moc"
