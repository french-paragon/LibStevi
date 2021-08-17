#include <QtTest/QtTest>

#include "correlation/cost_based_refinement.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

class TestCostRefinement: public QObject
{
	Q_OBJECT
private Q_SLOTS:

	void initTestCase();

	void test1dCostParabola();
	void test2dCostIsotropicParabola();
	void test2dCostAnisotropicParabola();

private:

	std::default_random_engine re;

};


void TestCostRefinement::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestCostRefinement::test1dCostParabola() {

	std::uniform_real_distribution<float> uniformDistA(-1, 1);
	float a = uniformDistA(re);

	std::uniform_real_distribution<float> uniformDistB(-2*std::fabs(a), 2*std::fabs(a));
	float b = uniformDistB(re);

	float expected_delta = -b/(2*a);

	Multidim::Array<StereoVision::Correlation::disp_t,2> rawDisp(1,1);
	Multidim::Array<float,3> truncated_cost_volume(1,1,3);

	rawDisp.at(0,0) = 0;

	truncated_cost_volume.at(0,0,0) = a - b;
	truncated_cost_volume.at(0,0,1) = 0;
	truncated_cost_volume.at(0,0,2) = a + b;

	Multidim::Array<float,2> ref = StereoVision::Correlation::refineDispCostInterpolation<StereoVision::Correlation::InterpolationKernel::Parabola>
			(truncated_cost_volume, rawDisp);

	QCOMPARE(ref.value(0,0), expected_delta);

}

void TestCostRefinement::test2dCostIsotropicParabola() {

	std::uniform_real_distribution<float> uniformDistS(-1, 1);
	std::uniform_real_distribution<float> uniformDistD(-1, 1);

	float s = uniformDistS(re);

	if (std::abs(s) < 1e-4) {
		s = 0.5;
	}

	float expectedx = uniformDistD(re);
	float expectedy = uniformDistD(re);

	Multidim::Array<StereoVision::Correlation::disp_t,3> rawDisp(1,1,2);
	Multidim::Array<float,4> truncated_cost_volume(1,1,3,3);

	rawDisp.at(0,0,0) = 0;
	rawDisp.at(0,0,1) = 0;

	auto cost = [&s, &expectedx, &expectedy] (int dx, int dy) {
		float c_x = dx - expectedx;
		float c_y = dy - expectedy;
		return s*(c_x*c_x + c_y*c_y);
	};

	truncated_cost_volume.at(0,0,0,0) = cost(-1,-1);
	truncated_cost_volume.at(0,0,1,0) = cost(0,-1);
	truncated_cost_volume.at(0,0,2,0) = cost(1,-1);

	truncated_cost_volume.at(0,0,0,1) = cost(-1,0);
	truncated_cost_volume.at(0,0,1,1) = cost(0,0);
	truncated_cost_volume.at(0,0,2,1) = cost(1,0);

	truncated_cost_volume.at(0,0,0,2) = cost(-1,1);
	truncated_cost_volume.at(0,0,1,2) = cost(0,1);
	truncated_cost_volume.at(0,0,2,2) = cost(1,1);

	Multidim::Array<float,3> ref = StereoVision::Correlation::refineDisp2dCostInterpolation<StereoVision::Correlation::InterpolationKernel::Parabola,
																							StereoVision::Correlation::IsotropyHypothesis::Isotropic>
			(truncated_cost_volume, rawDisp);

	float refinedx = ref.value(0,0,0);
	float refinedy = ref.value(0,0,1);

	QCOMPARE(refinedx, expectedx);
	QCOMPARE(refinedy, expectedy);

}

void TestCostRefinement::test2dCostAnisotropicParabola() {

	std::uniform_real_distribution<float> uniformDistS(-1, 1);
	std::uniform_real_distribution<float> uniformDistD(0.7, 1);

	float s = uniformDistS(re);

	if (std::abs(s) < 0.5) {
		s = std::copysign(0.5,s);
	}

	float d1 = s*uniformDistD(re);
	float d2 = s*uniformDistD(re);
	float alpha = 0.1*uniformDistS(re);

	if (std::signbit(d1) != std::signbit(d2)) {
		d2 = -d2;
	}

	Eigen::Rotation2D<float> rot(alpha);

	Eigen::Matrix2f A = rot.toRotationMatrix()*Eigen::DiagonalMatrix<float,2>(d1,d2)*rot.toRotationMatrix().transpose();

	std::uniform_real_distribution<float> uniformDistB(-0.5,0.5);
	Eigen::Vector2f b(uniformDistB(re), uniformDistB(re));

	Eigen::Vector2f expected_deltas = b;

	Multidim::Array<StereoVision::Correlation::disp_t,3> rawDisp(1,1,2);
	Multidim::Array<float,4> truncated_cost_volume(1,1,3,3);

	rawDisp.at(0,0,0) = 0;
	rawDisp.at(0,0,1) = 0;

	auto cost = [&A, &b] (int dx, int dy) {
		Eigen::Vector2f p(dx-b(0),dy-(b(1)));
		return p.dot(A*p);
	};

	truncated_cost_volume.at(0,0,0,0) = cost(-1,-1);
	truncated_cost_volume.at(0,0,1,0) = cost(0,-1);
	truncated_cost_volume.at(0,0,2,0) = cost(1,-1);

	truncated_cost_volume.at(0,0,0,1) = cost(-1,0);
	truncated_cost_volume.at(0,0,1,1) = cost(0,0);
	truncated_cost_volume.at(0,0,2,1) = cost(1,0);

	truncated_cost_volume.at(0,0,0,2) = cost(-1,1);
	truncated_cost_volume.at(0,0,1,2) = cost(0,1);
	truncated_cost_volume.at(0,0,2,2) = cost(1,1);

	Multidim::Array<float,3> ref = StereoVision::Correlation::refineDisp2dCostInterpolation<StereoVision::Correlation::InterpolationKernel::Parabola,
																							StereoVision::Correlation::IsotropyHypothesis::Anisotropic>
			(truncated_cost_volume, rawDisp);

	float refinedx = ref.value(0,0,0);
	float refinedy = ref.value(0,0,1);

	float expectedx = expected_deltas(0);
	float expectedy = expected_deltas(1);

	QCOMPARE(refinedx, expectedx);
	QCOMPARE(refinedy, expectedy);

}

QTEST_MAIN(TestCostRefinement);

#include "testCostRefinement.moc"
