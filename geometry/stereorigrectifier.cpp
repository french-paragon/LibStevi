#include "stereorigrectifier.h"

#include "optimization/l2optimization.h"
#include "lensdistortion.h"

namespace StereoVision {
namespace Geometry {

StereoRigRectifier::StereoRigRectifier(ShapePreservingTransform cam2tocam1,
									   float fLenCam1Px,
									   Eigen::Vector2f ppCam1,
									   Eigen::Vector2i sizeCam1,
									   std::optional<Eigen::Vector3f> kCam1,
									   std::optional<Eigen::Vector2f> tCam1,
									   std::optional<Eigen::Vector2f> BCam1,
									   float fLenCam2Px,
									   Eigen::Vector2f ppCam2,
									   Eigen::Vector2i sizeCam2,
									   std::optional<Eigen::Vector3f> kCam2,
									   std::optional<Eigen::Vector2f> tCam2,
									   std::optional<Eigen::Vector2f> BCam2):
	_cam2Tocam1(cam2tocam1),
	_fLenCam1Px(fLenCam1Px),
	_ppCam1(ppCam1),
	_sizeCam1(sizeCam1),
	_kCam1(kCam1),
	_tCam1(tCam1),
	_BCam1(BCam1),
	_fLenCam2Px(fLenCam2Px),
	_ppCam2(ppCam2),
	_sizeCam2(sizeCam2),
	_kCam2(kCam2),
	_tCam2(tCam2),
	_BCam2(BCam2)
{
	_cam2Tocam1.s = 1;
}

void StereoRigRectifier::clear() {

	_CorrRComputed = false;
	_forwardMapsComputed = false;
	_ROIComputed = false;
	_refocalLengthComputed = false;
	_backwardMapsComputed = false;

}
bool StereoRigRectifier::compute(TargetRangeSetMethod roiSetMethod, TargetRangeSetMethod resolutionSetMethod) {

	clear();

	bool ok = true;

	ok = ok and computeOptimalCamsRots();

	if (!ok) {
		return false;
	}

	ok = ok and computeForwardMaps();

	if (!ok) {
		return false;
	}

	ok = ok and computeROIs(roiSetMethod);

	if (!ok) {
		return false;
	}

	ok = ok and computeResolution(resolutionSetMethod);

	if (!ok) {
		return false;
	}

	ok = ok and computeBackwardMaps();

	return ok;

}

bool StereoRigRectifier::computeOptimalCamsRots() {

	Eigen::Vector3f tDir = _cam2Tocam1.t;

	if (tDir.norm() < 1e-4) {
		return false;
	}

	tDir.normalize();

	Eigen::Matrix3f RC2 = rodriguezFormula(_cam2Tocam1.r);

	Eigen::Vector3f forwardCam1(0,0,1);
	Eigen::Vector3f forwardCam2 = RC2*forwardCam1;

	//find the closest vector to forwardCam1 that is perpendicular to tDir
	Eigen::Vector3f crossC1 = tDir.cross(forwardCam1);
	Eigen::Vector3f dirC1 = crossC1.cross(tDir);

	float normC1 = dirC1.norm();

	if (normC1 < 1e-4) { //if the cameras are misaligned no need to continue
		return false;
	}

	dirC1 /= normC1;

	//find the closest vector to forwardCam2 that is perpendicular to tDir
	Eigen::Vector3f crossC2 = tDir.cross(forwardCam2);
	Eigen::Vector3f dirC2 = crossC2.cross(tDir);

	float normC2 = dirC2.norm();

	if (normC2 < 1e-4) { //if the cameras are misaligned no need to continue
		return false;
	}

	dirC2 /= normC2;

	if (dirC1.dot(dirC2) < 0.2) { //misaligned cameras
		return false;
	}

	//compute the target direction.
	Eigen::Vector3f dirMean = (dirC1 + dirC2)/2.;
	dirMean.normalize();

	Eigen::Vector3f rCam1 = forwardCam1.cross(dirMean);

	float normRC1 = rCam1.norm();
	if (normRC1 > 1e-3) { //large angle
		rCam1 *= std::asin(normRC1)/normRC1;
	}

	Eigen::Vector3f rCam2 = forwardCam2.cross(dirMean);

	float normRC2 = rCam2.norm();
	if (normRC2 > 1e-3) { //large angle
		rCam2 *= std::asin(normRC2)/normRC2;
	}

	Eigen::Matrix3f RotC1 = rodriguezFormula(rCam1);
	Eigen::Matrix3f RotC2 = rodriguezFormula(rCam2);

	Eigen::Vector3f xAxisC1(1,0,0);
	Eigen::Vector3f xAxisC2= RC2*xAxisC1;

	Eigen::Vector3f cxAxisC1 = RotC1*xAxisC1;
	Eigen::Vector3f cxAxisC2 = RotC2*xAxisC2;

	Eigen::Vector3f aCam1 = cxAxisC1.cross(tDir);

	float normAC1 = aCam1.norm();
	if (normAC1 > 1e-3) { //large angle
		aCam1 *= std::asin(normAC1)/normAC1;
	}

	Eigen::Vector3f aCam2 = cxAxisC2.cross(tDir);

	float normAC2 = aCam2.norm();
	if (normAC2 > 1e-3) { //large angle
		aCam2 *= std::asin(normAC2)/normAC2;
	}

	_CorrRComputed = true;
	_CorrRCam1 = rodriguezFormula(aCam1)*RotC1;
	Eigen::Matrix3f CorrRCam2InCam1Frame = rodriguezFormula(aCam2)*RotC2;
	_CorrRCam2 = RC2.transpose()*CorrRCam2InCam1Frame*RC2;

	return true;
}

Eigen::Vector2f StereoRigRectifier::computeForwardVec(Eigen::Vector2f const& vec,
								  Eigen::Vector2f const& pp,
								  float f,
								  Eigen::Matrix3f const& R) {
	Eigen::Vector3f v;
	v.block<2,1>(0,0) = (vec - pp)/f;
	v[2] = 1;

	v = R.transpose()*v;
	v /= v[2];

	return v.block<2,1>(0,0);
}

Eigen::Vector2f StereoRigRectifier::computeBackwardVec(Eigen::Vector2f const& vec,
													   Eigen::Vector2f const& pp,
													   float f,
													   Eigen::Vector2f const& pp_back,
													   float f_back,
													   Eigen::Matrix3f const& R,
													   std::optional<Eigen::Vector3f> k,
													   std::optional<Eigen::Vector2f> t,
													   std::optional<Eigen::Vector2f> B) {

	Eigen::Vector3f v;
	v.block<2,1>(0,0) = (vec - pp_back)/f_back;
	v[2] = 1;

	v = R*v;
	v /= v[2];

	Eigen::Vector2f r = v.block<2,1>(0,0);

	Eigen::Vector2f dRadial = Eigen::Vector2f::Zero();
	Eigen::Vector2f dTangential = Eigen::Vector2f::Zero();

	if (k.has_value()) {
		dRadial = radialDistortion(r, k.value());
	}

	if (t.has_value()) {
		dTangential = tangentialDistortion(r, t.value());
	}

	r += dRadial + dTangential;

	if (B.has_value()) {
		return skewDistortion(r, B.value(), f, pp);
	}

	return f*r + pp;

}

bool StereoRigRectifier::computeForwardMaps() {

	if (!_CorrRComputed) {
		return false;
	}


	_cordTopLeftC1 = computeForwardVec(Eigen::Vector2f(0,0), _ppCam1, _fLenCam1Px, _CorrRCam1);
	_cordTopRightC1 = computeForwardVec(Eigen::Vector2f(_sizeCam1[0],0), _ppCam1, _fLenCam1Px, _CorrRCam1);
	_cordBottomLeftC1 = computeForwardVec(Eigen::Vector2f(0,_sizeCam1[1]), _ppCam1, _fLenCam1Px, _CorrRCam1);
	_cordBottomRightC1 = computeForwardVec(Eigen::Vector2f(_sizeCam1[0],_sizeCam1[1]), _ppCam1, _fLenCam1Px, _CorrRCam1);


	_cordTopLeftC2 = computeForwardVec(Eigen::Vector2f(0,0), _ppCam2, _fLenCam2Px, _CorrRCam2);
	_cordTopRightC2 = computeForwardVec(Eigen::Vector2f(_sizeCam2[0],0), _ppCam2, _fLenCam2Px, _CorrRCam2);
	_cordBottomLeftC2 = computeForwardVec(Eigen::Vector2f(0,_sizeCam2[1]), _ppCam2, _fLenCam2Px, _CorrRCam2);
	_cordBottomRightC2 = computeForwardVec(Eigen::Vector2f(_sizeCam2[0],_sizeCam2[1]), _ppCam2, _fLenCam2Px, _CorrRCam2);


	if (!_cordTopLeftC1.array().isFinite().all()) {
		_forwardMapsComputed = false;
		return false;
	}
	if (!_cordTopRightC1.array().isFinite().all()) {
		_forwardMapsComputed = false;
		return false;
	}
	if (!_cordBottomLeftC1.array().isFinite().all()) {
		_forwardMapsComputed = false;
		return false;
	}
	if (!_cordBottomRightC1.array().isFinite().all()) {
		_forwardMapsComputed = false;
		return false;
	}

	if (!_cordTopLeftC2.array().isFinite().all()) {
		_forwardMapsComputed = false;
		return false;
	}
	if (!_cordTopRightC2.array().isFinite().all()) {
		_forwardMapsComputed = false;
		return false;
	}
	if (!_cordBottomLeftC2.array().isFinite().all()) {
		_forwardMapsComputed = false;
		return false;
	}
	if (!_cordBottomRightC2.array().isFinite().all()) {
		_forwardMapsComputed = false;
		return false;
	}


	_forwardMapsComputed = true;
	return true;

}

bool StereoRigRectifier::computeROIs(TargetRangeSetMethod roiSetMethod) {

	if (!_forwardMapsComputed) {
		return false;
	}

	if (roiSetMethod == Minimal) { //find a ROI which is the biggest that reqiures not interpolation of filling in values.

		//top lefts points
		_ROIC1TopLeft.y() = std::max(_cordTopLeftC1.y(), std::max(_cordTopRightC1.y(), std::max(_cordTopLeftC2.y(), _cordTopRightC2.y())));
		_ROIC2TopLeft.y() = _ROIC1TopLeft.y();

		_ROIC1TopLeft.x() = std::max(_cordTopLeftC1.x(), _cordBottomLeftC1.x());
		_ROIC2TopLeft.x() = std::max(_cordTopLeftC2.x(), _cordBottomLeftC2.x());

		//bottom right points
		_ROIC1BottomRight.y() = std::min(_cordBottomLeftC1.y(), std::min(_cordBottomRightC1.y(), std::min(_cordBottomLeftC2.y(), _cordBottomRightC2.y())));
		_ROIC2BottomRight.y() = _ROIC1BottomRight.y();

		_ROIC1BottomRight.x() = std::min(_cordTopRightC1.x(), _cordBottomRightC1.x());
		_ROIC2BottomRight.x() = std::min(_cordTopRightC2.x(), _cordBottomRightC2.x());

	} else if (roiSetMethod == Same) {


		Eigen::Matrix<float, 16,1> obs;
		Eigen::Matrix<float, 16,4> A; //four parameters: scale, translation x cam1, translation x cam2, translation y.
		Eigen::Matrix<float, 4,1> x;

		A.setZero();

		int minHeight = std::min(_sizeCam1[1], _sizeCam2[1]);

		float aspectRatioCam1 = float(_sizeCam1[0])/float(minHeight);
		float aspectRatioCam2 = float(_sizeCam2[0])/float(minHeight);

		//camera 1
		obs[0] = _cordTopLeftC1.x();
		obs[1] = _cordTopLeftC1.y();

		A(0,0) = 0;
		A(0,1) = 1;

		A(1,0) = 0;
		A(1,3) = 1;

		obs[2] = _cordTopRightC1.x();
		obs[3] = _cordTopRightC1.y();

		A(2,0) = aspectRatioCam1;
		A(2,1) = 1;

		A(3,0) = 0;
		A(3,3) = 1;


		obs[4] = _cordBottomLeftC1.x();
		obs[5] = _cordBottomLeftC1.y();

		A(4,0) = 0;
		A(4,1) = 1;

		A(5,0) = 1;
		A(5,3) = 1;

		obs[6] = _cordBottomRightC1.x();
		obs[7] = _cordBottomRightC1.y();

		A(6,0) = aspectRatioCam2;
		A(6,1) = 1;

		A(7,0) = 1;
		A(7,3) = 1;

		//camera 2
		obs[8] = _cordTopLeftC2.x();
		obs[9] = _cordTopLeftC2.y();

		A(8,0) = 0;
		A(8,2) = 1;

		A(9,0) = 0;
		A(9,3) = 1;

		obs[10] = _cordTopRightC2.x();
		obs[11] = _cordTopRightC2.y();

		A(10,0) = aspectRatioCam2;
		A(10,2) = 1;

		A(11,0) = 0;
		A(11,3) = 1;


		obs[12] = _cordBottomLeftC2.x();
		obs[13] = _cordBottomLeftC2.y();

		A(12,0) = 0;
		A(12,2) = 1;

		A(13,0) = 1;
		A(13,3) = 1;

		obs[14] = _cordBottomRightC2.x();
		obs[15] = _cordBottomRightC2.y();

		A(14,0) = aspectRatioCam2;
		A(14,2) = 1;

		A(15,0) = 1;
		A(15,3) = 1;

		x = Optimization::leastSquares(A, obs);

		//top lefts points
		_ROIC1TopLeft.y() = x[3];
		_ROIC2TopLeft.y() = x[3];

		_ROIC1TopLeft.x() = x[1];
		_ROIC2TopLeft.x() = x[2];

		//bottom right points
		_ROIC1BottomRight.y() = x[0] + x[3];
		_ROIC2BottomRight.y() = x[0] + x[3];

		_ROIC1BottomRight.x() = x[0]*aspectRatioCam1 + x[1];
		_ROIC2BottomRight.x() = x[0]*aspectRatioCam2 + x[2];

	} else if (roiSetMethod == Same) {

		//top lefts points
		_ROIC1TopLeft.y() = std::min(_cordTopLeftC1.y(), std::min(_cordTopRightC1.y(), std::min(_cordTopLeftC2.y(), _cordTopRightC2.y())));
		_ROIC2TopLeft.y() = _ROIC1TopLeft.y();

		_ROIC1TopLeft.x() = std::min(_cordTopLeftC1.x(), _cordBottomLeftC1.x());
		_ROIC2TopLeft.x() = std::min(_cordTopLeftC2.x(), _cordBottomLeftC2.x());

		//bottom right points
		_ROIC1BottomRight.y() = std::max(_cordBottomLeftC1.y(), std::max(_cordBottomRightC1.y(), std::max(_cordBottomLeftC2.y(), _cordBottomRightC2.y())));
		_ROIC2BottomRight.y() = _ROIC1BottomRight.y();

		_ROIC1BottomRight.x() = std::max(_cordTopRightC1.x(), _cordBottomRightC1.x());
		_ROIC2BottomRight.x() = std::max(_cordTopRightC2.x(), _cordBottomRightC2.x());
	}

	if (_ROIC1TopLeft.y() >= _ROIC1BottomRight.y()) {
		_ROIComputed = false;
		return false;
	}

	if (_ROIC1TopLeft.x() >= _ROIC1BottomRight.x()) {
		_ROIComputed = false;
		return false;
	}

	if (_ROIC2TopLeft.x() >= _ROIC2BottomRight.x()) {
		_ROIComputed = false;
		return false;
	}

	_ROIComputed = true;
	return true;

}

bool StereoRigRectifier::computeResolution(TargetRangeSetMethod resolutionSetMethod) {

	if (!_ROIComputed) {
		return false;
	}

	(void)(resolutionSetMethod); //TODO Implement the three different methods

	//Same
	int minHeight = std::min(_sizeCam1[1], _sizeCam2[1]);
	float vExtend = _ROIC1BottomRight.y() - _ROIC1TopLeft.y();

	_reprojectionFLen = float(minHeight)/vExtend;

	_nsizeCam1.y() = minHeight;
	_nsizeCam2.y() = minHeight;

	_nsizeCam1.x() = _sizeCam1[0];
	_nsizeCam2.x() = _sizeCam2[0];

	_nppCam1 = -_reprojectionFLen*_ROIC1TopLeft;
	_nppCam2 = -_reprojectionFLen*_ROIC2TopLeft;

	_normalizedBasline = _reprojectionFLen/_cam2Tocam1.t.norm();
	_dispDelta = _nppCam2.x() - _nppCam1.x();

	_refocalLengthComputed = true;
	return true;

}

bool StereoRigRectifier::computeBackwardMaps() {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	if (!_refocalLengthComputed) {
		return false;
	}

	_backwardCam1 = Multidim::Array<float,3>(_nsizeCam1.y(), _nsizeCam1.x(), 2);
	_backwardCam2 = Multidim::Array<float,3>(_nsizeCam2.y(), _nsizeCam2.x(), 2);

	for (int i = 0; i < _nsizeCam1.y(); i++) { //height
		for (int j = 0; j < _nsizeCam1.x(); j++) { //width

			Eigen::Vector2f coord = computeBackwardVec(Eigen::Vector2f(j,i),
													   _ppCam1,
													   _fLenCam1Px,
													   _nppCam1,
													   _reprojectionFLen,
													   _CorrRCam1,
													   _kCam1,
													   _tCam1,
													   _BCam1);

			_backwardCam1.at<Nc>(i,j,0) = coord[1];
			_backwardCam1.at<Nc>(i,j,1) = coord[0];

		}
	}

	for (int i = 0; i < _nsizeCam2.y(); i++) { //height
		for (int j = 0; j < _nsizeCam2.x(); j++) { //width

			Eigen::Vector2f coord = computeBackwardVec(Eigen::Vector2f(j,i),
													   _ppCam2,
													   _fLenCam2Px,
													   _nppCam2,
													   _reprojectionFLen,
													   _CorrRCam2,
													   _kCam2,
													   _tCam2,
													   _BCam2);

			_backwardCam2.at<Nc>(i,j,0) = coord[1];
			_backwardCam2.at<Nc>(i,j,1) = coord[0];

		}
	}

	_backwardMapsComputed = true;
	return true;

}

} // namespace Geometry
} // namespace StereoVision
