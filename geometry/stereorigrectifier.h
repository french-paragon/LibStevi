#ifndef STEREOVISION_GEOMETRY_STEREORIGRECTIFIER_H
#define STEREOVISION_GEOMETRY_STEREORIGRECTIFIER_H

#include "./rotations.h"
#include "MultidimArrays/MultidimArrays.h"

namespace StereoVision {
namespace Geometry {

class StereoRigRectifier
{
public:

	/*!
	 * \brief The TargetRangeSetMethod enum describe how automatic methods to choose the final images resolutions and ROI will operate
	 */
	enum TargetRangeSetMethod {
		Minimal, //! \brief target the minimal range so that no filling up or interpolation is required.
		Maximal, //! \brief target the maximal range, even if some values needs to be filled up or interpolated.
		Same, //! \brief target the range to match the original image.
	};

	StereoRigRectifier(ShapePreservingTransform cam2tocam1,
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
					   std::optional<Eigen::Vector2f> BCam2);

	void clear();
	bool compute(TargetRangeSetMethod roiSetMethod, TargetRangeSetMethod resolutionSetMethod);

	inline Multidim::Array<float,3> const& backWardMapCam1() const {
		return _backwardCam1;
	}

	inline Multidim::Array<float,3> const& backWardMapCam2() const {
		return _backwardCam2;
	}


	inline Eigen::Matrix3f CorrRCam1() const {
		return _CorrRCam1;
	}
	inline Eigen::Matrix3f CorrRCam2() const {
		return _CorrRCam2;
	}

	inline float reprojectionFLen() const {
		return _reprojectionFLen;
	}
	inline Eigen::Vector2i nsizeCam1() const {
		return _nsizeCam1;
	}
	inline Eigen::Vector2i nsizeCam2() const {
		return _nsizeCam2;
	}
	inline Eigen::Vector2f newPrincipalPointCam1() const {
		return _nppCam1;
	}
	inline Eigen::Vector2f newPrincipalPointCam2() const {
		return _nppCam2;
	}

	inline float normalizedBasline() const {
		return _normalizedBasline;
	}
	inline float dispDelta() const {
		return _dispDelta;
	}

	inline bool hasResultsComputed() const {
		return _backwardMapsComputed;
	}

	bool computeOptimalCamsRots();
	Eigen::Vector2f computeForwardVec(Eigen::Vector2f const& vec,
									  Eigen::Vector2f const& pp,
									  float f,
									  Eigen::Matrix3f const& R);

	Eigen::Vector2f computeBackwardVec(Eigen::Vector2f const& vec,
									   Eigen::Vector2f const& pp,
									   float f,
									   Eigen::Vector2f const& pp_back,
									   float f_back,
									   Eigen::Matrix3f const& R,
									   std::optional<Eigen::Vector3f> k,
									   std::optional<Eigen::Vector2f> t,
									   std::optional<Eigen::Vector2f> B);
	bool computeForwardMaps();
	bool computeROIs(TargetRangeSetMethod roiSetMethod);
	bool computeResolution(TargetRangeSetMethod resolutionSetMethod);
	bool computeBackwardMaps();


protected:

	ShapePreservingTransform _cam2Tocam1;

	bool _CorrRComputed;
	Eigen::Matrix3f _CorrRCam1;
	Eigen::Matrix3f _CorrRCam2;

	float _fLenCam1Px;
	Eigen::Vector2f _ppCam1;
	Eigen::Vector2i _sizeCam1;
	std::optional<Eigen::Vector3f> _kCam1;
	std::optional<Eigen::Vector2f> _tCam1;
	std::optional<Eigen::Vector2f> _BCam1;

	float _fLenCam2Px;
	Eigen::Vector2f _ppCam2;
	Eigen::Vector2i _sizeCam2;
	std::optional<Eigen::Vector3f> _kCam2;
	std::optional<Eigen::Vector2f> _tCam2;
	std::optional<Eigen::Vector2f> _BCam2;

	bool _forwardMapsComputed;
	Eigen::Vector2f _cordTopLeftC1;
	Eigen::Vector2f _cordTopRightC1;
	Eigen::Vector2f _cordBottomLeftC1;
	Eigen::Vector2f _cordBottomRightC1;

	Eigen::Vector2f _cordTopLeftC2;
	Eigen::Vector2f _cordTopRightC2;
	Eigen::Vector2f _cordBottomLeftC2;
	Eigen::Vector2f _cordBottomRightC2;

	bool _ROIComputed;
	Eigen::Vector2f _ROIC1TopLeft;
	Eigen::Vector2f _ROIC1BottomRight;

	bool _refocalLengthComputed;
	float _reprojectionFLen;
	Eigen::Vector2i _nsizeCam1;
	Eigen::Vector2i _nsizeCam2;
	Eigen::Vector2f _nppCam1;
	Eigen::Vector2f _nppCam2;

	Eigen::Vector2f _ROIC2TopLeft;
	Eigen::Vector2f _ROIC2BottomRight;

	bool _backwardMapsComputed;
	Multidim::Array<float,3> _backwardCam1;
	Multidim::Array<float,3> _backwardCam2;

	float _normalizedBasline;
	float _dispDelta;
};

} // namespace Geometry
} // namespace StereoVision

#endif // STEREOVISION_GEOMETRY_STEREORIGRECTIFIER_H
