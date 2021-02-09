#ifndef CORRELATION_BASE_H
#define CORRELATION_BASE_H

#include <MultidimArrays/MultidimArrays.h>

#include <cmath>
#include <limits>

namespace StereoVision {
namespace Correlation {

enum class dispExtractionStartegy{
	Cost = 0,
	Score = 1
};


enum class dispDirection{
	LeftToRight = 0,
	RightToLeft = 1
};

typedef int32_t disp_t;


template<dispExtractionStartegy strategy, class T_CV>
Multidim::Array<disp_t, 2> extractSelectedIndex(Multidim::Array<T_CV, 3> const& costVolume) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto cv_shape = costVolume.shape();

	Multidim::Array<disp_t, 2> disp(cv_shape[0], cv_shape[1]);

	#pragma omp parallel for
	for (int i = 0; i < cv_shape[0]; i++) {
		for (int j = 0; j < cv_shape[1]; j++) {

						T_CV selectedScore = costVolume.template value<Nc>(i,j,0);
			disp_t selectedDisp = 0;

			for (uint32_t d = 1; d < cv_shape[2]; d++) {
				if (strategy == dispExtractionStartegy::Cost) {
					if (costVolume.template value<Nc>(i,j,d) <= selectedScore) {
						selectedScore = costVolume.template value<Nc>(i,j,d);
						selectedDisp = d;
					}
				} else {
					if (costVolume.template value<Nc>(i,j,d) >= selectedScore) {
						selectedScore = costVolume.template value<Nc>(i,j,d);
						selectedDisp = d;
					}
				}
			}

			disp.at<Nc>(i,j) = selectedDisp;

		}
	}

	return disp;
}

template<typename DT, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<DT, 2> selectedIndexToDisp(Multidim::Array<DT, 2> const& selectedIndex,
											   disp_t disp_offset = 0) {

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = selectedIndex.shape();

	Multidim::Array<DT, 2> disp(shape[0], shape[1]);

	for(int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			disp.template at<Nc>(i,j) = deltaSign*selectedIndex.template value<Nc>(i,j) + disp_offset;
		}
	}

	return disp;

}

template<class T_CV>
Multidim::Array<T_CV, 2> selectedCost(Multidim::Array<T_CV, 3> const& costVolume,
									  Multidim::Array<disp_t, 2> const& selectedIndex) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto cv_shape = costVolume.shape();

	Multidim::Array<T_CV, 2> tcv(cv_shape[0], cv_shape[1]);

	#pragma omp parallel for
	for (int i = 0; i < cv_shape[0]; i++) {
		for (int j = 0; j < cv_shape[1]; j++) {
			uint32_t p = selectedIndex.value<Nc>(i,j);
			tcv.template at<Nc>(i,j) = costVolume.value<Nc>(i,j,p);
		}
	}

	return tcv;
}

template<class T_CV>
Multidim::Array<T_CV, 3> truncatedCostVolume(Multidim::Array<T_CV, 3> const& costVolume,
											 Multidim::Array<disp_t, 2> const& selectedIndex,
											 uint8_t cost_vol_radius) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto cv_shape = costVolume.shape();

	Multidim::Array<T_CV, 3> tcv(cv_shape[0], cv_shape[1], cost_vol_radius*2+1);

	#pragma omp parallel for
	for (int i = 0; i < cv_shape[0]; i++) {
		for (int j = 0; j < cv_shape[1]; j++) {
			for (int32_t d = 0; d <= 2*cost_vol_radius; d++) {
				int32_t p = selectedIndex.value<Nc>(i,j)+d-cost_vol_radius;

				if (p < 0 or p >= cv_shape[2]) {
					tcv.template at<Nc>(i,j,d) = 0;
				} else {
					tcv.template at<Nc>(i,j,d) = costVolume.value<Nc>(i,j,p);
				}
			}
		}
	}

	return tcv;

}

template<class T_IB>
Multidim::Array<T_IB, 3> extractInBoundDomain(Multidim::Array<disp_t, 2> const& selectedIndex,
											  uint32_t width,
											  uint8_t cost_vol_radius) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto im_shape = selectedIndex.shape();

	Multidim::Array<T_IB, 3> ib(im_shape[0], im_shape[1], cost_vol_radius*2+1);

	#pragma omp parallel for
	for (int i = 0; i < im_shape[0]; i++) {
		for (int j = 0; j < im_shape[1]; j++) {
			for (uint32_t d = -cost_vol_radius; d <= cost_vol_radius; d++) {
				uint32_t p = selectedIndex.value<Nc>(i,j)+d;

				if (p < 0 or p >= width) {
					ib.template at<Nc>(i,j,d+cost_vol_radius) = 0;
				} else {
					ib.template at<Nc>(i,j,d+cost_vol_radius) = 1;
				}
			}
		}
	}

	return ib;

}

template<class T_L, class T_R, dispDirection dDir>
class condImgRef {
};


template<class T_L, class T_R>
class condImgRef<T_L, T_R, dispDirection::RightToLeft> {
public:

	typedef T_R T_S;
	typedef T_L T_T;

	explicit condImgRef(Multidim::Array<T_L, 2> const& im_l, Multidim::Array<T_R, 2> const& im_r) :
		img_l(im_l),
		img_r(im_r) {
	}

	Multidim::Array<T_S, 2> const& source() const {
		return img_r;
	}

	Multidim::Array<T_T, 2> const& target() const {
		return img_l;
	}

private:
	Multidim::Array<T_L, 2> const& img_l;
	Multidim::Array<T_R, 2> const& img_r;
};


template<class T_L, class T_R>
class condImgRef<T_L, T_R, dispDirection::LeftToRight> {
public:

	typedef T_L T_S;
	typedef T_R T_T;

	explicit condImgRef(Multidim::Array<T_L, 2> const& im_l, Multidim::Array<T_R, 2> const& im_r) :
		img_l(im_l),
		img_r(im_r) {
	}

	Multidim::Array<T_S, 2> const& source() const {
		return img_l;
	}

	Multidim::Array<T_T, 2> const& target() const {
		return img_r;
	}

private:
	Multidim::Array<T_L, 2> const& img_l;
	Multidim::Array<T_R, 2> const& img_r;
};


template<class T_I>
Multidim::Array<float, 2> meanFilter2D (uint8_t h_radius,
										uint8_t v_radius,
										Multidim::Array<T_I, 2> const& in_data) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	size_t box_size = (2*v_radius + 1)*(2*h_radius + 1);

	auto shape = in_data.shape();

	Multidim::Array<float, 2> mean(in_data.shape()[0], in_data.shape()[1]);
	Multidim::Array<float, 2> mHorizontal(in_data.shape()[0], in_data.shape()[1]);

	//horizontal mean filters
	#pragma omp parallel for
	for(long i = 0; i < shape[0]; i++){

		mHorizontal.at<Nc>(i, h_radius) = 0;

		for(long j = 0; j <= 2*h_radius; j++){
			mHorizontal.at<Nc>(i, h_radius) += in_data.template value<Nc>(i,j);
		}

		for(long j = h_radius+1; j < shape[1]-h_radius; j++) {

			mHorizontal.at<Nc>(i, j) = mHorizontal.at<Nc>(i, j-1);
			mHorizontal.at<Nc>(i, j) += in_data.template value<Nc>(i,j + h_radius) - in_data.template value<Nc>(i,j - h_radius -1);
		}
	}

	//vertical mean filter
	#pragma omp parallel for
	for(long j = h_radius; j < shape[1]-h_radius; j++){

		mean.at<Nc>(v_radius, j) = 0.;

		for(long i = 0; i <= 2*v_radius; i++){
			mean.at<Nc>(v_radius, j) += mHorizontal.at<Nc>(i, j);
		}

		for(long i = v_radius+1; i < shape[0]-v_radius; i++){
			mean.at<Nc>(i, j) = mean.at<Nc>(i-1, j);
			mean.at<Nc>(i, j) += mHorizontal.at<Nc>(i + v_radius, j) - mHorizontal.at<Nc>(i - v_radius -1, j);
		}
	}

	/*#pragma omp parallel for
	for(long j = h_radius; j < shape[1]-h_radius; j++){
		for(long i = v_radius; i < shape[0]-v_radius; i++) {
			mean.at<Nc>(i, j) = 0;

			for(int k = -v_radius; k <= v_radius; k++) {
				for(int l = -h_radius; l <= h_radius; l++) {
					mean.at<Nc>(i, j) += in_data.template value<Nc>(i+k,j+l);
				}
			}
		}
	}*/

	#pragma omp parallel for
	for(long j = h_radius; j < shape[1]-h_radius; j++){
		for(long i = v_radius; i < shape[0]-v_radius; i++) {
			mean.at<Nc>(i, j) /= box_size;
		}
	}// mean filter computed

	return mean;

}

template<disp_t deltaSign = 1, bool rmIncompleteRanges = false>
bool indexIsInbound(int i,
					int j,
					int d,
					uint8_t h_radius,
					uint8_t v_radius,
					disp_t disp_width,
					disp_t disp_offset,
					std::array<int,2> s_shape,
					std::array<int,2> t_shape) {

	if (i < v_radius or i + v_radius >= s_shape[0]) { // if we are too high or too low
		return false;
	} else if (j < h_radius or j + h_radius >= s_shape[1]) { // if the source patch is partially outside the image
		return false;
	} else if (!rmIncompleteRanges and (j + disp_offset + deltaSign*d < h_radius or
			   j + disp_offset + deltaSign*d + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
		return false;
	} else if (rmIncompleteRanges and (j + disp_offset < h_radius or
			   j + disp_offset + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
		return false;
	} else if (rmIncompleteRanges and (j + disp_offset + deltaSign*disp_width < h_radius or
				j + disp_offset + deltaSign*disp_width + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
		return false;
	}

	return true;
}

template<class T_L,
		 class T_R,
		 class T_CV,
		 T_CV costFunction(T_L, T_R),
		 dispExtractionStartegy strat,
		 dispDirection dDir = dispDirection::RightToLeft,
		 bool rmIncompleteRanges = false>
Multidim::Array<T_CV, 3> buildCostVolume(Multidim::Array<T_L, 2> const& img_l,
										 Multidim::Array<T_R, 2> const& img_r,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	condImgRef<T_L, T_R, dDir> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<T_CV, 3>(0,0,0);
	}

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 2> const& img_s = cir.source();

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 2> const& img_t = cir.target();

	auto s_shape = img_s.shape();
	auto t_shape = img_t.shape();

	Multidim::Array<float, 3> costVolume(s_shape[0], s_shape[1], disp_width);

	#pragma omp parallel for
	for (int i = 0; i < s_shape[0]; i++) {

		for (int j = 0; j < s_shape[1]; j++) {

			for (int d = 0; d < disp_width; d++) {

				const T_CV errorVal;

				if (std::numeric_limits<T_CV>::has_infinity()) {
					errorVal = ((strat == dispExtractionStartegy::Cost) ? 1 : -1)*std::numeric_limits<T_CV>::infinity();
				} else {
					errorVal = (strat == dispExtractionStartegy::Cost) ? std::numeric_limits<T_CV>::max() : std::numeric_limits<T_CV>::lowest();
				}

				if (!indexIsInbound<deltaSign, rmIncompleteRanges>(i,
																   j,
																   d,
																   h_radius,
																   v_radius,
																   disp_width,
																   disp_offset,
																   s_shape,
																   t_shape)) { // if we are out of bound

					costVolume.at(i,j,d) = errorVal;

				} else {

					T_CV s = 0;

					for(int k = -h_radius; k <= h_radius; k++) {

						for (int l = -v_radius; l <= v_radius; l++) {

							T_L left = (dDir == dispDirection::RightToLeft) ?
										img_l.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l) :
										img_l.template value<Nc>(i+k, j+l);

							T_R right = (dDir == dispDirection::RightToLeft) ?
										img_r.template value<Nc>(i+k, j+l) :
										img_r.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l);

							s += costFunction(left, right);
						}
					}

					costVolume.at(i,j,d) = s;
				}
			}

		}

	}

	return costVolume;

}

typedef float CostFunctionType(float, float);
template<class T_S, class T_T, CostFunctionType costFunction, dispExtractionStartegy strat, disp_t deltaSign = 1, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> buildZeroMeanCostVolume(Multidim::Array<T_S, 2> const& img_s,
										   Multidim::Array<T_T, 2> const& img_t,
										   Multidim::Array<float, 2> const& s_mean,
										   Multidim::Array<float, 2> const& t_mean,
										   uint8_t h_radius,
										   uint8_t v_radius,
										   disp_t disp_width,
										   disp_t disp_offset = 0) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto s_shape = img_s.shape();
	auto t_shape = img_t.shape();

	Multidim::Array<float, 3> costVolume(s_shape[0], s_shape[1], disp_width);

	#pragma omp parallel for
	for (int i = 0; i < s_shape[0]; i++) {

		for (int j = 0; j < s_shape[1]; j++) {

			for (int d = 0; d < disp_width; d++) {

				const float errorVal = ((strat == dispExtractionStartegy::Cost) ? 1 : -1)*std::numeric_limits<float>::infinity();

				if (!indexIsInbound<deltaSign, rmIncompleteRanges>(i,
																   j,
																   d,
																   h_radius,
																   v_radius,
																   disp_width,
																   disp_offset,
																   s_shape,
																   t_shape)) { // if we are out of bound

					costVolume.at(i,j,d) = errorVal;

				} else {

					float s = 0;

					for(int k = -h_radius; k <= h_radius; k++) {

						for (int l = -v_radius; l <= v_radius; l++) {
							float source = img_s.template value<Nc>(i+k, j+l) - s_mean.value<Nc>(i,j);
							float target = img_t.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d);
							s += costFunction(source, target);
						}
					}

					costVolume.at(i,j,d) = s;
				}
			}

		}

	}

	return costVolume;

}

template<class T_L,
		 class T_R,
		 CostFunctionType costFunction,
		 dispExtractionStartegy strat,
		 dispDirection dDir = dispDirection::RightToLeft,
		 bool rmIncompleteRanges = false>
Multidim::Array<float, 3> buildZeroMeanCostVolume(Multidim::Array<T_L, 2> const& img_l,
												  Multidim::Array<T_R, 2> const& img_r,
												  uint8_t h_radius,
												  uint8_t v_radius,
												  disp_t disp_width,
												  disp_t disp_offset = 0) {

	condImgRef<T_L, T_R, dDir> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>(0,0,0);
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_l);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 2> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 2> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;

	return buildZeroMeanCostVolume<typename condImgRef<T_L, T_R, dDir>::T_S,
			typename condImgRef<T_L, T_R, dDir>::T_T,
			costFunction,
			strat,
			deltaSign,
			rmIncompleteRanges>(s_img,
								t_img,
								s_mean,
								t_mean,
								h_radius,
								v_radius,
								disp_width,
								disp_offset);

}

} // namespace Correlation
} // namespace StereoVision

#endif // CORRELATION_BASE_H
