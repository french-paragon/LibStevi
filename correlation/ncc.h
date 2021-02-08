#ifndef STEREOVISION_CORRELATION_NCC_H
#define STEREOVISION_CORRELATION_NCC_H

#include "./correlation_base.h"

#include <limits>
#include <iostream>

namespace StereoVision {
namespace Correlation {

template<class T_I>
Multidim::Array<float, 2> sigmaFilter(uint8_t h_radius,
									  uint8_t v_radius,
									  Multidim::Array<float, 2> const& mean,
									  Multidim::Array<T_I, 2> const& in_data) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 2> sigma(in_data.shape()[0], in_data.shape()[1]);

	auto shape = in_data.shape();

	#pragma omp parallel for
	for(long i = v_radius; i < shape[0]-v_radius; i++){
		for(long j = h_radius; j < shape[1]-h_radius; j++){

			float s = 0.;

			for(int k = -v_radius; k <= v_radius; k++) {

				for (int l = -h_radius; l <= h_radius; l++) {

					float tmp = in_data.template value<Nc>(i+k, j+l) - mean.value<Nc>(i,j);
					s += tmp*tmp;
				}
			}

			sigma.at<Nc>(i, j) = sqrtf(s);
		}
	}

	return sigma;
}

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

template<class T_S, class T_T, disp_t deltaSign = 1, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> crossCorrelation(Multidim::Array<T_S, 2> const& img_s,
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

				if (i < v_radius or i + v_radius >= s_shape[0]) { // if we are too high or too low
					costVolume.at(i,j,d) = -std::numeric_limits<float>::infinity();
				} else if (j < h_radius or j + h_radius >= s_shape[1]) { // if the source patch is partially outside the image
					costVolume.at(i,j,d) = -std::numeric_limits<float>::infinity();
				} else if (!rmIncompleteRanges and (j + disp_offset + deltaSign*d < h_radius or
						   j + disp_offset + deltaSign*d + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
					costVolume.at(i,j,d) = -std::numeric_limits<float>::infinity();
				} else if (rmIncompleteRanges and (j + disp_offset < h_radius or
						   j + disp_offset + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
					costVolume.at(i,j,d) = -std::numeric_limits<float>::infinity();
				} else if (rmIncompleteRanges and (j + disp_offset + deltaSign*disp_width < h_radius or
							j + disp_offset + deltaSign*disp_width + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
					costVolume.at(i,j,d) = -std::numeric_limits<float>::infinity();
				}  else {

					float s = 0;

					for(int k = -h_radius; k <= h_radius; k++) {

						for (int l = -v_radius; l <= v_radius; l++) {
                                                        float source = img_s.template value<Nc>(i+k, j+l) - s_mean.value<Nc>(i,j);
                                                        float target = img_t.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d);
							s += source*target;
						}
					}

					costVolume.at(i,j,d) = s;
				}
			}

		}
	}

	return costVolume;

}

template<class T_S, class T_T, disp_t deltaSign = 1, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> normalizedCrossCorrelation(Multidim::Array<float, 3> const& cc,
													 Multidim::Array<float, 2> const& s_sigma,
													 Multidim::Array<float, 2> const& t_sigma,
													 uint8_t h_radius,
													 uint8_t v_radius,
													 disp_t disp_width,
													 disp_t disp_offset = 0) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto s_shape = cc.shape();
	auto t_shape = t_sigma.shape();

	Multidim::Array<float, 3> costVolume(s_shape);

	#pragma omp parallel for
	for (int i = 0; i < s_shape[0]; i++) {

		for (int j = 0; j < s_shape[1]; j++) {

			for (int d = 0; d < disp_width; d++) {

				if (i < v_radius or i + v_radius >= s_shape[0]) { // if we are too high or too low
					costVolume.at(i,j,d) = -1.;
				} else if (j < h_radius or j + h_radius >= s_shape[1]) { // if the source patch is partially outside the image
					costVolume.at(i,j,d) = -1.;
				} else if (!rmIncompleteRanges and (j + disp_offset + deltaSign*d < h_radius or
						   j + disp_offset + deltaSign*d + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
					costVolume.at(i,j,d) = -1.;
				} else if (rmIncompleteRanges and (j + disp_offset < h_radius or
						   j + disp_offset + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
					costVolume.at(i,j,d) = -1.;
				} else if (rmIncompleteRanges and (j + disp_offset + deltaSign*disp_width < h_radius or
							j + disp_offset + deltaSign*disp_width + h_radius >= t_shape[1])) { // if the target patch is partially outside the image
					costVolume.at(i,j,d) = -1.;
				}  else {

					float s = cc.value(i,j,d);
					s /= s_sigma.value<Nc>(i,j)*t_sigma.value<Nc>(i, j + disp_offset + deltaSign*d);

					costVolume.at(i,j,d) = s;
				}
			}

		}
	}

	return costVolume;

}

template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> ccCostVolume(Multidim::Array<T_L, 2> const& img_l,
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
		return Multidim::Array<float, 3>();
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_l);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 2> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 2> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;

	return crossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
			typename condImgRef<T_L, T_R, dDir>::T_T,
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

template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> nccCostVolume(Multidim::Array<T_L, 2> const& img_l,
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
		return Multidim::Array<float, 3>();
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_l);

	Multidim::Array<float, 2> sigmaLeft = sigmaFilter(h_radius, v_radius, meanLeft, img_l);
	Multidim::Array<float, 2> sigmaRight = sigmaFilter(h_radius, v_radius, meanRight, img_r);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 2> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;
	Multidim::Array<float, 2> const& s_sigma = (dDir == dispDirection::RightToLeft) ? sigmaRight : sigmaLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 2> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;
	Multidim::Array<float, 2> const& t_sigma = (dDir == dispDirection::RightToLeft) ? sigmaLeft : sigmaRight;

	Multidim::Array<float, 3> costVolume = crossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
				typename condImgRef<T_L, T_R, dDir>::T_T,
				deltaSign,
				rmIncompleteRanges>(s_img,
									t_img,
									s_mean,
									t_mean,
									h_radius,
									v_radius,
									disp_width,
									disp_offset);

	return normalizedCrossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
			typename condImgRef<T_L, T_R, dDir>::T_T,
			deltaSign,
			rmIncompleteRanges> (costVolume,
								 s_sigma,
								 t_sigma,
								 h_radius,
								 v_radius,
								 disp_width,
								 disp_offset);

}

template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refinedNCCDisp(Multidim::Array<T_L, 2> const& img_l,
										 Multidim::Array<T_R, 2> const& img_r,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	condImgRef<T_L, T_R, dDir> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
                return Multidim::Array<float, 2>(0,0);
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_l);

	Multidim::Array<float, 2> sigmaLeft = sigmaFilter(h_radius, v_radius, meanLeft, img_l);
	Multidim::Array<float, 2> sigmaRight = sigmaFilter(h_radius, v_radius, meanRight, img_r);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 2> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;
	Multidim::Array<float, 2> const& s_sigma = (dDir == dispDirection::RightToLeft) ? sigmaRight : sigmaLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 2> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;
	Multidim::Array<float, 2> const& t_sigma = (dDir == dispDirection::RightToLeft) ? sigmaLeft : sigmaRight;

	Multidim::Array<float, 3> cc = crossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
				typename condImgRef<T_L, T_R, dDir>::T_T,
				deltaSign,
				rmIncompleteRanges>(s_img,
									t_img,
									s_mean,
									t_mean,
									h_radius,
									v_radius,
									disp_width,
									disp_offset);


	Multidim::Array<float, 3> costVolume = normalizedCrossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
			typename condImgRef<T_L, T_R, dDir>::T_T,
			deltaSign,
			rmIncompleteRanges> (cc,
								 s_sigma,
								 t_sigma,
								 h_radius,
								 v_radius,
								 disp_width,
								 disp_offset);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Score>(costVolume);

	auto d_shape = disp.shape();
	auto t_shape = t_img.shape();

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		for (int j = 0; j < d_shape[1]; j++) {

			disp_t const& d = disp.at<Nc>(i,j);

			if (i < v_radius or i + v_radius >= d_shape[0]) { // if we are too high or too low
				refinedDisp.at<Nc>(i,j) = d;
			} else if (j < h_radius+1 or j + h_radius + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (!rmIncompleteRanges and (j + disp_offset + deltaSign*d < h_radius + 1 or
					   j + disp_offset + deltaSign*d + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset < h_radius + 1 or
					   j + disp_offset + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset + deltaSign*disp_width < h_radius + 1 or
						j + disp_offset + deltaSign*disp_width + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			}  else {

				int jd = j + d;

				float k_plus = 0;
				float k_minus = 0;

				for(int k = -h_radius; k <= h_radius; k++) {

					for (int l = -v_radius; l <= v_radius; l++) {

						float k0 = t_img.template value<Nc>(i+k, jd+l-1) - t_mean.value<Nc>(i,jd-1);
						float k1 = t_img.template value<Nc>(i+k, jd+l) - t_mean.value<Nc>(i,jd);
						float k2 = t_img.template value<Nc>(i+k, jd+l+1) - t_mean.value<Nc>(i,jd+1);

						k_plus += k1*k2;
						k_minus += k0*k1;
					}
				}

				float sigma_m1 = t_sigma.value<Nc>(i,jd-1)*t_sigma.value<Nc>(i,jd-1);
				float sigma_0 = t_sigma.value<Nc>(i,jd)*t_sigma.value<Nc>(i,jd);
				float sigma_1 = t_sigma.value<Nc>(i,jd+1)*t_sigma.value<Nc>(i,jd+1);

				float a1_plus = sigma_0 + sigma_1 - 2*k_plus;
				float a1_minus = sigma_m1 + sigma_0 - 2*k_minus;

				float a2_plus = 2*k_plus - 2*sigma_0;
				float a2_minus = 2*k_minus - 2*sigma_m1;

				float a3_plus = sigma_0;
				float a3_minus = sigma_m1;

				float rho_m1 = cc.value<Nc>(i,j,d-1);
				float rho_0 = cc.value<Nc>(i,j,d);
				float rho_1 = cc.value<Nc>(i,j,d+1);

				float b1_plus = (rho_1 - rho_0);
				float b1_minus = (rho_0 - rho_m1);

				float b2_plus = rho_0;
				float b2_minus = rho_m1;

				float c1_plus = 1./2. * a2_plus * b1_plus - a1_plus * b2_plus;
				float c1_minus = 1./2. * a2_minus * b1_minus - a1_minus * b2_minus;

				float c2_plus = (a3_plus * b1_plus) - (1./2. * a2_plus * b2_plus);
				float c2_minus = (a3_minus * b1_minus) - (1./2. * a2_minus * b2_minus);

				float DeltaD_plus = -c2_plus/c1_plus;
				float DeltaD_minus = -c2_minus/c1_minus;

				float score = costVolume.value<Nc>(i,j,d);

				float DeltaD = 0;

				if (DeltaD_plus > 0 and DeltaD_plus < 1) {

					float interpRho = (rho_1 - rho_0)*DeltaD_plus + rho_0;
					float interpSigma_t = a1_plus*DeltaD_plus*DeltaD_plus + a2_plus*DeltaD_plus + a3_plus;
					float interpNCC = interpRho/(s_sigma.value<Nc>(i,j)*sqrtf(interpSigma_t));

					if (interpNCC > score) {
						score = interpNCC;
						DeltaD = DeltaD_plus;
					}

				}

				if (DeltaD_minus > 0 and DeltaD_minus < 1) {

					float interpRho = (rho_0 - rho_m1)*DeltaD_minus + rho_m1;
					float interpSigma_t = a1_minus*DeltaD_minus*DeltaD_minus + a2_minus*DeltaD_minus + a3_minus;
					float interpNCC = interpRho/(s_sigma.value<Nc>(i,j)*sqrtf(interpSigma_t));

					if (interpNCC > score) {
						score = interpNCC;
						DeltaD = DeltaD_minus - 1;
					}

				}

				refinedDisp.at<Nc>(i,j) = d + DeltaD;

			}

		}
	}

	return refinedDisp;

}

} // namespace Correlation
} // namespace StereoVision

#endif // STEREOVISION_CORRELATION_NCC_H