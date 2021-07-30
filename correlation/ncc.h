#ifndef STEREOVISION_CORRELATION_NCC_H
#define STEREOVISION_CORRELATION_NCC_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021  Paragon<french.paragon@gmail.com>

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

#include "./correlation_base.h"
#include "./cost_based_refinement.h"
#include "./cross_correlations.h"
#include "./unfold.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>

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
		#pragma omp simd
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

template<class T_I>
Multidim::Array<float, 2> sigmaFilter(uint8_t h_radius,
									  uint8_t v_radius,
									  Multidim::Array<float, 2> const& mean,
									  Multidim::Array<T_I, 3> const& in_data) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 2> sigma(in_data.shape()[0], in_data.shape()[1]);

	auto shape = in_data.shape();

	#pragma omp parallel for
	for(long i = v_radius; i < shape[0]-v_radius; i++){
		#pragma omp simd
		for(long j = h_radius; j < shape[1]-h_radius; j++){

			float s = 0.;

			for(int k = -v_radius; k <= v_radius; k++) {

				for (int l = -h_radius; l <= h_radius; l++) {

					for (int c = 0; c < shape[2]; c++) {

						float tmp = in_data.template value<Nc>(i+k, j+l, c) - mean.value<Nc>(i,j);
						s += tmp*tmp;
					}
				}
			}

			sigma.at<Nc>(i, j) = sqrtf(s);
		}
	}

	return sigma;
}

#pragma omp declare simd
inline float product(float p1, float p2) {
	return p1*p2;
}

template<class T_S, class T_T, int nImDim = 2, disp_t deltaSign = 1, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> crossCorrelation(Multidim::Array<T_S, nImDim> const& img_s,
										   Multidim::Array<T_T, nImDim> const& img_t,
										   Multidim::Array<float, 2> const& s_mean,
										   Multidim::Array<float, 2> const& t_mean,
										   uint8_t h_radius,
										   uint8_t v_radius,
										   disp_t disp_width,
										   disp_t disp_offset = 0) {

	return buildZeroMeanCostVolume<T_S,
			T_T,
			product,
			dispExtractionStartegy::Score,
			deltaSign,
			rmIncompleteRanges>(img_s,
								img_t,
								s_mean,
								t_mean,
								h_radius,
								v_radius,
								disp_width,
								disp_offset);

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
		#pragma omp simd
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

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> ccCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
									   Multidim::Array<T_R, nImDim> const& img_r,
									   uint8_t h_radius,
									   uint8_t v_radius,
									   disp_t disp_width,
									   disp_t disp_offset = 0) {

	return buildZeroMeanCostVolume<T_L,
			T_R,
			product,
			dispExtractionStartegy::Score,
			nImDim,
			dDir,
			rmIncompleteRanges>(img_l,
								img_r,
								h_radius,
								v_radius,
								disp_width,
								disp_offset);

}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> nccCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
										Multidim::Array<T_R, nImDim> const& img_r,
										uint8_t h_radius,
										uint8_t v_radius,
										disp_t disp_width,
										disp_t disp_offset = 0) {

	condImgRef<T_L, T_R, dDir, nImDim> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>(0,0,0);
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 3>(0,0,0);
		}
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_r);

	Multidim::Array<float, 2> sigmaLeft = sigmaFilter(h_radius, v_radius, meanLeft, img_l);
	Multidim::Array<float, 2> sigmaRight = sigmaFilter(h_radius, v_radius, meanRight, img_r);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, nImDim> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;
	Multidim::Array<float, 2> const& s_sigma = (dDir == dispDirection::RightToLeft) ? sigmaRight : sigmaLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, nImDim> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;
	Multidim::Array<float, 2> const& t_sigma = (dDir == dispDirection::RightToLeft) ? sigmaLeft : sigmaRight;

	Multidim::Array<float, 3> costVolume = crossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
				typename condImgRef<T_L, T_R, dDir>::T_T,
				nImDim,
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


template<dispDirection dDir = dispDirection::RightToLeft>
inline Multidim::Array<float, 3> nccFeatureVolume2CostVolume(Multidim::Array<float, 3> const& feature_vol_l,
															 Multidim::Array<float, 3> const& feature_vol_r,
															 Multidim::Array<float, 2> const& mean_l,
															 Multidim::Array<float, 2> const& mean_r,
															 Multidim::Array<float, 2> const& sigma_l,
															 Multidim::Array<float, 2> const& sigma_r,
															 disp_t disp_width) {

	Multidim::Array<float, 3> normalized_feature_volume_l = zeromeanNormalizedFeatureVolume(feature_vol_l, mean_l, sigma_l);
	Multidim::Array<float, 3> normalized_feature_volume_r = zeromeanNormalizedFeatureVolume(feature_vol_r, mean_r, sigma_r);

	return aggregateCost<matchingFunctions::ZNCC, dDir>(normalized_feature_volume_l, normalized_feature_volume_r, disp_width);

}

template<dispDirection dDir = dispDirection::RightToLeft>
inline Multidim::Array<float, 3> nccFeatureVolume2CostVolume(Multidim::Array<float, 3> const& feature_vol_l,
													  Multidim::Array<float, 3> const& feature_vol_r,
													  disp_t disp_width)
{

	return featureVolume2CostVolume<matchingFunctions::ZNCC, dDir>(feature_vol_l, feature_vol_r, disp_width);
}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> nccUnfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   uint8_t h_radius,
												   uint8_t v_radius,
												   disp_t disp_width)
{

	return unfoldBasedCostVolume<matchingFunctions::ZNCC,T_L,T_R,nImDim,dDir>(img_l, img_r, h_radius, v_radius, disp_width);
}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 3> nccUnfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   UnFoldCompressor const& compressor,
												   disp_t disp_width)
{

	return unfoldBasedCostVolume<matchingFunctions::ZNCC,T_L,T_R,nImDim,dDir>(img_l, img_r, compressor, disp_width);
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
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_r);

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
				2,
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
		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t const& d = disp.at<Nc>(i,j);

			int jd = j + deltaSign*d;

			if (i < v_radius or i + v_radius >= d_shape[0]) { // if we are too high or too low
				refinedDisp.at<Nc>(i,j) = d;
			} else if (j < h_radius+1 or j + h_radius + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (d == 0 or d+1 >= disp_width) {
				refinedDisp.at<Nc>(i,j) = d;
			}  else if (!rmIncompleteRanges and (jd + disp_offset < h_radius + 1 or
					   jd + disp_offset + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset < h_radius + 1 or
					   j + disp_offset + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset + deltaSign*disp_width < h_radius + 1 or
						j + disp_offset + deltaSign*disp_width + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			}  else {

				float k_plus = 0;
				float k_minus = 0;

				for(int k = -v_radius; k <= v_radius; k++) {

					for (int l = -h_radius; l <= h_radius; l++) {

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

template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refinedNCCDisp(Multidim::Array<T_L, 3> const& img_l,
										 Multidim::Array<T_R, 3> const& img_r,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	condImgRef<T_L, T_R, dDir, 3> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0] or l_shape[2] != r_shape[2]) {
				return Multidim::Array<float, 2>(0,0);
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_r);

	Multidim::Array<float, 2> sigmaLeft = sigmaFilter(h_radius, v_radius, meanLeft, img_l);
	Multidim::Array<float, 2> sigmaRight = sigmaFilter(h_radius, v_radius, meanRight, img_r);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 3> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;
	Multidim::Array<float, 2> const& s_sigma = (dDir == dispDirection::RightToLeft) ? sigmaRight : sigmaLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 3> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;
	Multidim::Array<float, 2> const& t_sigma = (dDir == dispDirection::RightToLeft) ? sigmaLeft : sigmaRight;

	Multidim::Array<float, 3> cc = crossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
				typename condImgRef<T_L, T_R, dDir>::T_T,
				3,
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

			int jd = j + deltaSign*d;

			if (i < v_radius or i + v_radius >= d_shape[0]) { // if we are too high or too low
				refinedDisp.at<Nc>(i,j) = d;
			} else if (j < h_radius+1 or j + h_radius + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (d == 0 or d+1 >= disp_width) {
				refinedDisp.at<Nc>(i,j) = d;
			}  else if (!rmIncompleteRanges and (jd + disp_offset < h_radius + 1 or
					   jd + disp_offset + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset < h_radius + 1 or
					   j + disp_offset + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (rmIncompleteRanges and (j + disp_offset + deltaSign*disp_width < h_radius + 1 or
						j + disp_offset + deltaSign*disp_width + h_radius + 1 >= t_shape[1])) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			}  else {

				float k_plus = 0;
				float k_minus = 0;

				for(int k = -v_radius; k <= v_radius; k++) {

					for (int l = -h_radius; l <= h_radius; l++) {

						for (int c = 0; c < t_shape[2]; c++) {

							float k0 = t_img.template value<Nc>(i+k, jd+l-1, c) - t_mean.value<Nc>(i,jd-1);
							float k1 = t_img.template value<Nc>(i+k, jd+l, c) - t_mean.value<Nc>(i,jd);
							float k2 = t_img.template value<Nc>(i+k, jd+l+1, c) - t_mean.value<Nc>(i,jd+1);

							k_plus += k1*k2;
							k_minus += k0*k1;
						}
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


template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refinedNCCCostSymmetricDisp(Multidim::Array<T_L, 2> const& img_l,
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
		return Multidim::Array<float, 2>(0,0);
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_r);

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
				2,
				deltaSign,
				rmIncompleteRanges>(s_img,
									t_img,
									s_mean,
									t_mean,
									h_radius,
									v_radius,
									disp_width,
									disp_offset);


	Multidim::Array<float, 3> cv = normalizedCrossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
			typename condImgRef<T_L, T_R, dDir>::T_T,
			deltaSign,
			rmIncompleteRanges> (cc,
								 s_sigma,
								 t_sigma,
								 h_radius,
								 v_radius,
								 disp_width,
								 disp_offset);

	Multidim::Array<disp_t, 2> raw_disp = extractSelectedIndex<dispExtractionStartegy::Score, float>(cv);

	Multidim::Array<float, 3> tcv = truncatedCostVolume(cv, raw_disp, h_radius, v_radius, 1);

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = raw_disp.shape();

	Multidim::Array<float, 2> refined(shape);

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float cm1 = tcv.value<Nc>(i,j,0);
			float c0 = tcv.value<Nc>(i,j,1);
			float c1 = tcv.value<Nc>(i,j,2);

			float delta = (cm1 - c1)/(2*(c1 - 2*c0 + cm1));

			disp_t d = raw_disp.value<Nc>(i,j);

			if (j + disp_offset + std::abs(deltaSign*d) + h_radius + 1 < t_img.shape()[1] and j + disp_offset + std::abs(deltaSign*d) - h_radius - 1 > 0 and
				i - v_radius > 0 and i + v_radius < t_img.shape()[0] and std::isfinite(cm1) and std::isfinite(c0) and std::isfinite(c1)) {

				float fm1 = 0;
				float f0 = 0;
				float f1 = 0;
				float sigmaSource = 0;

				disp_t dir = 1;

				if (delta > 0) {
					dir = -1;
				}

				for(int k = -v_radius; k <= v_radius; k++) {

					for (int l = -h_radius; l <= h_radius; l++) {
						float source = (s_img.template value<Nc>(i+k, j+l) - s_mean.value<Nc>(i,j) + s_img.template value<Nc>(i+k, j+l+dir) - s_mean.value<Nc>(i,j+dir))/2.;

						float targetm1 = t_img.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l - 1) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d - 1);
						float target0 = t_img.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d);
						float target1 = t_img.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l + 1) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d + 1);

						fm1 += source*targetm1;
						f0 += source*target0;
						f1 += source*target1;

						sigmaSource += source*source;
					}
				}

				sigmaSource = sqrtf(sigmaSource);
				fm1 /= sigmaSource*t_sigma.template value<Nc>(i,j + disp_offset + deltaSign*d - 1);
				f0 /= sigmaSource*t_sigma.template value<Nc>(i,j + disp_offset + deltaSign*d);
				f1 /= sigmaSource*t_sigma.template value<Nc>(i,j + disp_offset + deltaSign*d + 1);

				float delta2 = (fm1 - f1)/(2*(f1 - 2*f0 + fm1)) - dir*0.5;

				if (std::fabs(delta2) < 1.) {
					delta = (delta + delta2)/2;
				}

			}

			refined.at<Nc>(i,j) = d + delta;

		}

	}

	return refined;

}



template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 2> refinedNCCCostSymmetricDisp(Multidim::Array<T_L, 3> const& img_l,
													  Multidim::Array<T_R, 3> const& img_r,
													  uint8_t h_radius,
													  uint8_t v_radius,
													  disp_t disp_width,
													  disp_t disp_offset = 0) {


	condImgRef<T_L, T_R, dDir, 3> cir(img_l, img_r);

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0] or l_shape[2] != r_shape[2]) {
		return Multidim::Array<float, 2>(0,0);
	}

	Multidim::Array<float, 2> meanLeft = meanFilter2D(h_radius, v_radius, img_l);
	Multidim::Array<float, 2> meanRight = meanFilter2D(h_radius, v_radius, img_r);

	Multidim::Array<float, 2> sigmaLeft = sigmaFilter(h_radius, v_radius, meanLeft, img_l);
	Multidim::Array<float, 2> sigmaRight = sigmaFilter(h_radius, v_radius, meanRight, img_r);

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, 3> const& s_img = cir.source();
	Multidim::Array<float, 2> const& s_mean = (dDir == dispDirection::RightToLeft) ? meanRight : meanLeft;
	Multidim::Array<float, 2> const& s_sigma = (dDir == dispDirection::RightToLeft) ? sigmaRight : sigmaLeft;

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, 3> const& t_img = cir.target();
	Multidim::Array<float, 2> const& t_mean = (dDir == dispDirection::RightToLeft) ? meanLeft : meanRight;
	Multidim::Array<float, 2> const& t_sigma = (dDir == dispDirection::RightToLeft) ? sigmaLeft : sigmaRight;

	Multidim::Array<float, 3> cc = crossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
				typename condImgRef<T_L, T_R, dDir>::T_T,
				3,
				deltaSign,
				rmIncompleteRanges>(s_img,
									t_img,
									s_mean,
									t_mean,
									h_radius,
									v_radius,
									disp_width,
									disp_offset);


	Multidim::Array<float, 3> cv = normalizedCrossCorrelation<typename condImgRef<T_L, T_R, dDir>::T_S,
			typename condImgRef<T_L, T_R, dDir>::T_T,
			deltaSign,
			rmIncompleteRanges> (cc,
								 s_sigma,
								 t_sigma,
								 h_radius,
								 v_radius,
								 disp_width,
								 disp_offset);

	Multidim::Array<disp_t, 2> raw_disp = extractSelectedIndex<dispExtractionStartegy::Score, float>(cv);

	Multidim::Array<float, 3> tcv = truncatedCostVolume(cv, raw_disp, h_radius, v_radius, 1);

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = raw_disp.shape();

	Multidim::Array<float, 2> refined(shape);

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			float cm1 = tcv.value<Nc>(i,j,0);
			float c0 = tcv.value<Nc>(i,j,1);
			float c1 = tcv.value<Nc>(i,j,2);

			float delta = (cm1 - c1)/(2*(c1 - 2*c0 + cm1));

			disp_t d = raw_disp.value<Nc>(i,j);

			if (j + disp_offset + std::abs(deltaSign*d) + h_radius + 1 < t_img.shape()[1] and j + disp_offset + std::abs(deltaSign*d) - h_radius - 1 > 0 and
				i - v_radius > 0 and i + v_radius < t_img.shape()[0] and std::isfinite(cm1) and std::isfinite(c0) and std::isfinite(c1)) {

				float fm1 = 0;
				float f0 = 0;
				float f1 = 0;
				float sigmaSource = 0;

				disp_t dir = 1;

				if (delta > 0) {
					dir = -1;
				}

				for(int k = -v_radius; k <= v_radius; k++) {

					for (int l = -h_radius; l <= h_radius; l++) {

						for (int c = 0; c < l_shape[2]; c++) {

							float source = (s_img.template value<Nc>(i+k, j+l, c) - s_mean.value<Nc>(i,j) + s_img.template value<Nc>(i+k, j+l+dir, c) - s_mean.value<Nc>(i,j+dir))/2.;

							float targetm1 = t_img.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l - 1, c) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d - 1);
							float target0 = t_img.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l, c) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d);
							float target1 = t_img.template value<Nc>(i+k, j + disp_offset + deltaSign*d + l + 1, c) - t_mean.value<Nc>(i,j + disp_offset + deltaSign*d + 1);

							fm1 += source*targetm1;
							f0 += source*target0;
							f1 += source*target1;

							sigmaSource += source*source;
						}
					}
				}

				sigmaSource = sqrtf(sigmaSource);
				fm1 /= sigmaSource*t_sigma.template value<Nc>(i,j + disp_offset + deltaSign*d - 1);
				f0 /= sigmaSource*t_sigma.template value<Nc>(i,j + disp_offset + deltaSign*d);
				f1 /= sigmaSource*t_sigma.template value<Nc>(i,j + disp_offset + deltaSign*d + 1);

				float delta2 = (fm1 - f1)/(2*(f1 - 2*f0 + fm1)) - dir*0.5;

				if (std::fabs(delta2) < 1.) {
					delta = (delta + delta2)/2;
				}

			}

			refined.at<Nc>(i,j) = d + delta;

		}

	}

	return refined;

}

template<dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedNCCCostSymmetricFeatureVolumeDisp(Multidim::Array<float, 3> const& left_feature_volume,
																   Multidim::Array<float, 3> const& right_feature_volume,
																   disp_t disp_width) {

	auto l_shape = left_feature_volume.shape();
	auto r_shape = right_feature_volume.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	Multidim::Array<float, 2> mean_left = channelsMean(left_feature_volume);
	Multidim::Array<float, 2> mean_right = channelsMean(right_feature_volume);

	Multidim::Array<float, 2> sigma_left = channelsSigma(left_feature_volume, mean_left);
	Multidim::Array<float, 2> sigma_right = channelsSigma(right_feature_volume, mean_right);

	Multidim::Array<float, 3> CV = nccFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, sigma_left, sigma_right, disp_width);

	Multidim::Array<disp_t, 2> raw_disp = extractSelectedIndex<dispExtractionStartegy::Score>(CV);

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;
	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;


	condImgRef<float, float, dDir, 3> cfvr(left_feature_volume, right_feature_volume);
	condImgRef<float, float, dDir, 2> cmr(mean_left, mean_right);
	condImgRef<float, float, dDir, 2> csr(sigma_left, sigma_right);

	Multidim::Array<float, 3> const& source_feature_volume = cfvr.source();
	Multidim::Array<float, 3> const& target_feature_volume = cfvr.target();

	Multidim::Array<float, 2> const& source_mean = cmr.source();
	Multidim::Array<float, 2> const& target_mean = cmr.target();

	Multidim::Array<float, 2> const& target_sigma = csr.target();

	auto shape = raw_disp.shape();

	Multidim::Array<float, 2> refined(shape);

	#pragma omp parallel for
	for (int i = 0; i < shape[0]; i++) {

		for (int j = 0; j < shape[1]; j++) {

			disp_t d = raw_disp.value<Nc>(i,j);

			float delta = 0;

			if (j - 1 > 0 and j + 1 < shape[1] and d > 0 and d+1 < CV.shape()[2]) {

				float cm1 = CV.value<Nc>(i,j,d-1);
				float c0 = CV.value<Nc>(i,j,d);
				float c1 = CV.value<Nc>(i,j,d+1);

				delta = (cm1 - c1)/(2*(c1 - 2*c0 + cm1));

				if (j + deltaSign*d + 1 < shape[1] and j + deltaSign*d - 1 > 0) {

					float fm1 = 0;
					float f0 = 0;
					float f1 = 0;
					float sigmaSource = 0;

					disp_t dir = 1;

					if (delta > 0) {
						dir = -1;
					}

					for (int c = 0; c < source_feature_volume.shape()[2]; c++) {

						float source = (source_feature_volume.template value<Nc>(i, j, c) - source_mean.value<Nc>(i,j) +
										source_feature_volume.template value<Nc>(i, j+dir, c) - source_mean.value<Nc>(i,j+dir))/2.;

						float targetm1 = target_feature_volume.value<Nc>(i, j + deltaSign*d - 1, c) - target_mean.value<Nc>(i,j + deltaSign*d - 1);
						float target0 = target_feature_volume.value<Nc>(i, j + deltaSign*d, c) - target_mean.value<Nc>(i,j + deltaSign*d);
						float target1 = target_feature_volume.value<Nc>(i, j + deltaSign*d + 1, c) - target_mean.value<Nc>(i,j + deltaSign*d + 1);

						fm1 += source*targetm1;
						f0 += source*target0;
						f1 += source*target1;

						sigmaSource += source*source;
					}

					sigmaSource = sqrtf(sigmaSource);
					fm1 /= sigmaSource*target_sigma.value<Nc>(i,j + deltaSign*d - 1);
					f0 /= sigmaSource*target_sigma.value<Nc>(i,j + deltaSign*d);
					f1 /= sigmaSource*target_sigma.value<Nc>(i,j + deltaSign*d + 1);

					float delta2 = (fm1 - f1)/(2*(f1 - 2*f0 + fm1)) - dir*0.5;

					if (std::fabs(delta2) < 1.) {
						delta = (delta + delta2)/2;
					}

				}
			}

			refined.at<Nc>(i,j) = d + delta;

		}

	}

	return refined;
}


template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedNCCCostSymmetricUnfoldDisp(Multidim::Array<T_L, nImDim> const& img_l,
															Multidim::Array<T_R, nImDim> const& img_r,
															uint8_t h_radius,
															uint8_t v_radius,
															disp_t disp_width) {


	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 3>(0,0,0);
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 3>(0,0,0);
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(h_radius, v_radius, img_r);

	return refinedNCCCostSymmetricFeatureVolumeDisp<dDir>(left_feature_volume, right_feature_volume, disp_width);

}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedNCCCostSymmetricUnfoldDisp(Multidim::Array<T_L, nImDim> const& img_l,
															Multidim::Array<T_R, nImDim> const& img_r,
															UnFoldCompressor compressor,
															disp_t disp_width) {


	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(compressor, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(compressor, img_r);

	return refinedNCCCostSymmetricFeatureVolumeDisp<dDir>(left_feature_volume, right_feature_volume, disp_width);

}

template<dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refineFeatureVolumeNCCDisp(Multidim::Array<float, 3> const& feature_vol_l,
													 Multidim::Array<float, 3> const& feature_vol_r,
													 Multidim::Array<float, 2> const& mean_l,
													 Multidim::Array<float, 2> const& mean_r,
													 Multidim::Array<float, 2> const& sigma_l,
													 Multidim::Array<float, 2> const& sigma_r,
													 Multidim::Array<disp_t, 2> const& selectedIndex,
													 disp_t disp_width)
{

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	condImgRef<float, float, dDir, 3> cfvr(feature_vol_l, feature_vol_r);
	condImgRef<float, float, dDir, 2> cmr(mean_l, mean_r);
	condImgRef<float, float, dDir, 2> csr(sigma_l, sigma_r);

	Multidim::Array<float, 3> const& source_feature_volume = cfvr.source();
	Multidim::Array<float, 3> const& target_feature_volume = cfvr.target();

	Multidim::Array<float, 2> const& source_mean = cmr.source();
	Multidim::Array<float, 2> const& target_mean = cmr.target();

	Multidim::Array<float, 2> const& target_sigma = csr.target();

	auto d_shape = selectedIndex.shape();
	auto t_shape = target_feature_volume.shape();

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t d = selectedIndex.value<Nc>(i,j);

			int jd = j + deltaSign*d;

			if (j < 1 or j + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (jd < 1 or jd + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (d == 0 or d+1 >= disp_width) {
				refinedDisp.at<Nc>(i,j) = d;
			} else {

				float k_plus = 0;
				float k_minus = 0;

				for (int c = 0; c < t_shape[2]; c++) {

					float k0 = target_feature_volume.value<Nc>(i, jd-1, c) - target_mean.value<Nc>(i,jd-1);
					float k1 = target_feature_volume.value<Nc>(i, jd, c) - target_mean.value<Nc>(i,jd);
					float k2 = target_feature_volume.value<Nc>(i, jd+1, c) - target_mean.value<Nc>(i,jd+1);

					k_plus += k1*k2;
					k_minus += k0*k1;
				}

				float sigma_m1 = target_sigma.value<Nc>(i,jd-1)*target_sigma.value<Nc>(i,jd-1);
				float sigma_0 = target_sigma.value<Nc>(i,jd)*target_sigma.value<Nc>(i,jd);
				float sigma_1 = target_sigma.value<Nc>(i,jd+1)*target_sigma.value<Nc>(i,jd+1);

				float a1_plus = sigma_0 + sigma_1 - 2*k_plus;
				float a1_minus = sigma_m1 + sigma_0 - 2*k_minus;

				float a2_plus = 2*k_plus - 2*sigma_0;
				float a2_minus = 2*k_minus - 2*sigma_m1;

				float a3_plus = sigma_0;
				float a3_minus = sigma_m1;

				//float rho_m1 = cc.value<Nc>(i,j,d-1);
				//float rho_0 = cc.value<Nc>(i,j,d);
				//float rho_1 = cc.value<Nc>(i,j,d+1);

				float rho_m1 = 0;
				float rho_0 = 0;
				float rho_1 = 0;

				for (int c = 0; c < t_shape[2]; c++) {
					float detrend_s = source_feature_volume.value<Nc>(i,j,c) - source_mean.value<Nc>(i,j);
					float detrend_tm1 = target_feature_volume.value<Nc>(i,jd-1,c) - target_mean.value<Nc>(i,jd-1);
					float detrend_t0 = target_feature_volume.value<Nc>(i,jd,c) - target_mean.value<Nc>(i,jd);
					float detrend_t1 = target_feature_volume.value<Nc>(i,jd+1,c) - target_mean.value<Nc>(i,jd+1);

					rho_m1 += detrend_s*detrend_tm1;
					rho_0 += detrend_s*detrend_t0;
					rho_1 += detrend_s*detrend_t1;
				}

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

				float score = rho_0/sqrtf(sigma_0);

				float DeltaD = 0;

				if (DeltaD_plus > 0 and DeltaD_plus < 1) {

					float interpRho = (rho_1 - rho_0)*DeltaD_plus + rho_0;
					float interpSigma_t = a1_plus*DeltaD_plus*DeltaD_plus + a2_plus*DeltaD_plus + a3_plus;
					float tmpScore = interpRho/sqrtf(interpSigma_t);

					if (tmpScore > score) {
						score = tmpScore;
						DeltaD = DeltaD_plus;
					}

				}

				if (DeltaD_minus > 0 and DeltaD_minus < 1) {

					float interpRho = (rho_0 - rho_m1)*DeltaD_minus + rho_m1;
					float interpSigma_t = a1_minus*DeltaD_minus*DeltaD_minus + a2_minus*DeltaD_minus + a3_minus;
					float tmpScore = interpRho/sqrtf(interpSigma_t);

					if (tmpScore > score) {
						score = tmpScore;
						DeltaD = DeltaD_minus - 1;
					}

				}

				refinedDisp.at<Nc>(i,j) = d + DeltaD;

			}

		}
	}

	return refinedDisp;

}

template<int refineRadius = 1, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refineBarycentricSymmetricNCCDisp(Multidim::Array<float, 3> const& feature_vol_l,
															Multidim::Array<float, 3> const& feature_vol_r,
															Multidim::Array<float, 2> const& mean_l,
															Multidim::Array<float, 2> const& mean_r,
															Multidim::Array<disp_t, 2> const& selectedIndex,
															disp_t disp_width) {

	static_assert (refineRadius > 0, "refineBarycentricSymmetricNCCDisp cannot proceed with a refinement radius smaller than 1.");

	typedef Eigen::Matrix<float, Eigen::Dynamic, 2*refineRadius+1> TypeMatrixA;
	typedef Eigen::Matrix<float, Eigen::Dynamic, 2*refineRadius> TypeMatrixM;

	typedef Eigen::Matrix<float, 2*refineRadius, 1> TypeVectorAlpha;

	constexpr disp_t deltaSign = (dDir == dispDirection::RightToLeft) ? 1 : -1;
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	Multidim::Array<float, 3> const& source_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_r : feature_vol_l;
	Multidim::Array<float, 3> const& target_feature_volume = (dDir == dispDirection::RightToLeft) ? feature_vol_l : feature_vol_r;

	Multidim::Array<float, 2> const& source_mean = (dDir == dispDirection::RightToLeft) ?  mean_r : mean_l;
	Multidim::Array<float, 2> const& target_mean = (dDir == dispDirection::RightToLeft) ?  mean_l : mean_r;

	Multidim::Array<float, 3> source_zfeature_volume = zeromeanFeatureVolume(source_feature_volume, source_mean);
	Multidim::Array<float, 3> target_zfeature_volume = zeromeanFeatureVolume(target_feature_volume, target_mean);

	auto d_shape = selectedIndex.shape();
	auto t_shape = target_feature_volume.shape();

	Multidim::Array<float, 2> refinedDisp(d_shape);

	#pragma omp parallel for
	for (int i = 0; i < d_shape[0]; i++) {

		#pragma omp simd
		for (int j = 0; j < d_shape[1]; j++) {

			disp_t d = selectedIndex.value<Nc>(i,j);

			int jd = j + deltaSign*d;

			if (j < 1 or j + 1 >= d_shape[1]) { // if the source patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (jd < 1 + refineRadius or jd + 1 >= d_shape[1] - refineRadius) { // if the target patch is partially outside the image
				refinedDisp.at<Nc>(i,j) = d;
			} else if (d == 0 or d+1 >= disp_width) {
				refinedDisp.at<Nc>(i,j) = d;
			} else {

				Eigen::VectorXf source(t_shape[2]);

				for (int c = 0 ; c < t_shape[2]; c++) {
					source(c) = source_zfeature_volume.value<Nc>(i,j,c);
				}

				TypeMatrixA A(t_shape[2],2*refineRadius+1);

				for (int p = -refineRadius; p <= refineRadius; p++) {
					for (int c = 0 ; c < t_shape[2]; c++) {
						A(c, p+refineRadius) = target_zfeature_volume.value<Nc>(i,jd+p,c);
					}
				}

				Eigen::FullPivHouseholderQR<TypeMatrixA> QRA(A);

				Eigen::VectorXf fsPerp = A*QRA.solve(source);

				TypeMatrixM M(t_shape[2],2*refineRadius);

				for (int p = -refineRadius; p < refineRadius; p++) {
					for (int c = 0 ; c < t_shape[2]; c++) {
						M(c, p+refineRadius) = target_zfeature_volume.value<Nc>(i,jd+p,c) - target_zfeature_volume.value<Nc>(i,jd+refineRadius,c);
					}
				}

				Eigen::FullPivHouseholderQR<TypeMatrixM> QRM(M);

				Eigen::VectorXf ftPerp = M*QRM.solve(A.col(2*refineRadius));

				float g = (ftPerp.dot(ftPerp))/(ftPerp.dot(fsPerp));

				TypeVectorAlpha alpha = QRM.solve(g*fsPerp - A.col(2*refineRadius));

				float delta_d = refineRadius;
				for (int p = -refineRadius; p < refineRadius; p++) {
					delta_d += alpha(p)*float(p - refineRadius);
				}

				if (std::fabs(delta_d) < 1) { //subpixel adjustement is in the interval
					refinedDisp.at<Nc>(i,j) = d + delta_d;
				} else {
					refinedDisp.at<Nc>(i,j) = d;
				}
			}

		}

	}

	return refinedDisp;


}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedUnfoldNCCDisp(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   uint8_t h_radius,
												   uint8_t v_radius,
												   disp_t disp_width)
{
	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(h_radius, v_radius, img_r);

	Multidim::Array<float, 2> mean_left = channelsMean(left_feature_volume);
	Multidim::Array<float, 2> mean_right = channelsMean(right_feature_volume);

	Multidim::Array<float, 2> sigma_left = channelsSigma(left_feature_volume, mean_left);
	Multidim::Array<float, 2> sigma_right = channelsSigma(right_feature_volume, mean_right);

	Multidim::Array<float, 3> CV = nccFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, sigma_left, sigma_right, disp_width);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Score>(CV);

	return refineFeatureVolumeNCCDisp<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, sigma_left, sigma_right, disp, disp_width);
}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft>
Multidim::Array<float, 2> refinedUnfoldNCCDisp(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   UnFoldCompressor const& compressor,
												   disp_t disp_width)
{
	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(compressor, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(compressor, img_r);

	Multidim::Array<float, 2> mean_left = channelsMean(left_feature_volume);
	Multidim::Array<float, 2> mean_right = channelsMean(right_feature_volume);

	Multidim::Array<float, 2> sigma_left = channelsSigma(left_feature_volume, mean_left);
	Multidim::Array<float, 2> sigma_right = channelsSigma(right_feature_volume, mean_right);

	Multidim::Array<float, 3> CV = nccFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, sigma_left, sigma_right, disp_width);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Score>(CV);

	return refineFeatureVolumeNCCDisp<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, sigma_left, sigma_right, disp, disp_width);
}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, int refineRadius = 1>
Multidim::Array<float, 2> refinedBarycentricSymmetricNCCDisp(Multidim::Array<T_L, nImDim> const& img_l,
															 Multidim::Array<T_R, nImDim> const& img_r,
															 uint8_t h_radius,
															 uint8_t v_radius,
															 disp_t disp_width)
{

	static_assert (refineRadius > 0, "Barycentric symmetric refinement cannot run with a refinement radius smaller than 1 !");

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(h_radius, v_radius, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(h_radius, v_radius, img_r);

	Multidim::Array<float, 2> mean_left = channelsMean(left_feature_volume);
	Multidim::Array<float, 2> mean_right = channelsMean(right_feature_volume);

	Multidim::Array<float, 2> sigma_left = channelsSigma(left_feature_volume, mean_left);
	Multidim::Array<float, 2> sigma_right = channelsSigma(right_feature_volume, mean_right);

	Multidim::Array<float, 3> CV = nccFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, sigma_left, sigma_right, disp_width);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Score>(CV);

	return refineBarycentricSymmetricNCCDisp<refineRadius, dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, disp, disp_width);

}

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, int refineRadius = 1>
Multidim::Array<float, 2> refinedBarycentricSymmetricNCCDisp(Multidim::Array<T_L, nImDim> const& img_l,
															 Multidim::Array<T_R, nImDim> const& img_r,
															 UnFoldCompressor const& compressor,
															 disp_t disp_width)
{

	static_assert (refineRadius > 0, "Barycentric symmetric refinement cannot run with a refinement radius smaller than 1 !");

	auto l_shape = img_l.shape();
	auto r_shape = img_r.shape();

	if (l_shape[0] != r_shape[0]) {
		return Multidim::Array<float, 2>();
	}

	if (nImDim == 3) {
		if (l_shape[2] != r_shape[2]) {
			return Multidim::Array<float, 2>();
		}
	}

	Multidim::Array<float, 3> left_feature_volume = unfold(compressor, img_l);
	Multidim::Array<float, 3> right_feature_volume = unfold(compressor, img_r);

	Multidim::Array<float, 2> mean_left = channelsMean(left_feature_volume);
	Multidim::Array<float, 2> mean_right = channelsMean(right_feature_volume);

	Multidim::Array<float, 2> sigma_left = channelsSigma(left_feature_volume, mean_left);
	Multidim::Array<float, 2> sigma_right = channelsSigma(right_feature_volume, mean_right);

	Multidim::Array<float, 3> CV = nccFeatureVolume2CostVolume<dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, sigma_left, sigma_right, disp_width);

	Multidim::Array<disp_t, 2> disp = extractSelectedIndex<dispExtractionStartegy::Score>(CV);

	return refineBarycentricSymmetricNCCDisp<refineRadius, dDir>(left_feature_volume, right_feature_volume, mean_left, mean_right, disp, disp_width);

}

} // namespace Correlation
} // namespace StereoVision

#endif // STEREOVISION_CORRELATION_NCC_H
