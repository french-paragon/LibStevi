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
#include "./unfold.h"

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

template<class T_I>
Multidim::Array<float, 2> channelsSigma (Multidim::Array<T_I, 3> const& in_data,
									   Multidim::Array<float, 2> const& mean) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = in_data.shape()[0];
	int w = in_data.shape()[1];
	int f = in_data.shape()[2];

	Multidim::Array<float, 2> std(h, w);

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {

			std.at<Nc>(i,j) = 0;

			for (int c = 0; c < f; c++) {
				float tmp = static_cast<float>(in_data.template value<Nc>(i,j,c)) - mean.value<Nc>(i,j);
				std.at<Nc>(i,j) += tmp*tmp;
			}
		}
	}

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {
			std.at<Nc>(i,j) = sqrtf(std.value<Nc>(i,j));
		}
	}

	return std;
}

template<class T_I>
Multidim::Array<float, 2> channelsSigma (Multidim::Array<T_I, 3> const& in_data) {

	Multidim::Array<float, 2> mean = channelsMean(in_data);

	return channelsSigma(in_data, mean);

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

template<class T_L, class T_R, int nImDim = 2, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> nccUnfoldBasedCostVolume(Multidim::Array<T_L, nImDim> const& img_l,
												   Multidim::Array<T_R, nImDim> const& img_r,
												   uint8_t h_radius,
												   uint8_t v_radius,
												   disp_t disp_width)
{
	condImgRef<T_L, T_R, dDir, nImDim> cir(img_l, img_r);

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;
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

	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_S, nImDim> const& s_img = cir.source();
	Multidim::Array<typename condImgRef<T_L, T_R, dDir>::T_T, nImDim> const& t_img = cir.target();

	Multidim::Array<float, 3> source_feature_volume = unfold(h_radius, v_radius, s_img);
	Multidim::Array<float, 3> target_feature_volume = unfold(h_radius, v_radius, t_img);

	int h = source_feature_volume.shape()[0];
	int w = source_feature_volume.shape()[1];
	int f = source_feature_volume.shape()[2];

	Multidim::Array<float, 2> source_mean = channelsMean(source_feature_volume);
	Multidim::Array<float, 2> target_mean = channelsMean(target_feature_volume);

	Multidim::Array<float, 2> source_sigma = channelsSigma(source_feature_volume, source_mean);
	Multidim::Array<float, 2> target_sigma = channelsSigma(target_feature_volume, target_mean);

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {
			for (int c = 0; c < f; c++) {
				source_feature_volume.at<Nc>(i,j,c) = (source_feature_volume.value<Nc>(i,j,c) - source_mean.value<Nc>(i,j))/source_sigma.value<Nc>(i,j);
				target_feature_volume.at<Nc>(i,j,c) = (target_feature_volume.value<Nc>(i,j,c) - target_mean.value<Nc>(i,j))/target_sigma.value<Nc>(i,j);
			}
		}
	}

	Multidim::Array<float, 3> costVolume(h,w,disp_width);

	#pragma omp parallel for
	for (int i = 0; i < h; i++) {
		#pragma omp simd
		for (int j = 0; j < w; j++) {

			for (int d = 0; d < disp_width; d++) {

				costVolume.at<Nc>(i,j,d) = 0.0;

				for (int c = 0; c < f; c++) {
					float s = source_feature_volume.value<Nc>(i,j,c);
					float t = target_feature_volume.valueOrAlt({i,j+deltaSign*d,c}, 0);
					costVolume.at<Nc>(i,j,d) += s*t;
				}
			}
		}
	}

	return costVolume;
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

} // namespace Correlation
} // namespace StereoVision

#endif // STEREOVISION_CORRELATION_NCC_H
