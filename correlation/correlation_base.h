#ifndef STEREOVISION_CORRELATION_BASE_H
#define STEREOVISION_CORRELATION_BASE_H

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

enum class truncatedCostVolumeDirection{
	Same = 0,
	Reversed = 1,
	Both = 2
};

typedef int32_t disp_t;

static_assert (std::is_integral<disp_t>::value, "The typedef for disp_t should be an integer type !");
static_assert (std::is_signed<disp_t>::value, "The typedef for disp_t should be a signed integer !");

template<int nDim>
class searchOffset{
public:

	searchOffset() :
		_isValid(false)
	{
		std::fill(_upperOffsets.begin(), _upperOffsets.end(), 0);
		std::fill(_lowerOffsets.begin(), _lowerOffsets.end(), 0);
	}

	template<typename... Ds>
	searchOffset(disp_t lowerOffset0, disp_t upperOffset0, Ds... nextOffsets) :
		_isValid(true)
	{
		static_assert(sizeof...(nextOffsets) == 2*(nDim-1),
				"The number of offsets provided to the constructor should be twice the number of dimensions !");

		std::array<disp_t, 2*(nDim-1)> nOffsets({nextOffsets...});

		_lowerOffsets[0] = lowerOffset0;
		_upperOffsets[0] = upperOffset0;

		for (int i = 1; i < nDim; i++) {
			_lowerOffsets[i] = nOffsets[2*(i-1)];
			_upperOffsets[i] = nOffsets[2*(i-1)+1];
		}
	}

	bool isValid() const {
		return _isValid;
	}

	template<int dim>
	int& upperOffset() {
		return _upperOffsets[dim];
	}

	template<int dim>
	int& lowerOffset() {
		return _lowerOffsets[dim];
	}

	template<int dim>
	int const& upperOffset() const {
		return _upperOffsets[dim];
	}

	template<int dim>
	int const& lowerOffset() const {
		return _lowerOffsets[dim];
	}

	int& upperOffset(int dim) {
		return _upperOffsets[dim];
	}

	int& lowerOffset(int dim) {
		return _lowerOffsets[dim];
	}

	int const& upperOffset(int dim) const {
		return _upperOffsets[dim];
	}

	int const& lowerOffset(int dim) const {
		return _lowerOffsets[dim];
	}



	template<int dim>
	int dimRange() const {
		return _upperOffsets[dim] - _lowerOffsets[dim] + 1;
	}

	int dimRange(int dim) const {
		return _upperOffsets[dim] - _lowerOffsets[dim] + 1;
	}

	template<int dim>
	inline bool valueInRange(int val) const {
		return val >= _lowerOffsets[dim] and val <= _upperOffsets[dim];
	}

	inline bool valueInRange(int val, int dim) const {
		return val >= _lowerOffsets[dim] and val <= _upperOffsets[dim];
	}

	template<int dim>
	inline int setValueInRange(int val) const {
		return std::abs(val%dimRange<dim>())+lowerOffset<dim>();
	}

	inline int setValueInRange(int val, int dim) const {
		return std::abs(val%dimRange(dim))+lowerOffset(dim);
	}

	template<int dim>
	int idx2disp(int idx) const{
		return _lowerOffsets[dim] + idx;
	}

	int idx2disp(int dim, int idx) const{
		return _lowerOffsets[dim] + idx;
	}

	template<int dim>
	int disp2idx(int disp) const{
		return disp - _lowerOffsets[dim];
	}

	int disp2idx(int dim, int disp) const{
		return disp - _lowerOffsets[dim];
	}

private:
	std::array<disp_t,nDim> _upperOffsets;
	std::array<disp_t,nDim> _lowerOffsets;

	bool _isValid;
};

template<typename SearchRangeType>
struct searchRangeTypeInfos {

};

template<>
struct searchRangeTypeInfos<disp_t> {
	static const int CostVolumeDims = 3;
};

template<int nDim>
struct searchRangeTypeInfos<searchOffset<nDim> > {
	static const int CostVolumeDims = 2*nDim;
};


template<dispExtractionStartegy strategy, class T_CV>
Multidim::Array<disp_t, 2> extractSelectedIndex(Multidim::Array<T_CV, 3> const& costVolume) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto cv_shape = costVolume.shape();

	Multidim::Array<disp_t, 2> disp(cv_shape[0], cv_shape[1]);

	#pragma omp parallel for
	for (int i = 0; i < cv_shape[0]; i++) {
		#pragma omp simd
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

template<dispExtractionStartegy strategy, class T_CV>
Multidim::Array<disp_t, 3> extractSelected2dIndex(Multidim::Array<T_CV, 4> const& costVolume) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto cv_shape = costVolume.shape();

	Multidim::Array<disp_t, 3> disp(cv_shape[0], cv_shape[1], 2);

	#pragma omp parallel for
	for (int i = 0; i < cv_shape[0]; i++) {
		#pragma omp simd
		for (int j = 0; j < cv_shape[1]; j++) {

			T_CV selectedScore = costVolume.template value<Nc>(i,j,0,0);
			disp_t selectedDisp1 = 0;
			disp_t selectedDisp2 = 0;

			for (uint32_t d1 = 0; d1 < cv_shape[2]; d1++) {
				for (uint32_t d2 = 0; d2 < cv_shape[3]; d2++) {
					if (strategy == dispExtractionStartegy::Cost) {
						if (costVolume.template value<Nc>(i,j,d1,d2) <= selectedScore) {
							selectedScore = costVolume.template value<Nc>(i,j,d1,d2);
							selectedDisp1 = d1;
							selectedDisp2 = d2;
						}
					} else {
						if (costVolume.template value<Nc>(i,j,d1,d2) >= selectedScore) {
							selectedScore = costVolume.template value<Nc>(i,j,d1,d2);
							selectedDisp1 = d1;
							selectedDisp2 = d2;
						}
					}
				}
			}

			disp.at<Nc>(i,j,0) = selectedDisp1;
			disp.at<Nc>(i,j,1) = selectedDisp2;

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

	#pragma omp parallel for
	for(int i = 0; i < shape[0]; i++) {
		#pragma omp simd
		for (int j = 0; j < shape[1]; j++) {
			disp.template at<Nc>(i,j) = deltaSign*selectedIndex.template value<Nc>(i,j) + disp_offset;
		}
	}

	return disp;

}

template<typename DT>
Multidim::Array<DT, 3> selected2dIndexToDisp(Multidim::Array<DT, 3> const& selectedIndex,
											 searchOffset<2> const& offset) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = selectedIndex.shape();

	Multidim::Array<DT, 3> disp(shape[0], shape[1], 2);

	#pragma omp parallel for
	for(int i = 0; i < shape[0]; i++) {
		#pragma omp simd
		for (int j = 0; j < shape[1]; j++) {
			disp.template at<Nc>(i,j,0) = selectedIndex.template value<Nc>(i,j, 0) + offset.lowerOffset(0);
			disp.template at<Nc>(i,j,1) = selectedIndex.template value<Nc>(i,j, 1) + offset.lowerOffset(1);
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
		#pragma omp simd
		for (int j = 0; j < cv_shape[1]; j++) {
			uint32_t p = selectedIndex.value<Nc>(i,j);
			tcv.template at<Nc>(i,j) = costVolume.value<Nc>(i,j,p);
		}
	}

	return tcv;
}

template<class T_CV,
		 dispDirection dir = dispDirection::RightToLeft,
		 truncatedCostVolumeDirection sdir = truncatedCostVolumeDirection::Same>
Multidim::Array<T_CV, 3> truncatedCostVolume(Multidim::Array<T_CV, 3> const& costVolume,
											 Multidim::Array<disp_t, 2> const& selectedIndex,
											 uint8_t h_radius,
											 uint8_t v_radius,
											 uint8_t cost_vol_radius) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto cv_shape = costVolume.shape();

	Multidim::Array<T_CV, 3> tcv(cv_shape[0],
								cv_shape[1],
								(sdir == truncatedCostVolumeDirection::Both) ? cost_vol_radius*4+1 : cost_vol_radius*2+1);

	#pragma omp parallel for
	for (int i = 0; i < cv_shape[0]; i++) {
		#pragma omp simd
		for (int j = 0; j < cv_shape[1]; j++) {

			if (sdir == truncatedCostVolumeDirection::Same) {

				for (int32_t d = 0; d <= 2*cost_vol_radius; d++) {
					int32_t p = selectedIndex.value<Nc>(i,j)+d-cost_vol_radius;

					if (p < 0 or p >= cv_shape[2]
							or j < h_radius or j+p+h_radius >= cv_shape[1]
							or i < v_radius or i+v_radius >= cv_shape[0]) {
						tcv.template at<Nc>(i,j,d) = std::nanf("");
					} else {
						tcv.template at<Nc>(i,j,d) = costVolume.template value<Nc>(i,j,p);
					}
				}

			} else if (sdir == truncatedCostVolumeDirection::Reversed) {

				for (int32_t d = 0; d <= 2*cost_vol_radius; d++) {
					int32_t sgn = (dir == dispDirection::RightToLeft) ? -1 : 1;

					int32_t p = selectedIndex.value<Nc>(i,j)+d-cost_vol_radius;
					int32_t jp = j+sgn*(d-cost_vol_radius);

					if (p < 0 or p >= cv_shape[2]
							or std::min(jp, j) < h_radius or std::max(jp, j) + h_radius >= cv_shape[1]
							or i < v_radius or i+v_radius >= cv_shape[0]) {
						tcv.template at<Nc>(i,j,d) = std::nanf("");
					} else {
						tcv.template at<Nc>(i,j,d) = costVolume.template value<Nc>(i,jp,p);
					}
				}

			} else {

				for (int32_t d = 0; d <= 2*cost_vol_radius; d++) {
					int32_t sgn = (dir == dispDirection::RightToLeft) ? -1 : 1;

					int32_t p = selectedIndex.value<Nc>(i,j)+d-cost_vol_radius;
					int32_t jp = j+sgn*(d-cost_vol_radius);

					int32_t d_d = 2*d;
					int32_t d_r = 2*d+1;

					if (d == cost_vol_radius) {
						jp = -1;
					}

					if (d > cost_vol_radius) {
						d_d -= 1;
						d_r -= 1;
					}

					if (p < 0 or p >= cv_shape[2]
							or j < h_radius or j+p+h_radius >= cv_shape[1]
							or i < v_radius or i+v_radius >= cv_shape[0]) {
						tcv.template at<Nc>(i,j,d_d) = std::nanf("");
					} else {
						tcv.template at<Nc>(i,j,d_d) = costVolume.template value<Nc>(i,j,p);
					}

					if (p < 0 or p >= cv_shape[2]
							or std::min(jp, j) < h_radius or std::max(jp, j) + h_radius >= cv_shape[1]
							or i < v_radius or i+v_radius >= cv_shape[0]) {
						tcv.template at<Nc>(i,j,d_r) = std::nanf("");
					} else {
						tcv.template at<Nc>(i,j,d_r) = costVolume.template value<Nc>(i,jp,p);
					}
				}
			}
		}
	}

	return tcv;

}


template<class T_CV,
		 dispDirection dir = dispDirection::RightToLeft>
Multidim::Array<T_CV, 4> truncatedBidirectionaCostVolume(Multidim::Array<T_CV, 4> const& costVolume,
														 Multidim::Array<disp_t, 3> const& selectedIndex,
														 uint8_t cost_vol_radius0 = -1,
														 uint8_t cost_vol_radius1 = -1)
{

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	uint8_t cv_radius0 = cost_vol_radius0;
	uint8_t cv_radius1 = cost_vol_radius1;

	if (cv_radius0 < 1) {
		cv_radius0 = 1;
	}

	if (cv_radius1 < 1) {
		cv_radius1 = cv_radius0;
	}

	auto cv_shape = costVolume.shape();

	Multidim::Array<T_CV, 4> tcv(cv_shape[0],
								cv_shape[1],
								cv_radius0*2+1,
								cv_radius1*2+1);

	#pragma omp parallel for
	for (int i = 0; i < cv_shape[0]; i++) {
		#pragma omp simd
		for (int j = 0; j < cv_shape[1]; j++) {

			for (int32_t d0 = 0; d0 <= 2*cv_radius0; d0++) {

				int32_t p0 = selectedIndex.value<Nc>(i,j,0)+d0-cv_radius0;

				for (int32_t d1 = 0; d1 <= 2*cv_radius1; d1++) {

					int32_t p1 = selectedIndex.value<Nc>(i,j,1)+d1-cv_radius1;

					tcv.template at<Nc>(i,j,d0,d1) = costVolume.valueOrAlt({i,j,p0,p1},std::nanf(""));
				}
			}
		}
	}

	return tcv;
}

template<class T_IB,
		 dispDirection dir = dispDirection::RightToLeft,
		 truncatedCostVolumeDirection sdir = truncatedCostVolumeDirection::Same>
Multidim::Array<T_IB, 3> extractInBoundDomain(Multidim::Array<disp_t, 2> const& selectedIndex,
											  uint32_t width,
											  uint8_t h_radius,
											  uint8_t v_radius,
											  uint8_t cost_vol_radius) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto im_shape = selectedIndex.shape();

	Multidim::Array<T_IB, 3> ib(im_shape[0],
								im_shape[1],
								(sdir == truncatedCostVolumeDirection::Both) ? cost_vol_radius*4+1 : cost_vol_radius*2+1);

	#pragma omp parallel for
	for (int i = 0; i < im_shape[0]; i++) {
		#pragma omp simd
		for (int j = 0; j < im_shape[1]; j++) {

			if (sdir == truncatedCostVolumeDirection::Same) {
				for (int32_t d = -cost_vol_radius; d <= cost_vol_radius; d++) {
					int32_t p = selectedIndex.value<Nc>(i,j)+d;

					if (p < 0 or p >= static_cast<int32_t>(width)
							or j < h_radius or j+p+h_radius >= im_shape[1]
							or i < v_radius or i+v_radius >= im_shape[0]) {
						ib.template at<Nc>(i,j,d+cost_vol_radius) = 0;
					} else {
						ib.template at<Nc>(i,j,d+cost_vol_radius) = 1;
					}
				}
			} else if (sdir == truncatedCostVolumeDirection::Reversed) {

				for (int32_t d = -cost_vol_radius; d <= cost_vol_radius; d++) {

					int32_t sgn = (dir == dispDirection::RightToLeft) ? -1 : 1;

					int32_t p = selectedIndex.value<Nc>(i,j)+d;
					int32_t jp = j+sgn*d;

					if (p < 0 or p >= static_cast<int32_t>(width)
							or std::min(jp, j) < h_radius or std::max(jp, j) + h_radius >= im_shape[1]
							or i < v_radius or i+v_radius >= im_shape[0]) {
						ib.template at<Nc>(i,j,d+cost_vol_radius) = 0;
					} else {
						ib.template at<Nc>(i,j,d+cost_vol_radius) = 1;
					}
				}
			} else {

				for (int32_t d = -cost_vol_radius; d <= cost_vol_radius; d++) {

					int32_t sgn = (dir == dispDirection::RightToLeft) ? -1 : 1;

					int32_t p = selectedIndex.value<Nc>(i,j)+d;
					int32_t jp = j+sgn*d;

					int32_t d_d = 2*(d+cost_vol_radius);
					int32_t d_r = 2*(d+cost_vol_radius)+1;

					if (d == 0) {
						jp = -1;
					}

					if (d > 0) {
						d_d -= 1;
						d_r -= 1;
					}

					if (p < 0 or p >= static_cast<int32_t>(width)
							or std::min(jp, j) < h_radius or std::max(jp, j) + h_radius >= im_shape[1]
							or i < v_radius or i+v_radius >= im_shape[0]) {
						ib.template at<Nc>(i,j,d_r) = 0;
					} else {
						ib.template at<Nc>(i,j,d_r) = 1;
					}

					if (p < 0 or p >= static_cast<int32_t>(width)
							or j < h_radius or j+p+h_radius >= im_shape[1]
							or i < v_radius or i+v_radius >= im_shape[0]) {
						ib.template at<Nc>(i,j,d_d) = 0;
					} else {
						ib.template at<Nc>(i,j,d_d) = 1;
					}
				}

			}
		}
	}

	return ib;

}

template<class T_L, class T_R, dispDirection dDir, int nImDims = 2>
class condImgRef {
};


template<class T_L, class T_R, int nImDims>
class condImgRef<T_L, T_R, dispDirection::RightToLeft, nImDims> {
public:

	typedef T_R T_S;
	typedef T_L T_T;

	explicit condImgRef(Multidim::Array<T_L, nImDims> const& im_l, Multidim::Array<T_R, nImDims> const& im_r) :
		img_l(im_l),
		img_r(im_r) {
	}

	Multidim::Array<T_S, nImDims> const& source() const {
		return img_r;
	}

	Multidim::Array<T_T, nImDims> const& target() const {
		return img_l;
	}

private:
	Multidim::Array<T_L, nImDims> const& img_l;
	Multidim::Array<T_R, nImDims> const& img_r;
};


template<class T_L, class T_R, int nImDims>
class condImgRef<T_L, T_R, dispDirection::LeftToRight, nImDims> {
public:

	typedef T_L T_S;
	typedef T_R T_T;

	explicit condImgRef(Multidim::Array<T_L, nImDims> const& im_l, Multidim::Array<T_R, nImDims> const& im_r) :
		img_l(im_l),
		img_r(im_r) {
	}

	Multidim::Array<T_S, nImDims> const& source() const {
		return img_l;
	}

	Multidim::Array<T_T, nImDims> const& target() const {
		return img_r;
	}

private:
	Multidim::Array<T_L, nImDims> const& img_l;
	Multidim::Array<T_R, nImDims> const& img_r;
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

	#pragma omp parallel for
	for(long j = h_radius; j < shape[1]-h_radius; j++){
		#pragma omp simd
		for(long i = v_radius; i < shape[0]-v_radius; i++) {
			mean.at<Nc>(i, j) /= box_size;
		}
	}// mean filter computed

	return mean;

}


template<class T_I>
Multidim::Array<float, 2> meanFilter2D (uint8_t h_radius,
										uint8_t v_radius,
										Multidim::Array<T_I, 3> const& in_data) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = in_data.shape();

	size_t box_size = (2*v_radius + 1)*(2*h_radius + 1)*shape[2];

	Multidim::Array<float, 2> mean(in_data.shape()[0], in_data.shape()[1]);
	Multidim::Array<float, 2> mHorizontal(in_data.shape()[0], in_data.shape()[1]);

	//horizontal mean filters
	#pragma omp parallel for
	for(long i = 0; i < shape[0]; i++){

		mHorizontal.at<Nc>(i, h_radius) = 0;

		for(long j = 0; j <= 2*h_radius; j++){

			for (int c = 0; c < shape[2]; c++) {
				mHorizontal.at<Nc>(i, h_radius) += in_data.template value<Nc>(i,j,c);
			}
		}

		for(long j = h_radius+1; j < shape[1]-h_radius; j++) {

			mHorizontal.at<Nc>(i, j) = mHorizontal.at<Nc>(i, j-1);

			for (int c = 0; c < shape[2]; c++) {
				mHorizontal.at<Nc>(i, j) += in_data.template value<Nc>(i, j + h_radius, c) - in_data.template value<Nc>(i, j - h_radius -1, c);
			}
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

	#pragma omp parallel for
	for(long j = h_radius; j < shape[1]-h_radius; j++){
		#pragma omp simd
		for(long i = v_radius; i < shape[0]-v_radius; i++) {
			mean.at<Nc>(i, j) /= box_size;
		}
	}// mean filter computed

	return mean;

}

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

template<class T_I, class T_O = float>
inline Multidim::Array<T_O, 2> channelsMean (Multidim::Array<T_I, 3> const& in_data) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = in_data.shape()[0];
	int w = in_data.shape()[1];
	int f = in_data.shape()[2];

	Multidim::Array<T_O, 2> mean(h, w);

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {

                        mean.template at<Nc>(i,j) = 0;

			for (int c = 0; c < f; c++) {
                                mean.template at<Nc>(i,j) += static_cast<T_O>(in_data.template value<Nc>(i,j,c));
			}
		}
	}

	float scale = 1./static_cast<float>(f);

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {
                        mean.template at<Nc>(i,j) *= scale;
		}
	}

	return mean;

}

template<>
inline Multidim::Array<uint8_t, 2> channelsMean<uint8_t, uint8_t> (Multidim::Array<uint8_t, 3> const& in_data) {
	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	int h = in_data.shape()[0];
	int w = in_data.shape()[1];
	int f = in_data.shape()[2];

	Multidim::Array<uint8_t, 2> mean(h, w);

	#pragma omp parallel for
	for(int i = 0; i < h; i++) {
		#pragma omp simd
		for(int j = 0; j < w; j++) {

			uint16_t acc = 0;

			for (int c = 0; c < f; c++) {
				acc += in_data.value<Nc>(i,j,c);
			}

			mean.at<Nc>(i,j) = acc/f;
		}
	}

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

} // namespace Correlation
} // namespace StereoVision

#endif // CORRELATION_BASE_H
