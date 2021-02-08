#ifndef CORRELATION_BASE_H
#define CORRELATION_BASE_H

#include <MultidimArrays/MultidimArrays.h>
#include <cmath>

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

} // namespace Correlation
} // namespace StereoVision

#endif // CORRELATION_BASE_H
