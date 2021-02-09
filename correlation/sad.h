#ifndef SAD_H
#define SAD_H

#include "./correlation_base.h"

namespace StereoVision {
namespace Correlation {

inline float absDiff(float p1, float p2) {
	return std::fabs(p1-p2);
}

template<class T_S, class T_T, disp_t deltaSign = 1, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> ZSADCostVolume(Multidim::Array<T_S, 2> const& img_s,
										 Multidim::Array<T_T, 2> const& img_t,
										 Multidim::Array<float, 2> const& s_mean,
										 Multidim::Array<float, 2> const& t_mean,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	return buildZeroMeanCostVolume<T_S,
			T_T,
			absDiff,
			dispExtractionStartegy::Cost,
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


template<class T_L, class T_R, dispDirection dDir = dispDirection::RightToLeft, bool rmIncompleteRanges = false>
Multidim::Array<float, 3> ZSADCostVolume(Multidim::Array<T_L, 2> const& img_l,
										 Multidim::Array<T_R, 2> const& img_r,
										 uint8_t h_radius,
										 uint8_t v_radius,
										 disp_t disp_width,
										 disp_t disp_offset = 0) {

	return buildZeroMeanCostVolume<T_L,
			T_R,
			absDiff,
			dispExtractionStartegy::Cost,
			dDir,
			rmIncompleteRanges>(img_l,
								img_r,
								h_radius,
								v_radius,
								disp_width,
								disp_offset);

}

} //namespace Correlation
} //namespace StereoVision

#endif // SAD_H
