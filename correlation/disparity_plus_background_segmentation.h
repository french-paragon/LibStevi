#ifndef STEREOVISION_DISPARITY_PLUS_BACKGROUND_SEGMENTATION_H
#define STEREOVISION_DISPARITY_PLUS_BACKGROUND_SEGMENTATION_H

#include "correlation_base.h"
#include "cross_correlations.h"
#include "on_demand_cost_volume.h"

namespace StereoVision {
namespace Correlation {

struct StereoDispWithBgMask {

	enum MaskInfo {
		Foreground = 1,
		Background = 0
	};

	StereoDispWithBgMask() :
		fg_mask(),
		disp()
	{

	}

	StereoDispWithBgMask(int h, int w) :
		fg_mask(h, w),
		disp(h, w)
	{

	}

	StereoDispWithBgMask(DispWithBgMask && other) :
		fg_mask(other.fg_mask),
		disp(other.disp)
	{

	}

	Multidim::Array<MaskInfo, 2> fg_mask;
	Multidim::Array<disp_t, 2> disp;
};

template<matchingFunctions matchFunc, class T_CV, class T_FV>
class DisparityEstimatorWithBackgroundRemoval {
	using T_F = typename MatchingFuncComputeTypeInfos<matchFunc, T_FV>::FeatureType;

	template<Multidim::ArrayDataAccessConstness constnessS,
			 Multidim::ArrayDataAccessConstness constnessT>
	using OnDemandCVT = OnDemandStereoCostVolume<matchFunc, T_CV, T_FV, T_FV, constnessS, constnessT>;

	DisparityEstimatorWithBackgroundRemoval(float relative_tolerance = 0.8, disp_t disp_tol = 2) :
		_rel_tol(relative_tolerance),
		_disp_tol(disp_tol)
	{

	}

	template<Multidim::ArrayDataAccessConstness constnessS,
			 Multidim::ArrayDataAccessConstness constnessT>
	bool computeBackgroundDisp(Multidim::Array<T_FV, 3, constnessS> const& source_f,
							   Multidim::Array<T_FV, 3, constnessT> const& target_f,
							   searchOffset<1> const search_offset) {

		_searchOffset = search_offset;

		if (!_searchOffset.isValid()) {
			//invalid search offset
			return false;
		}

		_source_bg_features = getFeatureVolumeForMatchFunc<matchFunc, T_FV, constnessT, T_F>(source_f);
		_target_bg_features = getFeatureVolumeForMatchFunc<matchFunc, T_FV, constnessT, T_F>(target_f);

		_bg_cost_volume = featureVolume2CostVolume<matchFunc, float, float, disp_t, dispDirection::RightToLeft, float>
				(_target_bg_features, _source_bg_features, _searchOffset);

		auto bg_index = extractSelectedIndex(_bg_cost_volume);

		_bg_disp_idx = selectedIndexToDisp(bg_index, _searchOffset.lowerOffset<0>());
	}

	template<Multidim::ArrayDataAccessConstness constnessS,
			 Multidim::ArrayDataAccessConstness constnessT>
	StereoDispWithBgMask computeDispAndForegroundMask(OnDemandCVT<constnessS, constnessT> const& on_demand_cv) {

		if (!_searchOffset.isValid()) {
			//invalid search offset
			return StereoDispWithBgMask();
		}

		if (_source_bg_features.empty() or _target_bg_features.empty() or _bg_cost_volume.empty() or _bg_disp_idx.empty()) {
			//background not computed yet
			return StereoDispWithBgMask();
		}

		if (on_demand_cv.shape() != _bg_cost_volume.shape()) {
			//invalid cost volume shape
			return StereoDispWithBgMask();
		}

		auto shape = _bg_cost_volume.shape();

		StereoDispWithBgMask ret(shape[0], shape[1]);

		#pragma omp parallel for
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {

				disp_t idx_bg = _bg_disp_idx.valueUnchecked(i,j);
				T_CV cost_bg = _bg_cost_volume.valueUnchecked(i,j,idx_bg);

				auto opt_fg_cost = on_demand_cv.costValue(i,j,idx_bg);

				if (!opt_fg_cost.has_value()) {
					ret.disp.atUnchecked(i,j) = _searchOffset.idx2disp<0>(idx_bg);
					ret.fg_mask.atUnchecked(i,j) = StereoDispWithBgMask::Background;
					continue;
				}

				T_CV cost_fg = opt_fg_cost.value();
				disp_t idx_fg = idx_bg;

				if (std::min(cost_bg, cost_fg)/std::max(cost_bg, cost_fg) < _rel_tol) {
					for (int d = 0; d < shape[2]; d++) {
						auto opt_fg_cost = on_demand_cv.costValue(i,j,d);
						if (!opt_fg_cost.has_value()) {
							continue;
						}

						T_CV cost_fg_cand = opt_fg_cost.value();
						std::tie(idx_fg, cost_fg) = optimalDispAndCost<matchFunc>(idx_fg,
																   cost_fg,
																   d,
																   cost_fg_cand);
					}
				}

				ret.disp.atUnchecked(i,j) = _searchOffset.idx2disp<0>(idx_fg);
				ret.fg_mask.atUnchecked(i,j) = (std::abs(idx_fg - idx_bg) >= _disp_tol) ? StereoDispWithBgMask::Foreground : StereoDispWithBgMask::Background;

			}
		}

		return ret;

	}

protected:

	float _rel_tol;
	disp_t _disp_tol;

	searchOffset<1> _searchOffset;

	Multidim::Array<T_F, 3> _source_bg_features;
	Multidim::Array<T_F, 3> _target_bg_features;

	Multidim::Array<T_CV, 3> _bg_cost_volume;
	Multidim::Array<disp_t, 2> _bg_disp_idx;

};

}
}

#endif // STEREOVISION_DISPARITY_PLUS_BACKGROUND_SEGMENTATION_H
