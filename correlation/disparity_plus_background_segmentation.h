#ifndef STEREOVISION_DISPARITY_PLUS_BACKGROUND_SEGMENTATION_H
#define STEREOVISION_DISPARITY_PLUS_BACKGROUND_SEGMENTATION_H

#include "correlation_base.h"
#include "cross_correlations.h"
#include "on_demand_cost_volume.h"

#include <set>
#include <queue>

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

	StereoDispWithBgMask(StereoDispWithBgMask && other) :
		fg_mask(other.fg_mask),
		disp(other.disp)
	{

	}

	Multidim::Array<MaskInfo, 2> fg_mask;
	Multidim::Array<disp_t, 2> disp;
};

template<matchingFunctions matchFunc, class T_CV, class T_FV>
class DisparityEstimatorWithBackgroundRemoval {

public:
	using T_F = typename MatchingFuncComputeTypeInfos<matchFunc, T_FV>::FeatureType;

	template<Multidim::ArrayDataAccessConstness constnessS,
			 Multidim::ArrayDataAccessConstness constnessT>
	using OnDemandCVT = OnDemandStereoCostVolume<matchFunc, T_CV, T_FV, T_FV, constnessS, constnessT>;

	DisparityEstimatorWithBackgroundRemoval(float relative_threshold = 0.8, disp_t disp_tol = 2) :
		_rel_threshold(relative_threshold),
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

		_bg_cost_volume = featureVolume2CostVolume<matchFunc, float, float, searchOffset<1>, dispDirection::RightToLeft, float>
				(_target_bg_features, _source_bg_features, _searchOffset);

		auto bg_index = extractSelectedIndex<MatchingFunctionTraits<matchFunc>::extractionStrategy>(_bg_cost_volume);

		_bg_disp_idx = selectedIndexToDisp(bg_index, _searchOffset.lowerOffset<0>());

		return true;
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

		std::set<std::tuple<int, int>> checkedPixels;

		#pragma omp parallel for
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {

				std::queue<std::tuple<int, int>> pixels2check;

				pixels2check.push({i,j});
				bool first = true;

				while (!pixels2check.empty()) {

					bool is_first = first;
					first = false;

					int ti;
					int tj;

					std::tie(ti, tj) = pixels2check.front();
					pixels2check.pop();

					#pragma omp critical
					{
						if (checkedPixels.count({ti,tj}) > 0) {
							continue;
						}

						checkedPixels.insert({ti,tj});
					}

					disp_t idx_bg = _bg_disp_idx.valueUnchecked(ti,tj);
					T_CV cost_bg = _bg_cost_volume.valueUnchecked(ti,tj,idx_bg);

					auto opt_fg_cost = on_demand_cv.costValue({ti,tj},{idx_bg});

					if (!opt_fg_cost.has_value()) {
						ret.disp.atUnchecked(ti,tj) = _searchOffset.idx2disp<0>(idx_bg);
						ret.fg_mask.atUnchecked(ti,tj) = StereoDispWithBgMask::Background;
						continue;
					}

					T_CV cost_fg = opt_fg_cost.value();
					disp_t idx_fg = idx_bg;

					if (is_first and std::min(cost_bg, cost_fg)/std::max(cost_bg, cost_fg) > _rel_threshold) {
						continue; //ignore pixels which have the same disp cost as the background, but only in front of a chain.
					}

					for (int d = 0; d < shape[2]; d++) {
						auto opt_fg_cost = on_demand_cv.costValue({ti,tj},{d});
						if (!opt_fg_cost.has_value()) {
							continue;
						}

						T_CV cost_fg_cand = opt_fg_cost.value();
						std::tie(idx_fg, cost_fg) = optimalDispAndCost<matchFunc>(idx_fg,
																	   cost_fg,
																	   d,
																	   cost_fg_cand);

					}

					if ((std::abs(idx_fg - idx_bg) >= _disp_tol)) {
						//do not extend the search space if a pixel has been found close to the background

						for (int di : std::array<int, 3>{-1,0,1}) {

							if (ti+di < 0 or ti+di >= shape[0]) {
								continue;
							}

							for (int dj : std::array<int, 3>{-1,0,1}) {

								if (di == 0 and dj == 0) {
									continue;
								}

								if (tj+dj < 0 or tj+dj >= shape[1]) {
									continue;
								}

								#pragma omp critical
								{
									if (checkedPixels.count({ti+di,tj+dj}) == 0) {
										pixels2check.push({ti+di, tj+dj});
									}
								}
							}
						}
					}

					ret.disp.atUnchecked(ti,tj) = _searchOffset.idx2disp<0>(idx_fg);
					ret.fg_mask.atUnchecked(ti,tj) = (std::abs(idx_fg - idx_bg) >= _disp_tol) ? StereoDispWithBgMask::Foreground : StereoDispWithBgMask::Background;
				}
			}
		}

		return ret;

	}

protected:

	float _rel_threshold;
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
