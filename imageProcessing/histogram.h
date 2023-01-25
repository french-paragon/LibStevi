#ifndef LIBSTEVI_HISTOGRAM_H
#define LIBSTEVI_HISTOGRAM_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2023  Paragon<french.paragon@gmail.com>

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
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <cmath>

namespace StereoVision {
namespace ImageProcessing {

template<typename ImT>
class Histogram {
public:

	Histogram() : _isValid(false) {

	}

        template<int nDim, typename T = ImT>
        Histogram(Multidim::Array<ImT, nDim> const& img, typename std::enable_if<std::is_same_v<T, ImT> and std::is_integral_v<T>, void>::type* = nullptr) : _isValid(true) {

		int nChannels = 1;

		Multidim::IndexConverter<nDim> idxConv(img.shape());

		ImT min = img.valueUnchecked(idxConv.initialIndex());
		ImT max = img.valueUnchecked(idxConv.initialIndex());

		for (int i = 0; i < idxConv.numberOfPossibleIndices(); i++) {
			auto idx = idxConv.getIndexFromPseudoFlatId(i);

			ImT val = img.valueUnchecked(idx);

			if (val < min) {
				min = val;
			}

			if (val > max) {
				max = val;
			}
		}

		_minVal = min;
		_maxVal = max;

		int bins = max - min + 1;

		_histogramValues = Multidim::Array<int, 2>(bins, nChannels);

		for (int i = 0; i < bins; i++) {
			_histogramValues.atUnchecked(i, 0) = 0;
		}

		for (int i = 0; i < idxConv.numberOfPossibleIndices(); i++) {
			auto idx = idxConv.getIndexFromPseudoFlatId(i);

			ImT val = img.valueUnchecked(idx);
			int bin = getBinId(val);

			_histogramValues.atUnchecked(bin, 0) += 1;
		}

	}

        template<int nDim, typename T = ImT>
        Histogram(Multidim::Array<ImT, nDim> const& img, ImT min, ImT max, typename std::enable_if<std::is_same_v<T, ImT> and std::is_integral_v<T>, void>::type* = nullptr) :
		_isValid(true),
		_minVal(std::min(min, max)),
		_maxVal(std::max(min, max))
	{

		int nChannels = 1;

		Multidim::IndexConverter<nDim> idxConv(img.shape());

		int bins = _maxVal - _minVal + 1;

		_histogramValues = Multidim::Array<int, 2>(bins, nChannels);

		for (int i = 0; i < bins; i++) {
			_histogramValues.atUnchecked(i, 0) = 0;
		}

		for (int i = 0; i < idxConv.numberOfPossibleIndices(); i++) {
			auto idx = idxConv.getIndexFromPseudoFlatId(i);

			ImT val = img.valueUnchecked(idx);
			int bin = getBinId(val);

                        if (bin < 0) {
                            bin = 0;
                        }

                        if (bin >= nBins()) {
                            bin = nBins()-1;
                        }

			_histogramValues.atUnchecked(bin, 0) += 1;
		}

	}


        template<int nDim, typename T = ImT>
        Histogram(Multidim::Array<ImT, nDim> const& img, ImT binWidth, typename std::enable_if<std::is_same_v<T, ImT> and std::is_floating_point_v<T>, void>::type* = nullptr) : _isValid(true) {

		int nChannels = 1;

		Multidim::IndexConverter<nDim> idxConv(img.shape());

		ImT min = img.valueUnchecked(idxConv.initialIndex());
		ImT max = img.valueUnchecked(idxConv.initialIndex());

		for (int i = 0; i < idxConv.numberOfPossibleIndices(); i++) {
			auto idx = idxConv.getIndexFromPseudoFlatId(i);

			ImT val = img.valueUnchecked(idx);

			if (val < min) {
				min = val;
			}

			if (val > max) {
				max = val;
			}
		}

		_minVal = min;
		_maxVal = max;

		int bins = std::ceil((max - min)/binWidth);
		ImT delta = (binWidth*bins - (max - min))/2;
		_minVal -= delta;
		_maxVal += delta;

		_histogramValues = Multidim::Array<int, 2>(bins, nChannels);

		for (int i = 0; i < bins; i++) {
			_histogramValues.atUnchecked(i, 0) = 0;
		}

		for (int i = 0; i < idxConv.numberOfPossibleIndices(); i++) {
			auto idx = idxConv.getIndexFromPseudoFlatId(i);

			ImT val = img.valueUnchecked(idx);
			int bin = getBinId(val);

			_histogramValues.atUnchecked(bin, 0) += 1;
		}

	}

        template<int nDim, typename T = ImT>
        Histogram(Multidim::Array<ImT, nDim> const& img, ImT binWidth,ImT min, ImT max, typename std::enable_if<std::is_same_v<T, ImT> and std::is_floating_point_v<T>, void>::type* = nullptr) :
		_isValid(true),
		_minVal(std::min(min, max)),
		_maxVal(std::max(min, max))
	{

		int nChannels = 1;

		Multidim::IndexConverter<nDim> idxConv(img.shape());

		int bins = std::ceil((_maxVal - _minVal)/binWidth);
		ImT delta = (binWidth*bins - (max - min))/2;
		_minVal -= delta;
		_maxVal += delta;

		_histogramValues = Multidim::Array<int, 2>(bins, nChannels);

		for (int i = 0; i < bins; i++) {
			_histogramValues.atUnchecked(i, 0) = 0;
		}

		for (int i = 0; i < idxConv.numberOfPossibleIndices(); i++) {
			auto idx = idxConv.getIndexFromPseudoFlatId(i);

			ImT val = img.valueUnchecked(idx);
			int bin = getBinId(val);

                        if (bin < 0) {
                            bin = 0;
                        }

                        if (bin >= nBins()) {
                            bin = nBins()-1;
                        }

			_histogramValues.atUnchecked(bin, 0) += 1;
		}

	}

	inline bool isValid() {
		return _isValid;
	}

	inline int nBins() const {
		return _histogramValues.shape()[0];
	}
	int getBinCount(int binId, int channel = 0) const {
		return _histogramValues.value(binId, channel);
	}

	inline int nChannels() const {
		return _histogramValues.shape()[1];
	}

	int getBinId(ImT val) {

		if (std::is_integral_v<ImT>) {
			return val - _minVal;
		}

		if (val == _maxVal) {
			return nBins()-1;
		}

		ImT deltaPlus = val - _minVal;
		float prop = float(deltaPlus)/(_maxVal-_minVal);
		float fracId = prop*nBins();

		int id = std::floor(fracId);

		return id;
	}
	ImT getBinBase(ImT val) {
		if (std::is_integral_v<ImT>) {
			return getBinId(val) + _minVal;
		}

		int binid = getBinId(val);

		return binid*(_maxVal-_minVal)/(nBins());
	}
	ImT getBinSup(ImT val) {
		if (std::is_integral_v<ImT>) {
			return getBinId(val) + _minVal + 1;
		}

		int binid = getBinId(val) + _minVal + 1;

		return binid*(_maxVal-_minVal)/(nBins());
	}

        double entropy(int channel = 0) const {

            double entropy = 0;

            for (int i = 0; i < nBins(); i++) {
                entropy += getBinCount(i, channel)*getBinCount(i, channel);
            }

            return entropy;
        }

private:

	bool _isValid;

	ImT _minVal;
	ImT _maxVal;

	Multidim::Array<int, 2> _histogramValues;
};

} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_HISTOGRAM_H
