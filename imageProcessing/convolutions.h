#ifndef CONVOLUTIONS_H
#define CONVOLUTIONS_H

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

#include <array>

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

namespace StereoVision {
namespace ImageProcessing {
namespace Convolution {


enum class AxisType {
    Moving,
    Aggregate,
    BatchedInput,
    BatchedOutput
};

enum class PaddingType {
    Constant,
    Periodic,
    Mirror
};

struct PaddingInfos {

    PaddingInfos() :
        pre_padding(0),
        post_padding(0),
        padding_type(PaddingType::Constant)
    {

    }

    PaddingInfos(int padding, PaddingType paddingType = PaddingType::Constant) :
        pre_padding(padding),
        post_padding(padding),
        padding_type(paddingType)
    {

    }

    PaddingInfos(int pre, int post, PaddingType paddingType = PaddingType::Constant) :
        pre_padding(pre),
        post_padding(post),
        padding_type(paddingType)
    {

    }

    int pre_padding;
    int post_padding;

    PaddingType padding_type;
};

/*!
 * \brief The MovingWindowAxis class represent an axis along which a moving windows will be
 */
class MovingWindowAxis {
public:
    inline constexpr static AxisType getAxisType() {
        return AxisType::Moving;
    }

    MovingWindowAxis(int moving_windows_stride = 0, PaddingInfos padding = PaddingInfos()) :
        _stride(moving_windows_stride),
        _padding(padding)
    {

    }

    MovingWindowAxis(PaddingInfos padding) :
        _stride(0),
        _padding(padding)
    {

    }

    inline int stride() const {
        return _stride;
    }

    inline PaddingInfos padding() const {
        return _padding;
    }

protected:
    int _stride; //the jump in index for the window for that axis
    PaddingInfos _padding; //the padding on the axis
};

/*!
 * \brief The AggregateWindowsAxis class represent an axis that is going to be summed over and removed from the output
 */
class AggregateWindowsAxis {
public:
    inline constexpr static AxisType getAxisType() {
        return AxisType::Aggregate;
    }

    AggregateWindowsAxis()
    {

    }

};

/*!
 * \brief The BatchedInputAxis class represent an axis in the input array that is batched.
 *
 * Each subarray along batched axis will be correlated on its own and left in the output.
 */
class BatchedInputAxis {
public:
    inline constexpr static AxisType getAxisType() {
        return AxisType::BatchedInput;
    }

    BatchedInputAxis()
    {

    }

};

/*!
 * \brief The BatchedOutputAxis class represent an axis in the filter that is batched.
 *
 * Each subfilter along the batched axis will be correlated on its own
 * and added along a corresponding axis in the output.
 */
class BatchedOutputAxis {
public:
    inline constexpr static AxisType getAxisType() {
        return AxisType::BatchedOutput;
    }

    BatchedOutputAxis()
    {

    }

};

template<typename T, typename... Ds>
class Filter {

public:
    constexpr static int nDim = sizeof... (Ds);
    constexpr static std::array<AxisType, nDim> axisTypes = {Ds::getAxisType()...};
protected:

    constexpr static std::array<int, nDim> correspondanceForExcludedAxisType(AxisType excludedType) {
        std::array<int, nDim> ret = {};

        int id = -1;
        for (int i = 0; i < nDim; i++) {
            if (axisTypes[i] != excludedType) {
                id++;
                ret[i] = id;
            } else {
                ret[i] = -1;
            }
        }

        return ret;
    }

    template <int N>
    constexpr static std::array<int, N> invertCorrespondances(std::array<int, nDim> const& directCorrespondances) {
        std::array<int, N> ret = {};

        for (int i = 0; i < N; i++) {
            ret[i] = -1;
        }

        for (int i = 0; i < nDim; i++) {
            if (directCorrespondances[i] >= 0) {
                ret[directCorrespondances[i]] = i;
            }
        }

        return ret;
    }

    constexpr static int countCorrespondance(std::array<int, nDim> const& correspondances) {
        int count = 0;

        for (int i = 0; i < nDim; i++) {
            if (correspondances[i] >= 0) {
                count++;
            }
        }

        return count;
    }

    constexpr static std::array<int, nDim> correspondanceWithInput() {
        return correspondanceForExcludedAxisType(AxisType::BatchedOutput);
    }
    constexpr static std::array<int, nDim> correspondanceWithOutput() {
        return correspondanceForExcludedAxisType(AxisType::Aggregate);
    }
    constexpr static std::array<int, nDim> correspondanceWithFilter() {
        return correspondanceForExcludedAxisType(AxisType::BatchedInput);
    }

    static constexpr int nAxesOfType(AxisType type) {
        int count = 0;

        for (int i = 0; i < nDim; i++) {
            if (axisTypes[i] == type) {
                count++;
            }
        }

        return count;
    }

    static constexpr int nAxesNotOfType(AxisType type) {
        int count = 0;

        for (int i = 0; i < nDim; i++) {
            if (axisTypes[i] != type) {
                count++;
            }
        }

        return count;
    }

public:

    constexpr static int nInputAxes = countCorrespondance(correspondanceWithInput());

    constexpr static int nOutputAxes = countCorrespondance(correspondanceWithOutput());

    constexpr static int nFilterAxes = countCorrespondance(correspondanceWithFilter());

    Filter(Multidim::Array<T, nFilterAxes> coefficients, Ds ... axisDefinitions) {
        _filter = coefficients;

        _strides = {getStrideForAxis(axisDefinitions)...};
        _paddinginfos = {getPaddingForAxis(axisDefinitions)...};
    }

    bool inputShapeValid(std::array<int, nInputAxes> const& inputShape) {
        std::array<int, nInputAxes> cInput = correspondanceFromInput();
        std::array<int, nDim> cFilter = correspondanceWithFilter();

        for (int i = 0; i < nInputAxes; i++) {
            int dimId = cInput[i];
            int filterId = cFilter[dimId];

            if (filterId < 0) {
                if (axisTypes[dimId] == AxisType::BatchedInput) {
                    continue;
                }
                return false;
            }

            if (axisTypes[dimId] == AxisType::Aggregate) {
                if (inputShape[i] != _filter.shape()[filterId]) {
                    return false;
                }
            }
        }

        return true;
    }

    std::array<int, nOutputAxes> outputShape(std::array<int, nInputAxes> const& inputShape) {

        std::array<int, nOutputAxes> cOutput = correspondanceFromOutput();

        std::array<int, nDim> cInput = correspondanceWithInput();
        std::array<int, nDim> cFilter = correspondanceWithFilter();

        std::array<int, nOutputAxes> ret;

        for (int i = 0; i < nOutputAxes; i++) {
            ret[i] = -1;
            int dimId = cOutput[i];

            int inputAxis = cInput[dimId];
            int filterAxis = cFilter[dimId];

            if (axisTypes[dimId] == AxisType::Moving) {
                PaddingInfos& pInfos = _paddinginfos[dimId];
                ret[i] = inputShape[inputAxis] - _filter.shape()[filterAxis] + 1 + pInfos.pre_padding + pInfos.post_padding;
            }

            if (axisTypes[dimId] == AxisType::BatchedInput) {
                ret[i] = inputShape[inputAxis];
            }

            if (axisTypes[dimId] == AxisType::BatchedOutput) {
                ret[i] = _filter.shape()[filterAxis];
            }
        }

        return ret;

    }

    Multidim::Array<T, nOutputAxes> convolve(Multidim::Array<T, nInputAxes> const& input) {

        if (nOutputAxes == 0) {
            return Multidim::Array<T, nOutputAxes>();
        }

        std::array<int, nInputAxes> inputShape = input.shape();

        if (!inputShapeValid(inputShape)) {
            return Multidim::Array<T, nOutputAxes>();
        }

        std::array<int, nOutputAxes> outputsShape = outputShape(inputShape);

        for (int i = 0; i < nOutputAxes; i++) {
            if (outputsShape[i] <= 0) {
                return Multidim::Array<T, nOutputAxes>();
            }
        }

        std::array<int, nInputAxes> inputIdx;
        std::array<int, nFilterAxes> filterIdx;

        constexpr std::array<int, nOutputAxes> crspdFrmOutput = correspondanceFromOutput();
        constexpr std::array<int, nDim> crspdWthOutput = correspondanceWithOutput();

        constexpr std::array<int, nFilterAxes> crspdFrmFilter = correspondanceFromFilter();
        constexpr std::array<int, nDim> crspdWthFilter = correspondanceWithFilter();

        constexpr std::array<int, nInputAxes> crspdFrmInput = correspondanceFromInput();
        constexpr std::array<int, nDim> crspdWthInput = correspondanceWithInput();

        Multidim::Array<T, nOutputAxes> ret(outputsShape);

        // iterate over the output positions first
        Multidim::IndexConverter<nOutputAxes> outputIdxsConverter(ret.shape());

        //#pragma omp parallel for
        for (int out_idx = 0; out_idx < outputIdxsConverter.numberOfPossibleIndices(); out_idx++) {

            std::array<int, nOutputAxes> outputIdx = outputIdxsConverter.getIndexFromPseudoFlatId(out_idx);
            std::array<int, nOutputAxes> outputIdxInternalId;

            for (int i = 0; i < nOutputAxes; i++) {
                outputIdxInternalId[i] = crspdFrmOutput[outputIdx[i]];
            }

            constexpr int nBatchedOutputAxes = nAxesOfType(AxisType::BatchedOutput);
            std::array<int, nBatchedOutputAxes> jointFilterOutputAxesInternalId = axesOfType<AxisType::BatchedOutput>();

            std::array<int, nBatchedOutputAxes> filterExcludedAxis;

            for (int i = 0; i < nBatchedOutputAxes; i++) {
                filterExcludedAxis[i] = crspdWthFilter[jointFilterOutputAxesInternalId[i]];
            }

            Multidim::DimsExclusionSet<nFilterAxes> batchFiltersAxes(static_cast<std::array<int, nBatchedOutputAxes> const&>(filterExcludedAxis));

            Multidim::IndexConverter<nFilterAxes> filterIdxsConverter(_filter.shape(), batchFiltersAxes);

            std::array<int, nFilterAxes> baseFilterIdx = filterIdxsConverter.initialIndex();

            for (int i = 0; i < nBatchedOutputAxes; i++) {
                int outputAxis = crspdWthOutput[jointFilterOutputAxesInternalId[i]];
                int filterAxis = crspdWthOutput[jointFilterOutputAxesInternalId[i]];
            }

            std::array<int, nInputAxes> baseInputIdx;

            for (int i = 0; i < nInputAxes; i++) {
                int dimId = crspdFrmInput[i];

                baseInputIdx[i] = 0;

                if (crspdWthOutput[dimId] >= 0) {
                    baseInputIdx[i] = outputIdx[crspdWthOutput[dimId]];
                }
            }

            T accumulated = 0;

            for (int filt_idx = 0; filt_idx < filterIdxsConverter.numberOfPossibleIndices(); filt_idx++) {

                std::array<int, nFilterAxes> filterIdx = filterIdxsConverter.getIndexFromPseudoFlatId(filt_idx, baseFilterIdx);

                std::array<int, nInputAxes> inputIdx = baseInputIdx;

                //compute the position for moving axis
                for (int dimId : axesOfType<AxisType::Moving>()) {
                    int filterAxis = crspdWthFilter[dimId];
                    int inputAxis = crspdWthInput[dimId];

                    int delta = filterIdx[filterAxis] - _paddinginfos[dimId].pre_padding;

                    inputIdx[inputAxis] += delta;
                    //TODO: add more padding modes here.

                    if (_paddinginfos[dimId].padding_type == PaddingType::Periodic) {
                        int s = inputShape[inputAxis];
                        inputIdx[inputAxis] = (s + (inputIdx[inputAxis] % s)) % s;
                    }

                    if (_paddinginfos[dimId].padding_type == PaddingType::Mirror) {
                        int s = inputShape[inputAxis];
                        int sp = 2*s;

                        int tmp = (sp + (inputIdx[inputAxis] % sp)) % sp;

                        inputIdx[inputAxis] = (s-1)-std::abs((s-1)-tmp);
                    }
                }

                //compute the position for aggregation axis
                for (int dimId : axesOfType<AxisType::Aggregate>()) {
                    int filterAxis = crspdWthFilter[dimId];
                    int inputAxis = crspdWthInput[dimId];

                    inputIdx[inputAxis] = filterIdx[filterAxis];
                }

                T fVal = _filter.valueUnchecked(filterIdx);
                T iVal = input.valueOrAlt(inputIdx,_padding_constant);

                accumulated += fVal*iVal;
            }

            ret.atUnchecked(outputIdx) = accumulated;


        }

        return ret;

    }

    T paddingConstant() const {
        return _padding_constant;
    }

    void setPaddingConstant(T paddingConstant) {
        _padding_constant = paddingConstant;
    }

protected:

    constexpr static std::array<int, nInputAxes> correspondanceFromInput() {
        return invertCorrespondances<nInputAxes>(correspondanceWithInput());
    }
    constexpr static std::array<int, nOutputAxes> correspondanceFromOutput() {
        return invertCorrespondances<nOutputAxes>(correspondanceWithOutput());
    }
    constexpr static std::array<int, nFilterAxes> correspondanceFromFilter() {
        return invertCorrespondances<nFilterAxes>(correspondanceWithFilter());
    }

    template <AxisType axisType>
    constexpr static std::array<int, nAxesOfType(axisType)> axesOfType() {
        std::array<int, nAxesOfType(axisType)> ret;

        int count = 0;
        for (int i = 0; i < nDim; i++) {
            if (axisTypes[i] == axisType) {
                ret[count] = i;
                count++;
            }
        }

        return ret;
    }

    template <AxisType axisType>
    constexpr static std::array<int, nAxesNotOfType(axisType)> axesNotOfType() {
        std::array<int, nAxesNotOfType(axisType)> ret;

        int count = 0;
        for (int i = 0; i < nDim; i++) {
            if (axisTypes[i] != axisType) {
                ret[count] = i;
                count++;
            }
        }

        return ret;
    }

    template<typename AT>
    PaddingInfos getPaddingForAxis(AT const& axis_definition) {
        return PaddingInfos();
    };

    PaddingInfos getPaddingForAxis(MovingWindowAxis const& axis_definition) {
        return axis_definition.padding();
    };

    template<typename AT>
    int getStrideForAxis(AT axis_definition) {
        return 0;
    };

    int getStrideForAxis(MovingWindowAxis axis_definition) {
        return axis_definition.stride();
    };

    std::array<int, nDim> _strides;
    std::array<PaddingInfos, nDim> _paddinginfos;
    Multidim::Array<T, nFilterAxes> _filter;

    T _padding_constant;
};

} //namespace Convolution
} //namespace ImageProcessing
} //namespace StereoVision

#endif // CONVOLUTIONS_H
