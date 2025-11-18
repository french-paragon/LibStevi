#ifndef LIBSTEVI_FOURIERTRANSFORM_H
#define LIBSTEVI_FOURIERTRANSFORM_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2025  Paragon<french.paragon@gmail.com>

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

#include <fftw3.h>
#include <type_traits>
#include <complex>

namespace StereoVision {
namespace ImageProcessing {

/*!
 * \brief The FourierTransformCalculator class is a convinience wrapper of the fftw library, mainly to streamline ressource management using RAII
 */
template<typename T>
class FourierTransformCalculator{
public:

    using InputT = T;
    using OutputT = std::complex<double>;

    static constexpr bool InputIsReal = std::is_same_v<T,float> or std::is_same_v<T,double> or std::is_same_v<T,long double>;

    FourierTransformCalculator() {
        _in = nullptr;
        _out = nullptr;
        _plan = nullptr;
    }

    FourierTransformCalculator(int bufferSize) {

        _inSize = bufferSize;
        _outSize = (InputIsReal) ? bufferSize/2 +1 : bufferSize;

        _in = (InputBufferT*) fftw_malloc(sizeof(InputBufferT) * _inSize);
        _out = (OutputBufferT*) fftw_malloc(sizeof(OutputBufferT) * _outSize);

        if constexpr (InputIsReal) {
            _plan = fftw_plan_dft_r2c_1d(_inSize, _in, _out, FFTW_ESTIMATE);
        } else {
            _plan = fftw_plan_dft_1d(_inSize, _in, _out, FFTW_FORWARD, FFTW_ESTIMATE);
        }
    }

    FourierTransformCalculator(FourierTransformCalculator && other) {
        _plan = other._plan;
        _in = other._in;
        _out = other._out;
        _inSize = other._inSize;
        _outSize = other._outSize;

        other._plan = nullptr;
        other._in = nullptr;
        other._out = nullptr;
    }

    FourierTransformCalculator(FourierTransformCalculator const& other) :
        FourierTransformCalculator(other.inSize())
    {
        for (int i = 0; i < _inSize; i++) {
            _in[i] = other._in[i];
        }
        for (int i = 0; i < _outSize; i++) {
            _out[i] = other._out[i];
        }
    }

    FourierTransformCalculator& operator=(FourierTransformCalculator && other) {
        _plan = other._plan;
        _in = other._in;
        _out = other._out;
        _inSize = other._inSize;
        _outSize = other._outSize;

        other._plan = nullptr;
        other._in = nullptr;
        other._out = nullptr;

        return *this;
    }

    FourierTransformCalculator& operator=(FourierTransformCalculator const& other)
    {
        if (_plan != nullptr) {
            fftw_destroy_plan(_plan);
        }

        if (_in != nullptr) {
            fftw_free(_in);
        }

        if (_out != nullptr) {
            fftw_free(_out);
        }

        int bufferSize = other.inSize();

        _inSize = bufferSize;
        _outSize = (InputIsReal) ? bufferSize/2 +1 : bufferSize;

        _in = (InputBufferT*) fftw_malloc(sizeof(InputBufferT) * _inSize);
        _out = (OutputBufferT*) fftw_malloc(sizeof(OutputBufferT) * _outSize);

        if constexpr (InputIsReal) {
            _plan = fftw_plan_dft_r2c_1d(_inSize, _in, _out, FFTW_ESTIMATE);
        } else {
            _plan = fftw_plan_dft_1d(_inSize, _in, _out, FFTW_FORWARD, FFTW_ESTIMATE);
        }

        for (int i = 0; i < _inSize; i++) {
            _in[i] = other._in[i];
        }
        for (int i = 0; i < _outSize; i++) {
            _out[i] = other._out[i];
        }
    }

    ~FourierTransformCalculator() {
        if (_plan != nullptr) {
            fftw_destroy_plan(_plan);
        }

        if (_in != nullptr) {
            fftw_free(_in);
        }

        if (_out != nullptr) {
            fftw_free(_out);
        }
    }

    inline void setInbufferElement(int i, InputT val) {
        _in[i] = val;
    }

    inline OutputT outElement(int i) const {
        int idx = i;
        return OutputT{_out[idx][0], _out[idx][1]};
    }

    inline double outElementPhase(int i) const {
        return std::arg(outElement(i));
    }
    inline double outElementAmplitude(int i) const {
        return std::abs(outElement(i));
    }

    inline double outElementReal(int i) const {
        return std::real(outElement(i));
    }
    inline double outElementImaginary(int i) const {
        return std::imag(outElement(i));
    }

    inline int inSize() const {
        return _inSize;
    }
    inline int outSize() const {
        return _outSize;
    }

    inline void execute() {
        fftw_execute(_plan);
    }

protected:

    using InputBufferT = typename std::conditional<InputIsReal,double,fftw_complex>::type;
    using OutputBufferT = fftw_complex;

    int _inSize;
    int _outSize;
    fftw_plan _plan;
    InputBufferT* _in;
    OutputBufferT* _out;

};

} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_FOURIERTRANSFORM_H
