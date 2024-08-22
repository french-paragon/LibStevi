#ifndef HUBER_KERNEL_H
#define HUBER_KERNEL_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024  Paragon<french.paragon@gmail.com>

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

#include <cmath>

namespace StereoVision {
namespace Optimization {

/*!
 * \brief huberLoss
 * \param val the input parameter of the loss
 * \param threshold the threshold metaparameter where the loss transition from quadratic to affine.
 * \return the Huber loss of val.
 */
template<typename T>
inline T huberLoss(T val, T threshold = 1) {
    T vAbs = std::abs(val);

    if (vAbs < threshold) {
        return (1./2.) * val*val;
    }

    return threshold * (vAbs - (1./2.) * threshold);
}

/*!
 * \brief diffHuberLoss is the first derivative of the Huber loss
 * \param val the input parameter of the loss
 * \param threshold the threshold metaparameter where the loss goes from quadratic to affine.
 * \return the huber loss derivative at val.
 */
template<typename T>
inline T diffHuberLoss(T val, T threshold = 1) {
    T vAbs = std::abs(val);

    if (vAbs < threshold) {
        return val;
    }

    return threshold * ((val > 0) ? 1 : -1);
}

/*!
 * \brief diff2HuberLoss is the second derivative of the Huber loss
 * \param val the input parameter of the loss
 * \param threshold the threshold metaparameter where the loss goes from quadratic to affine.
 * \return the huber loss second derivative at val.
 */
template<typename T>
inline T diff2HuberLoss(T val, T threshold = 1) {
    T vAbs = std::abs(val);

    if (vAbs < threshold) {
        return 1;
    }

    return 0;
}

/*!
 * \brief pseudoHuberLoss
 * \param val the input parameter of the loss
 * \param threshold the threshold metaparameter where the loss transition from quadratic to affine.
 * \return the Pseudo Huber loss of val.
 */
template<typename T>
inline T pseudoHuberLoss(T val, T threshold = 1) {

    T rel = val/threshold;

    return threshold*threshold * (std::sqrt(1+rel*rel) - 1);
}

/*!
 * \brief diffPseudoHuberLoss is the first derivative of the Pseudo Huber loss
 * \param val the input parameter of the loss
 * \param threshold the threshold metaparameter where the loss goes from quadratic to affine.
 * \return the pseudo huber loss derivative at val.
 */
template<typename T>
inline T diffPseudoHuberLoss(T val, T threshold = 1) {

    T rel = val/threshold;

    return 1./std::sqrt(1+rel*rel) * val;
}

/*!
 * \brief diff2PseudoHuberLoss is the second derivative of the pseudo Huber loss
 * \param val the input parameter of the loss
 * \param threshold the threshold metaparameter where the loss goes from quadratic to affine.
 * \return the huber loss second derivative at val.
 */
template<typename T>
inline T diff2PseudoHuberLoss(T val, T threshold = 1) {

    T rel = val/threshold;

    return 1./std::sqrt(1+rel*rel) - (1./((1+rel*rel)*sqrt(1+rel*rel))*rel)*val;
}



/*!
 * \brief sqrtHuberLoss is the square root of the Huber loss
 * \param val the input parameter of the loss
 * \param threshold the threshold metaparameter where the loss transition from quadratic to affine.
 * \return the Huber loss sqrt of val.
 *
 * The sqrt of the Huber loss is usefull to replace the quadratic loss with the Huber loss in the Gauss Newtown algorithm.
 */
template<typename T>
inline T sqrtHuberLoss(T val, T threshold = 1) {
    T vAbs = std::abs(val);

    if (vAbs < threshold) {
        return (1./std::sqrt(2.)) * val;
    }

    return std::sqrt(threshold * (vAbs - threshold/2.)) * (val < 0 ? -1 : 1);
}

/*!
 * \brief diffSqrtHuberLoss is the first derivative of the Huber loss square root
 * \param val the input parameter of the loss
 * \param threshold the threshold metaparameter where the loss goes from quadratic to affine.
 * \return the huber loss square root derivative at val.
 */
template<typename T>
inline T diffSqrtHuberLoss(T val, T threshold = 1) {
    T vAbs = std::abs(val);

    if (vAbs < threshold) {
        return (1./std::sqrt(2.));
    }

    return threshold/2. * 1/std::sqrt(threshold * (vAbs - threshold/2.));
}

}
}

#endif // HUBER_KERNEL_H
