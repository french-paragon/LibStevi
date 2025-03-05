#ifndef SENSORFRAMESCONVENTION_H
#define SENSORFRAMESCONVENTION_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2025 Paragon<french.paragon@gmail.com>

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
#include <cmath>

#include <Eigen/Core>

namespace StereoVision {
namespace Geometry {

enum SensorAxis {
    Left = -1,
    Right = 1,
    Front = 2,
    Back = -2,
    Up = -4,
    Down = 4
};

enum StadardFrames {
    FRD,
    LFD,
    FLU,
    RFU
};

using AxisSet = std::array<SensorAxis,3>;

inline constexpr bool axisValid(AxisSet const& set) {
    int mask = std::abs(set[0]) | std::abs(set[1]) | std::abs(set[2]);
    return mask == (Right | Front | Down);
}

inline constexpr bool axisIsRightHand(AxisSet const& set) {

    switch(set[0]) {
    case Left:
        switch(set[1]) {
        case Left:
        case Right:
            return false;
        case Front:
            return set[2] == Down;
        case Back:
            return set[2] == Up;
        case Up:
            return set[2] == Front;
        case Down:
            return set[2] == Back;
        }
        break;
    case Right:
        switch(set[1]) {
        case Left:
        case Right:
            return false;
        case Front:
            return set[2] == Up;
        case Back:
            return set[2] == Down;
        case Up:
            return set[2] == Back;
        case Down:
            return set[2] == Front;
        }
        break;
    case Front:
        switch(set[1]) {
        case Left:
            return set[2] == Up;
        case Right:
            return set[2] == Down;
        case Front:
        case Back:
            return false;
        case Up:
            return set[2] == Right;
        case Down:
            return set[2] == Left;
        }
        break;
    case Back:
        switch(set[1]) {
        case Left:
            return set[2] == Down;
        case Right:
            return set[2] == Up;
        case Front:
        case Back:
            return false;
        case Up:
            return set[2] == Left;
        case Down:
            return set[2] == Right;
        }
        break;
    case Up:
        switch(set[1]) {
        case Left:
            return set[2] == Back;
        case Right:
            return set[2] == Front;
        case Front:
            return set[2] == Left;
        case Back:
            return set[2] == Right;
        case Up:
        case Down:
            return false;
        }
        break;
    case Down:
        switch(set[1]) {
        case Left:
            return set[2] == Front;
        case Right:
            return set[2] == Back;
        case Front:
            return set[2] == Right;
        case Back:
            return set[2] == Left;
        case Up:
        case Down:
            return false;
        }
        break;
    }

    return false;
}

template <SensorAxis ax1, SensorAxis ax2, SensorAxis ax3>
struct AxisSystemDefintion {
    static constexpr AxisSet axisSet{ax1, ax2, ax3};
};

namespace internal {

static_assert (axisIsRightHand({Front, Right, Down}));
static_assert (axisIsRightHand({Right, Back, Down}));
static_assert (axisIsRightHand({Back, Left, Down}));
static_assert (axisIsRightHand({Left, Front, Down}));
static_assert (axisIsRightHand({Front, Left, Up}));
static_assert (axisIsRightHand({Right, Front, Up}));
static_assert (axisIsRightHand({Back, Right, Up}));
static_assert (axisIsRightHand({Left, Back, Up}));
static_assert (axisIsRightHand({Up, Right, Front}));
static_assert (axisIsRightHand({Right, Down, Front}));
static_assert (axisIsRightHand({Down, Left, Front}));
static_assert (axisIsRightHand({Left, Up, Front}));
static_assert (axisIsRightHand({Up, Left, Back}));
static_assert (axisIsRightHand({Right, Up, Back}));
static_assert (axisIsRightHand({Down, Right, Back}));
static_assert (axisIsRightHand({Left, Down, Back}));
static_assert (axisIsRightHand({Front, Up, Right}));
static_assert (axisIsRightHand({Up, Back, Right}));
static_assert (axisIsRightHand({Back, Down, Right}));
static_assert (axisIsRightHand({Down, Front, Right}));
static_assert (axisIsRightHand({Up, Front, Left}));
static_assert (axisIsRightHand({Back, Up, Left}));
static_assert (axisIsRightHand({Down, Back, Left}));
static_assert (axisIsRightHand({Front, Down, Left}));

static_assert (!axisIsRightHand({Front, Right, Up}));
static_assert (!axisIsRightHand({Right, Back, Up}));
static_assert (!axisIsRightHand({Back, Left, Up}));
static_assert (!axisIsRightHand({Left, Front, Up}));
static_assert (!axisIsRightHand({Front, Left, Down}));
static_assert (!axisIsRightHand({Right, Front, Down}));
static_assert (!axisIsRightHand({Back, Right, Down}));
static_assert (!axisIsRightHand({Left, Back, Down}));
static_assert (!axisIsRightHand({Up, Right, Back}));
static_assert (!axisIsRightHand({Right, Down, Back}));
static_assert (!axisIsRightHand({Down, Left, Back}));
static_assert (!axisIsRightHand({Left, Up, Back}));
static_assert (!axisIsRightHand({Up, Left, Front}));
static_assert (!axisIsRightHand({Right, Up, Front}));
static_assert (!axisIsRightHand({Down, Right, Front}));
static_assert (!axisIsRightHand({Left, Down, Front}));
static_assert (!axisIsRightHand({Front, Up, Left}));
static_assert (!axisIsRightHand({Up, Back, Left}));
static_assert (!axisIsRightHand({Back, Down, Left}));
static_assert (!axisIsRightHand({Down, Front, Left}));
static_assert (!axisIsRightHand({Up, Front, Right}));
static_assert (!axisIsRightHand({Back, Up, Right}));
static_assert (!axisIsRightHand({Down, Back, Right}));
static_assert (!axisIsRightHand({Front, Down, Right}));

static_assert (!axisIsRightHand({Front, Right, Right}));
static_assert (!axisIsRightHand({Right, Back, Right}));
static_assert (!axisIsRightHand({Back, Left, Back}));
static_assert (!axisIsRightHand({Left, Front, Front}));
static_assert (!axisIsRightHand({Front, Left, Left}));
static_assert (!axisIsRightHand({Right, Front, Right}));
static_assert (!axisIsRightHand({Back, Right, Right}));
static_assert (!axisIsRightHand({Left, Back, Back}));
static_assert (!axisIsRightHand({Right, Right, Right}));
static_assert (!axisIsRightHand({Right, Down, Down}));
static_assert (!axisIsRightHand({Down, Left, Down}));
static_assert (!axisIsRightHand({Left, Up, Up}));
static_assert (!axisIsRightHand({Up, Front, Front}));
static_assert (!axisIsRightHand({Front, Up, Front}));

} //internal

/*!
 * \brief getSensorFrameConversion return the rotation matrix converting vectors in frame convention1 to frame convention2
 * \return a roation matrix
 */
template<typename Source, typename Target>
inline Eigen::Matrix3i getSensorFrameConversion() {

    static_assert (axisIsRightHand(Source::axisSet), "Source does not follow the right hand rule!");
    static_assert (axisIsRightHand(Target::axisSet), "Target does not follow the right hand rule!");

    if (std::is_same_v<Source, Target>) {
        return Eigen::Matrix3i::Identity();
    }

    Eigen::Matrix3i convention1_to_RFU;
    Eigen::Matrix3i RFU_to_convention2;

    for (int i = 0; i < 3; i++) {
        switch(Source::axisSet[i]) {
        case Left:
            convention1_to_RFU(0,i) = -1;
            convention1_to_RFU(1,i) = 0;
            convention1_to_RFU(2,i) = 0;
            break;
        case Right:
            convention1_to_RFU(0,i) = 1;
            convention1_to_RFU(1,i) = 0;
            convention1_to_RFU(2,i) = 0;
            break;
        case Front:
            convention1_to_RFU(0,i) = 0;
            convention1_to_RFU(1,i) = 1;
            convention1_to_RFU(2,i) = 0;
            break;
        case Back:
            convention1_to_RFU(0,i) = 0;
            convention1_to_RFU(1,i) = -1;
            convention1_to_RFU(2,i) = 0;
            break;
        case Up:
            convention1_to_RFU(0,i) = 0;
            convention1_to_RFU(1,i) = 0;
            convention1_to_RFU(2,i) = 1;
            break;
        case Down:
            convention1_to_RFU(0,i) = 0;
            convention1_to_RFU(1,i) = 0;
            convention1_to_RFU(2,i) = -1;
            break;
        }

        switch(Target::axisSet[i]) {
        case Left:
            RFU_to_convention2(i,0) = -1;
            RFU_to_convention2(i,1) = 0;
            RFU_to_convention2(i,2) = 0;
            break;
        case Right:
            RFU_to_convention2(i,0) = 1;
            RFU_to_convention2(i,1) = 0;
            RFU_to_convention2(i,2) = 0;
            break;
        case Front:
            RFU_to_convention2(i,0) = 0;
            RFU_to_convention2(i,1) = 1;
            RFU_to_convention2(i,2) = 0;
            break;
        case Back:
            RFU_to_convention2(i,0) = 0;
            RFU_to_convention2(i,1) = -1;
            RFU_to_convention2(i,2) = 0;
            break;
        case Up:
            RFU_to_convention2(i,0) = 0;
            RFU_to_convention2(i,1) = 0;
            RFU_to_convention2(i,2) = 1;
            break;
        case Down:
            RFU_to_convention2(i,0) = 0;
            RFU_to_convention2(i,1) = 0;
            RFU_to_convention2(i,2) = -1;
            break;
        }
    }

    return RFU_to_convention2*convention1_to_RFU;
}

} // namespace Geometry
} // namespace StereoVision

#endif // SENSORFRAMESCONVENTION_H
