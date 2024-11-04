#ifndef GENERICBOUNDINGVOLUMEHIERARCHY_H
#define GENERICBOUNDINGVOLUMEHIERARCHY_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024 Paragon<french.paragon@gmail.com>

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

#include <algorithm>
#include <limits>
#include <array>
#include <cmath>
#include <vector>

namespace StereoVision {
namespace Geometry {


/*!
 * \brief The BVHObjectWrapper class acts as a wrapper for an object that will go into a BVH
 *
 * A BVHObjectWrapper is expected to provide an operator[] with an int parameter representing the dimension.
 * The operator[] must return a PosT (default = float) representing the coordinate of the object in space.
 * If operator[] return a BVHDimRange, then the object is already expected to represent an Axis Aligned Bounding Box
 *
 * The default implementation forward the call to T's own operator[] (so it should work by default for Eigen's vectors for example).
 */
template<typename T, typename PT = float>
class BSPObjectWrapper {
public:

    typedef PT PosT;

    BSPObjectWrapper(T const& obj) :
        _obj(obj)
    {

    }

    inline PosT operator[](int dim) const {
        return _obj[dim];
    }

protected:
    T const& _obj;
};

/*!
 * \brief The GenericBSP class represent a binary space partitioning (BSP) tree for an arbitrary type T
 *
 * The class takes ownership of a collection of objects of arbitrary types contained in an arbitrary container class-
 * It and provide a spatial ordering in arbitrary number of dimensions.
 * Construction support move semantic for optimal data transfer, or full copy of the data.
 * A wapper class has to be provided to extract spatial position of the objects
 */
template<typename T, int nD, typename WT = BSPObjectWrapper<T,float>, typename CT = std::vector<T>>
class GenericBSP {
public:

    static constexpr int nDims = nD;

    typedef CT ContainerT;
    typedef WT WrapperT;
    typedef typename WrapperT::PosT PosT;

    struct SearchRange {
        PosT min;
        PosT max;
    };

    struct IndexRange {
        int node;
        int start;
        int end;
        int dim;

        inline int size() const {
            return end - start;
        }
    };

    typedef std::array<SearchRange, nDims> SearchBoxT;

    inline static constexpr int getRangeCutIdx(int size) {
        return std::max(0,(size%2 == 0) ? size/2-1 : size/2);
    }

    inline static constexpr int getCurrentDim(int level) {
        return level % nDims;
    }


    GenericBSP(ContainerT const& container) :
        _data(container)
    {
        sortRange(_data.begin(), _data.end());
    }

    GenericBSP(ContainerT && container) :
        _data(container)
    {
        sortRange(_data.begin(), _data.end());
    }

    T & operator[](int idx) {
        return _data[idx];
    }

    T const& operator[](int idx) const {
        return _data[idx];
    }

    T & closest(T const& target) {
        PosT bestDistanceSq = std::numeric_limits<PosT>::max();
        RangeLimits range;
        auto it = findClosest(target, _data.begin(), _data.end(), 0, bestDistanceSq, range);
        return *it;
    }

    int closestInRange(T const& target, T const& min, T const& max) {

        WrapperT targetW(target);
        WrapperT minW(min);
        WrapperT maxW(max);

        for (int i = 0; i < nD; i++) {
            if (minW[i] > maxW[i]) {
                return -1;
            }
            if (targetW[i] > maxW[i]) {
                return -1;
            }
            if (minW[i] > targetW[i]) {
                return -1;
            }
        }

        PosT bestDistanceSq = std::numeric_limits<PosT>::max();
        auto it = findClosestInRange(target, _data.begin(), _data.end(), min, max, 0, bestDistanceSq);

        if (it == _data.end()) {
            return -1;
        }

        return std::distance(_data.begin(), it);
    }

    T const& closest(T const& target) const{
        PosT bestDistanceSq = std::numeric_limits<PosT>::max();
        RangeLimits range;
        auto it = findClosest(target, _data.cbegin(), _data.cend(), 0, bestDistanceSq, range);
        return *it;
    }

    /*!
     * \brief elementsInRange return all the elements in a given range
     * \param min the element which coordinates correspond to all min coordinates in the range
     * \param max the element which coordinates correspond to all max coordinates in the range
     * \return a vector containing all the elements which coordinates are in between min and max ccordinates.
     */
    std::vector<T> elementsInRange(T const& min, T const& max) const {
        WrapperT wMin(min);
        WrapperT wMax(max);
        RangeLimits range(wMin, wMax);

        return findElementsInRange(range, _data.cbegin(), _data.cend());
    }

    /*!
     * \brief indexRange give the infos about the full range of points
     * \return the range convering the full tree
     */
    IndexRange indexRange() const {
        int n = getRangeCutIdx(_data.size());
        return {n, 0, _data.size(), 0};
    }

    /*!
     * \brief previousRange give the small range from a given range
     * \param range the range under consideration
     * \return the range corresponding to the smaller values.
     *
     * Note that if the range is not constructed from the base range, then the value returned from this function is underfined (and very likely invalid).
     */
    IndexRange previousRange(IndexRange const& range) const {
        int n = getRangeCutIdx(range.end - range.start);
        return {range.start + n, range.start, range.node, (range.dim + 1)%nDims};
    }

    /*!
     * \brief previousRange give the small range from a given range
     * \param range the range under consideration
     * \return the range corresponding to the smaller values.
     *
     * Note that if the range is not constructed from the base range, then the value returned from this function is underfined (and very likely invalid).
     */
    IndexRange nextRange(IndexRange const& range) const {
        int n = getRangeCutIdx(range.end - range.start);
        return {range.start + n, range.start, range.node, (range.dim + 1)%nDims};
    }

protected:

    struct RangeLimits {

        std::array<PosT, nDims> lowLimits;
        std::array<PosT, nDims> highLimits;

        RangeLimits() {
            for (int i = 0; i < nDims; i++) {
                lowLimits[i] = (std::numeric_limits<PosT>::has_infinity) ? -std::numeric_limits<PosT>::infinity() : std::numeric_limits<PosT>::lowest();
                highLimits[i] = (std::numeric_limits<PosT>::has_infinity) ? std::numeric_limits<PosT>::infinity() : std::numeric_limits<PosT>::max();
            }
        }

        RangeLimits(WrapperT const& pt1, WrapperT const& pt2) {
            for (int i = 0; i < nDims; i++) {
                PosT v1 = pt1[i];
                PosT v2 = pt2[i];
                lowLimits[i] = (v1 < v2) ? v1 : v2;
                highLimits[i] = (v1 < v2) ? v2 : v1;
            }
        }

        RangeLimits(RangeLimits const& other) :
            lowLimits(other.lowLimits),
            highLimits(other.highLimits)
        {

        }

        inline bool pointInRange(WrapperT const& pt) const {
            for (int i = 0; i < nDims; i++) {
                if (pt[i] < lowLimits[i]) {
                    return false;
                }
                if (pt[i] > highLimits[i]) {
                    return false;
                }
            }
            return true;
        }

        inline bool circleIntersectRange(WrapperT const& center, WrapperT const& side) const {
            bool centerInRange = pointInRange(center);

            if (centerInRange) {
                return true;
            }

            std::array<PosT, nDims> closest;

            for (int i = 0; i < nDims; i++) {
                PosT val = center[i];
                PosT const& low = lowLimits[i];
                PosT const& high = highLimits[i];
                closest[i] = (std::abs(low - val) < std::abs(high - val)) ? low : high;
            }

            PosT sqrRad = 0;
            PosT sqrDist = 0;

            for (int i = 0; i < nDims; i++) {
                PosT tmp;
                tmp = center[i] - side[i];
                sqrRad += tmp*tmp;
                tmp = center[i] - closest[i];
                sqrDist += tmp*tmp;
            }
            return sqrDist < sqrRad;
        }

        inline bool circleIntersectRange(WrapperT const& center, PosT const& radiusSquared) const {
            bool centerInRange = pointInRange(center);

            if (centerInRange) {
                return true;
            }

            std::array<PosT, nDims> closest;

            for (int i = 0; i < nDims; i++) {
                PosT val = center[i];
                PosT const& low = lowLimits[i];
                PosT const& high = highLimits[i];
                if (val >= low and val <= high) {
                    closest[i] = val;
                } else if (val > high) {
                    closest[i] = high;
                } else {
                    closest[i] = low;
                }
            }

            PosT sqrDist = 0;

            for (int i = 0; i < nDims; i++) {
                PosT tmp;
                tmp = center[i] - closest[i];
                sqrDist += tmp*tmp;
            }
            return sqrDist < radiusSquared;
        }
    };

    inline static PosT computeDistSq(T const& elem1, T const& elem2) {
        WrapperT w1(elem1);
        WrapperT w2(elem2);

        PosT dist = 0;

        for (int i = 0; i < nDims; i++) {
            PosT tmp = w1[i] - w2[i];
            dist += tmp*tmp;
        }

        return dist;
    }

    template< class RandomIt>
    void sortRange(RandomIt start, RandomIt end, int level = 0) {

        int dim = getCurrentDim(level);

        auto compare = [dim] (T const& first, T const& second) {
            WrapperT wFirst(first);
            WrapperT wSecond(second);
            return wFirst[dim] < wSecond[dim];
        };

        int len = std::distance(start, end);
        int n = getRangeCutIdx(len);

        RandomIt nth = start;
        std::advance(nth, n);

        RandomIt nthp1 = nth;
        nthp1++;

        std::nth_element(start, nth, end, compare);

        if (start != nth and nth != end) {
            sortRange(start, nth, level+1);
        }

        if (nthp1 != end and nth != end) {
            sortRange(nthp1, end, level+1);
        }
    }

    template< class RandomIt>
    RandomIt findClosest(T const& target,
                         RandomIt start,
                         RandomIt end,
                         int level,
                         PosT & bestDistanceSq,
                         RangeLimits & range) {

        WrapperT wTarget(target);

        int dim = getCurrentDim(level);

        int len = std::distance(start, end);
        int n = getRangeCutIdx(len);

        RandomIt nth = start;
        std::advance(nth, n);
        WrapperT wNode(*nth);

        RandomIt cand = end;

        PosT nodeDistanceSq = computeDistSq(target, *nth);

        if (nodeDistanceSq <= bestDistanceSq) {
            bestDistanceSq = nodeDistanceSq;
            cand = nth;
        }

        PosT planeDist = wTarget[dim] - wNode[dim];
        PosT planeDistSq = planeDist*planeDist;

        RandomIt nthp1 = nth;
        nthp1++;

        PosT prevRangeMax = range.highLimits[dim];
        PosT prevRangeMin = range.lowLimits[dim];

        if (planeDist < 0) {

            if (start != nth and nth != end) {

                range.highLimits[dim] = wNode[dim];

                RandomIt candPrev = findClosest(target, start, nth, level+1, bestDistanceSq, range);

                if (candPrev != nth) {
                    cand = candPrev;
                }
            }

            range.highLimits[dim] = prevRangeMax;

            range.lowLimits[dim] = wNode[dim];

            if (range.circleIntersectRange(wTarget, bestDistanceSq)) {
                if (nthp1 != end and nth != end) {

                    RandomIt candNext = findClosest(target, nthp1, end, level+1, bestDistanceSq, range);

                    if (candNext != end) {
                        cand = candNext;
                    }
                }
            }

            range.lowLimits[dim] = prevRangeMin;

        } else if (planeDist >= 0) {

            if (nthp1 != end and nth != end) {

                range.lowLimits[dim] = wNode[dim];

                RandomIt candNext = findClosest(target, nthp1, end, level+1, bestDistanceSq, range);

                if (candNext != end) {
                    cand = candNext;
                }
            }

            range.lowLimits[dim] = prevRangeMin;

            range.highLimits[dim] = wNode[dim];

            if (range.circleIntersectRange(wTarget, bestDistanceSq)) {
                if (start != nth and nth != end) {

                    RandomIt candPrev = findClosest(target, start, nth, level+1, bestDistanceSq, range);

                    if (candPrev != nth) {
                        cand = candPrev;
                    }
                }
            }

            range.highLimits[dim] = prevRangeMax;
        }

        return cand;
    }


    template< class RandomIt>
    RandomIt findClosestInRange(T const& target,
                                RandomIt start,
                                RandomIt end,
                                T const& min,
                                T const& max,
                                int level,
                                PosT & bestDistanceSq) {

        WrapperT wTarget(target);
        WrapperT wMin(min);
        WrapperT wMax(max);

        int dim = getCurrentDim(level);

        int len = std::distance(start, end);
        int n = getRangeCutIdx(len);

        RandomIt nth = start;
        std::advance(nth, n);
        WrapperT wNode(*nth);

        RandomIt cand = end;

        bool inRange = true;
        for (int i = 0; i < nD; i++) {
            if (wNode[dim] < wMin[dim] or wNode[dim] > wMax[dim]) {
                inRange = false;
            }
        }

        if (inRange) {
            PosT nodeDistanceSq = computeDistSq(target, *nth);

            if (nodeDistanceSq <= bestDistanceSq) {
                bestDistanceSq = nodeDistanceSq;
                cand = nth;
            }
        }

        PosT planeDist = wTarget[dim] - wNode[dim];
        PosT planeDistSq = planeDist*planeDist;

        RandomIt nthp1 = nth;
        nthp1++;

        if (planeDist < 0) {

            if (start != nth and nth != end and wNode[dim] > wMin[dim]) {
                RandomIt candPrev = findClosestInRange(target, start, nth, min, max, level+1, bestDistanceSq);

                if (candPrev != nth) {
                    cand = candPrev;
                }
            }

            if (planeDistSq < bestDistanceSq and wNode[dim] < wMax[dim]) {
                if (nthp1 != end and nth != end) {
                    RandomIt candNext = findClosestInRange(target, nthp1, end, min, max, level+1, bestDistanceSq);

                    if (candNext != end) {
                        cand = candNext;
                    }
                }
            }

        } else if (planeDist >= 0) {

            if (nthp1 != end and nth != end and wNode[dim] < wMax[dim]) {
                RandomIt candNext = findClosestInRange(target, nthp1, end, min, max, level+1, bestDistanceSq);

                if (candNext != end) {
                    cand = candNext;
                }
            }

            if (planeDistSq < bestDistanceSq and wNode[dim] > wMin[dim]) {
                if (start != nth and nth != end) {
                    RandomIt candPrev = findClosestInRange(target, start, nth, min, max, level+1, bestDistanceSq);

                    if (candPrev != nth) {
                        cand = candPrev;
                    }
                }
            }
        }

        return cand;

    }

    template< class RandomIt>
    std::vector<T> findElementsInRange(RangeLimits const& range, RandomIt start, RandomIt end, int level = 0) {

        int dim = getCurrentDim(level);

        int len = std::distance(start, end);
        int n = getRangeCutIdx(len);

        RandomIt nth = start;
        std::advance(nth, n);
        WrapperT wNode(*nth);

        bool nodeInRange = range.pointInRange(wNode);

        std::vector<T> low;
        std::vector<T> high;

        RandomIt nthp1 = nth;
        nthp1++;

        if (start != nth and nth != end) {
            if (range.lowLimits[dim] < wNode[dim]) {
                low = findElementsInRange(range, start, nth, level+1);
            }
        }

    }

    ContainerT _data;
};

} // namespace Geometry
} // namespace StereoVision

#endif // GENERICBOUNDINGVOLUMEHIERARCHY_H
