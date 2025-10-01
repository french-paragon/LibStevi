#ifndef GENERICBOUNDINGVOLUMEHIERARCHY_H
#define GENERICBOUNDINGVOLUMEHIERARCHY_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024-2025 Paragon<french.paragon@gmail.com>

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
#include <functional>
#include <optional>
#include <set>

namespace StereoVision {
namespace Geometry {


/*!
 * \brief The BSPObjectWrapper class acts as a wrapper for an object that will go into a BSP
 *
 * A BSPObjectWrapper is expected to provide an operator[] with an int parameter representing the dimension.
 * The operator[] must return a PosT (default = float) representing the coordinate of the object in space.
 * If operator[] return a GenericBSP::SearchRange, then the object is already expected to represent an Axis Aligned Bounding Box
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
 * The class takes ownership of a collection of objects of arbitrary types contained in an arbitrary container class.
 * It provide a spatial ordering in arbitrary number of dimensions.
 * Construction support move semantic for optimal data transfer, or full copy of the data.
 * A function to get a range given an object and a dimension has to be provided.
 * Optionally, additional functions (e.g. if a point is within the object, if a ray interesect the object, ...) can be provided.
 */
template<typename T, int nD, typename WT = BSPObjectWrapper<T,float>, typename CT = std::vector<T>>
class GenericBSP {
public:

    static constexpr int nDims = nD;

    typedef CT ContainerT;
    typedef WT WrapperT;
    typedef typename WrapperT::PosT PosT;

    static_assert(std::is_arithmetic<PosT>::value, "Invalid type used in GenericBSP");

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

/*!
 * \brief The GenericBVH class represent a bounding volume hierarchy (BVH) tree for arbitrary type T
 *
 * The class takes ownership of a collection of objects of arbitrary types contained in an arbitrary container class
 * It provide spatial ordering in arbitrary number of dimensions.
 * A wapper class has to be provided to extract the bounding boxes of the objects
 */
template<typename T, int nD, typename PT = float, typename CT = std::vector<T>>
class GenericBVH {

public:
    static constexpr int nDims = nD;

    typedef CT ContainerT;
    typedef PT PosT;

    struct Range {
        PosT min;
        PosT max;

        inline bool isEmpty() const {
            return min >= max;
        }

        inline Range intersection(Range const& other) {
            return Range{std::max(min, other.min), std::min(max, other.max)};
        }

        inline Range join(Range const& other) {
            if (isEmpty()) {
                return other;
            }
            if (other.isEmpty()) {
                return *this;
            }
            return Range{std::min(min, other.min), std::max(max, other.max)};
        }

        inline PosT extent() const {
            return max - min;
        }
    };

    using GenericPoint = std::array<PosT, nDims>;
    using GenericVec = std::array<PosT, nDims>;

    using RangeFunc = std::function<Range(T const& obj, int dim)>;
    using ContainPointFunc = std::function<bool(T const& obj, GenericPoint const& pt)>;
    using RayIntersectFunc = std::function<std::optional<GenericPoint>(T const& obj, GenericPoint const& origin, GenericVec const& direction)>;

    GenericBVH(ContainerT const& container,
               RangeFunc const& rangeFunc,
               ContainPointFunc const& containPointFunc = ContainPointFunc(),
               RayIntersectFunc const& rayIntersectFunc = RayIntersectFunc()) :
        _rangeFunc(rangeFunc),
        _pointFunc(containPointFunc),
        _rayFunc(rayIntersectFunc),
        _data(container)
    {
        rebuildHierarchy();
    }

    GenericBVH(ContainerT && container,
               RangeFunc const& rangeFunc,
               ContainPointFunc const& containPointFunc = ContainPointFunc(),
               RayIntersectFunc const& rayIntersectFunc = RayIntersectFunc()) :
        _rangeFunc(rangeFunc),
        _pointFunc(containPointFunc),
        _rayFunc(rayIntersectFunc),
        _data(container)
    {
        rebuildHierarchy();
    }

    ~GenericBVH() {
        if (_hierarchyRoot != nullptr) {
            delete _hierarchyRoot;
        }
    }

    /*!
     * \brief ContainPoint indicate if the object contain a given point
     * \param point the point (must be of a class with an operator[](int) and the correct number of dimensions.
     * \return true if the object contain the point.
     *
     * By default, the object is treated as an axis aligned box, unless a specific ContainPointFunc has been provided
     */
    template <typename PtT>
    inline bool containPoint(T const& obj, PtT const& point) {

        if (_pointFunc) {

            GenericPoint pt;

            for (int i = 0; i < nDims; i++) {
                pt[i] = point[i];
            }

            return _pointFunc(obj, pt);
        }

        for (int i = 0; i < nDims; i++) {
            Range range = _rangeFunc(obj, i);
            if (point[i] < range.min or point[i] > range.max) {
                return false;
            }
        }

        return true;
    }

    /*!
     * \brief RayIntersect indicate if a ray interesect the object.
     * \param obj the object under consideration
     * \param origin the origin of the ray
     * \param direction the direction of the ray
     * \return the point of intersection, as a GenericPoint, or nullopt if there is no intersection
     *
     * By default, the object is treated as an axis aligned box, unless a specific ContainPointFunc has been provided
     */
    template <typename OT, typename DT>
    inline std::optional<GenericPoint> rayIntersect(T const& obj, OT const& origin, DT const& direction) {

        if (_rayFunc) {

            GenericPoint orig;
            GenericVec dir;

            for (int i = 0; i < nDims; i++) {
                orig[i] = origin[i];
                dir[i] = direction[i];
            }

            return _rayFunc(obj, orig, dir);
        }

        Range rayRange{0, std::numeric_limits<PosT>::infinity()};

        for (int i = 0; i < nDims; i++) {
            Range range = _rangeFunc(obj, i);

            Range localRayRange;

            localRayRange.min = (range.min - origin[i])/direction[i];
            localRayRange.max = (range.max - origin[i])/direction[i];

            rayRange = rayRange.intersection(localRayRange);
        }

        if (rayRange.isEmpty()) {
            return std::nullopt;
        }

        GenericPoint intersection;

        for (int i = 0; i < nDims; i++) {
            intersection[i] = origin[i] + rayRange.min*direction[i];
        }

        return intersection;
    }


    template <typename PtT>
    inline std::vector<int> itemsContainingPoint(PtT const& point) {
        std::set<int> idxs;

        checkItemsIntersectingPoints(_hierarchyRoot, idxs, point);

        return std::vector<int>(idxs.begin(), idxs.end());
    }

    struct RayIntersection {
        GenericPoint point;
        int objIdx;
    };

    template <typename OT, typename DT>
    inline std::optional<RayIntersection> rayIntersection(OT const& origin, DT const& direction) {
        return rayIntersection(_hierarchyRoot, origin, direction);
    }

protected:

    struct GroupInfos {

        ~GroupInfos() {
            if (subGroup1 != nullptr) {
                delete subGroup1;
            }
            if (subGroup2 != nullptr) {
                delete subGroup2;
            }
        }

        std::array<Range, nDims> ranges;
        GroupInfos* subGroup1;
        GroupInfos* subGroup2;
        int subObj1Id;
        int subObj2Id;
    };

    template <typename PtT>
    inline bool ContainPoint(GroupInfos const& groupInfos, PtT const& point) {

        for (int i = 0; i < nDims; i++) {
            Range const& range = groupInfos.ranges[i];
            if (point[i] < range.min or point[i] > range.max) {
                return false;
            }
        }

        return true;
    }

    template <typename OT, typename DT>
    inline bool rayIntersect(GroupInfos const& groupInfos, OT const& origin, DT const& direction) {

        Range rayRange{0, std::numeric_limits<PosT>::infinity()};

        for (int i = 0; i < nDims; i++) {
            Range range = groupInfos.ranges[i];

            Range localRayRange;

            localRayRange.min = (range.min - origin[i])/direction[i];
            localRayRange.max = (range.max - origin[i])/direction[i];

            rayRange = rayRange.intersection(localRayRange);
        }

        if (rayRange.isEmpty()) {
            return false;
        }

        return true;
    }

    GroupInfos* buildGroupInfos(int* objIdxs, int nIdxs) {

        if (nIdxs <= 0) {
            return nullptr;
        }

        if (nIdxs == 1 or nIdxs == 2) {
            GroupInfos* infos = new GroupInfos();

            infos->subGroup1 = nullptr;
            infos->subGroup1 = nullptr;

            infos->subObj1Id = objIdxs[0];
            infos->subObj2Id = (nIdxs == 2) ? objIdxs[1] : objIdxs[0];

            for (int i = 0; i < nDims; i++) {

                infos->ranges[i] = _rangeFunc(_data[infos->subObj1Id], i).join(_rangeFunc(_data[infos->subObj2Id], i));
            }

            return infos;
        }

        GroupInfos* infos = new GroupInfos();

        std::array<Range, nDims>& ranges = infos->ranges;

        for (int i = 0; i < nDims; i++) {
            for (int j = 0; j < nIdxs; j++) {
                ranges[i] = _rangeFunc(_data[objIdxs[j]], i).join((j == 0) ? Range{1,-1} : ranges[i]);
            }
        }


        int currentDim = 0;
        PosT currentRange = ranges[0].extent();

        for (int i = 1; i < nDims; i++) {
            PosT extent = ranges[i].extent();
            if (extent > currentRange) {
                currentDim = i;
                currentRange = extent;
            }
        }


        std::sort(objIdxs, objIdxs + nIdxs, [this, currentDim] (int i1, int i2) {
            Range range1 = _rangeFunc(_data[i1], currentDim);
            Range range2 = _rangeFunc(_data[i2], currentDim);

            return range1.min + range1.max < range2.min + range2.max;
        });

        int n1 = nIdxs/2;
        int n2 = nIdxs - n1;

        infos->subGroup1 = buildGroupInfos(objIdxs, n1);
        infos->subGroup2 = buildGroupInfos(objIdxs + n1, n2);

        infos->subObj1Id = -1;
        infos->subObj2Id = -1;

        return infos;


    }

    inline void rebuildHierarchy() {
        std::vector<int> idxs(_data.size());

        for (int i = 0; i < idxs.size(); i++) {
            idxs[i] = i;
        }

        _hierarchyRoot = buildGroupInfos(idxs.data(), idxs.size());
    }

    template <typename PtT>
    void checkItemsIntersectingPoints(GroupInfos* group, std::set<int> & items, PtT const& point) {
        if (!ContainPoint(*group, point)) {
            return;
        }

        if (group->subGroup1 != nullptr) {
            checkItemsIntersectingPoints(group->subGroup1, items, point);
        }

        if (group->subGroup2 != nullptr) {
            checkItemsIntersectingPoints(group->subGroup2, items, point);
        }

        if (group->subObj1Id >= 0) {
            if (containPoint(_data[group->subObj1Id], point)) {
                items.insert(group->subObj1Id);
            }
        }

        if (group->subObj2Id >= 0) {
            if (containPoint(_data[group->subObj2Id], point)) {
                items.insert(group->subObj2Id);
            }
        }
    }

    template <typename OT>
    std::optional<RayIntersection> selectRayIntersection(std::optional<RayIntersection> const& opt1,
                                                         std::optional<RayIntersection> const& opt2,
                                                         OT const& origin) {
        if (opt1.has_value()) {

            if (opt2.has_value()) {

                PosT d1 = 0;
                PosT d2 = 0;

                for (int i = 0; i < nDims; i++) {
                    PosT e1 = opt1.value().point[i] - origin[i];
                    d1 += e1*e1;
                    PosT e2 = opt2.value().point[i] - origin[i];
                    d2 += e2*e2;
                }

                if (d1 < d2) {
                    return opt1;
                } else {
                    return opt2;
                }

            } else {
                return opt1;
            }

        }

        return opt2;
    }

    template <typename OT, typename DT>
    inline std::optional<RayIntersection> rayIntersection(GroupInfos* group, OT const& origin, DT const& direction) {

        if (!rayIntersect(*group, origin, direction)) {
            return std::nullopt;
        }

        std::optional<RayIntersection> subGroup1 = std::nullopt;
        std::optional<RayIntersection> subGroup2 = std::nullopt;

        if (group->subGroup1 != nullptr) {
            subGroup1 = rayIntersection(group->subGroup1, origin, direction);
        }

        if (group->subGroup2 != nullptr) {
            subGroup2 = rayIntersection(group->subGroup2, origin, direction);
        }

        std::optional<RayIntersection> subItem1 = std::nullopt;
        std::optional<RayIntersection> subItem2 = std::nullopt;

        if (group->subObj1Id >= 0) {
            std::optional<GenericPoint> intersection = rayIntersect(_data[group->subObj1Id], origin, direction);
            if (intersection.has_value()) {
                subItem1 = {intersection.value(), group->subObj1Id};
            }
        }

        if (group->subObj2Id >= 0) {
            std::optional<GenericPoint> intersection = rayIntersect(_data[group->subObj2Id], origin, direction);
            if (intersection.has_value()) {
                subItem2 = {intersection.value(), group->subObj2Id};
            }
        }

        std::optional<RayIntersection> selected = std::nullopt;
        selected = selectRayIntersection(selected, subGroup1, origin);
        selected = selectRayIntersection(selected, subGroup2, origin);
        selected = selectRayIntersection(selected, subItem1, origin);
        selected = selectRayIntersection(selected, subItem2, origin);

        return selected;
    }

    RangeFunc _rangeFunc;
    ContainPointFunc _pointFunc;
    RayIntersectFunc _rayFunc;

    ContainerT _data;
    GroupInfos* _hierarchyRoot;

};

} // namespace Geometry
} // namespace StereoVision

#endif // GENERICBOUNDINGVOLUMEHIERARCHY_H
