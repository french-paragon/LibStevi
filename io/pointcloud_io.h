#ifndef STEREOVISION_IO_POINTCLOUD_IO_H
#define STEREOVISION_IO_POINTCLOUD_IO_H

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

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <tuple>
#include <variant>
#include <optional>
#include <memory>
#include <algorithm>
#include <type_traits>

namespace StereoVision {
namespace IO {

template<typename Geometry_T>
struct PtGeometry{
    Geometry_T x;
    Geometry_T y;
    Geometry_T z;
};

template<typename Color_T>
struct PtColor{
    Color_T r;
    Color_T g;
    Color_T b;
    Color_T a;
};

template<>
struct PtColor<void>{
};

using PointCloudGenericAttribute =
std::variant<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, std::string,
std::vector<int8_t>, std::vector<uint8_t>, std::vector<int16_t>, std::vector<uint16_t>, std::vector<int32_t>,
std::vector<uint32_t>, std::vector<int64_t>, std::vector<uint64_t>, std::vector<float>, std::vector<double>, std::vector<std::string>>;

inline bool isAttributeList(PointCloudGenericAttribute const& val) {
    return std::holds_alternative<std::vector<int8_t>>(val) or
           std::holds_alternative<std::vector<uint8_t>>(val) or
           std::holds_alternative<std::vector<int16_t>>(val) or
           std::holds_alternative<std::vector<uint16_t>>(val) or
           std::holds_alternative<std::vector<int32_t>>(val) or
           std::holds_alternative<std::vector<uint32_t>>(val) or
           std::holds_alternative<std::vector<int64_t>>(val) or
           std::holds_alternative<std::vector<uint64_t>>(val) or
           std::holds_alternative<std::vector<float>>(val) or
           std::holds_alternative<std::vector<double>>(val) or
           std::holds_alternative<std::vector<std::string>>(val);
}

template<typename T>
T castedPointCloudAttribute(PointCloudGenericAttribute const& val) {

    static_assert (std::is_integral_v<T> or std::is_floating_point_v<T> or std::is_same_v<std::string, T>,
            "Target type must be a supported type!");

    if (std::holds_alternative<T>(val)) {
        return std::get<T>(val);
    }

    if (std::is_same_v<T, std::string>) {
        std::stringstream strs;
        std::visit([&strs] (auto&& arg) {strs << arg;}, val);
        return strs.str();
    }

    if (std::holds_alternative<std::string>(val)) {
        double tmp = std::stod(std::get<std::string>(val));
        return static_cast<T>(tmp);
    }

    T ret = std::visit([] (auto&& arg) {return static_cast<T>(arg);}, val);
    return ret;

}

class PointCloudHeaderInterface {
public:

    inline PointCloudHeaderInterface() {};
    inline virtual ~PointCloudHeaderInterface() {};

    virtual std::optional<PointCloudGenericAttribute> getAttributeById(int id) const = 0;
    virtual std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const = 0;

    inline std::optional<PointCloudGenericAttribute> operator[](int id) {
        return getAttributeById(id);
    }

    inline std::optional<PointCloudGenericAttribute> operator[](const char* attributeName) {
        return getAttributeByName(attributeName);
    }

    virtual std::vector<std::string> attributeList() const = 0;

};

/*!
 * \brief The PointCloudPointAccessInterface class represent a generic reader for a point in a point cloud.
 *
 * The idead is that a function reading a point cloud return an iterable sequence of such reader.
 */
class PointCloudPointAccessInterface {
public:

    inline PointCloudPointAccessInterface() {};
    inline virtual ~PointCloudPointAccessInterface() {};

    virtual PtGeometry<PointCloudGenericAttribute> getPointPosition() const = 0;
    virtual std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const = 0;

    virtual std::optional<PointCloudGenericAttribute> getAttributeById(int id) const = 0;
    virtual std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const = 0;

    inline std::optional<PointCloudGenericAttribute> operator[](int id) {
        return getAttributeById(id);
    }

    inline std::optional<PointCloudGenericAttribute> operator[](const char* attributeName) {
        return getAttributeByName(attributeName);
    }

    virtual std::vector<std::string> attributeList() const = 0;

    /*!
     * \brief getNext get the next point in the point cloud
     * \return true if data is still available, false if the end of the data is reached.
     */
    virtual bool gotoNext() = 0;

    template<typename Geometry_T>
    PtGeometry<Geometry_T> castedPointGeometry() const {
        static_assert (std::is_integral_v<Geometry_T> or std::is_floating_point_v<Geometry_T>, "Geometry type needs to be an integral or floating point type");
        PtGeometry<Geometry_T> ret;
        PtGeometry<PointCloudGenericAttribute> raw = getPointPosition();

        ret.x = castedPointCloudAttribute<Geometry_T>(raw.x);
        ret.y = castedPointCloudAttribute<Geometry_T>(raw.y);
        ret.z = castedPointCloudAttribute<Geometry_T>(raw.z);

        return ret;
    }

    template<typename Color_T>
    std::optional<PtGeometry<Color_T>> castedPointColor() const {
        static_assert (std::is_integral_v<Color_T> or std::is_floating_point_v<Color_T>, "Color type needs to be an integral or floating point type");

        std::optional<PtColor<PointCloudGenericAttribute>> raw_opt = getPointColor();

        if (!raw_opt.has_value()) {
            return std::nullopt;
        }

        PtColor<Color_T> ret;
        PtColor<PointCloudGenericAttribute>& raw = raw_opt.value();

        ret.r = castedPointCloudAttribute<Color_T>(raw.r);
        ret.g = castedPointCloudAttribute<Color_T>(raw.g);
        ret.b = castedPointCloudAttribute<Color_T>(raw.b);
        ret.a = castedPointCloudAttribute<Color_T>(raw.a);

        return ret;
    }

};

/*!
 * \brief The FullPointCloudAccessInterface struct represent the access interface for a full point cloud.
 *
 * This struct holds a points to the header interface and a pointer to the points interface.
 * When the struct is deleted, it deletes the pointers as well.
 *
 * This struct can be constructed, moved but cannot be copied.
 */
struct FullPointCloudAccessInterface {

public:

    FullPointCloudAccessInterface(PointCloudHeaderInterface* header = nullptr, PointCloudPointAccessInterface* points = nullptr);
    //make the struct move constructible
    FullPointCloudAccessInterface(FullPointCloudAccessInterface && other);

    std::unique_ptr<PointCloudHeaderInterface> headerAccess;
    std::unique_ptr<PointCloudPointAccessInterface> pointAccess;
};

/*!
 * \brief The GenericPointCloud class represent a generic implementation to store a point cloud in memory
 */
template<typename Geometry_T, typename Color_T>
class GenericPointCloud {

public:

    struct Point {
        PtGeometry<Geometry_T> xyz;
        PtColor<Color_T> rgba;
        std::map<std::string, PointCloudGenericAttribute> attributes;

        /*!
         * \brief operator [] shortcut to access the geometry
         * \param i the axis index (0 = x, 1 = y, 2 = z)
         * \return the value of the point coordinate for the given axis.
         */
        inline Geometry_T operator[](int i) const {
            return (i == 0) ? xyz.x : (i == 1) ? xyz.y : xyz.z;
        }
    };

    inline GenericPointCloud() {

    }
    inline GenericPointCloud(GenericPointCloud const& other) :
        _global_attributes(other._global_attributes),
        _attributes(other._attributes),
        _points(other._points)
    {

    }
    inline GenericPointCloud(GenericPointCloud && other) :
        _global_attributes(std::move(other._global_attributes)),
        _attributes(std::move(other._attributes)),
        _points(std::move(other._points))
    {

    }

    inline typename std::vector<Point>::iterator begin() {
        return _points.begin();
    }

    inline typename std::vector<Point>::iterator end() {
        return _points.end();
    }

    inline typename std::vector<Point>::const_iterator begin() const {
        return _points.begin();
    }

    inline typename std::vector<Point>::const_iterator end() const {
        return _points.end();
    }

    inline Point& operator[](int i) {
        return _points[i];
    }

    inline Point& operator[](int i) const {
        return _points[i];
    }

    std::vector<std::string> const& attributes() const { return _attributes; }

    inline void addAttribute(std::string const& name) {

        for (std::string const& attr : _attributes) {
            if (attr == name) {
                return; //attribute already present
            }
        }

        _attributes.push_back(name);
    }

    inline void clearAttribute(std::string const& name) {

        std::vector<std::string>::iterator position = std::find(_attributes.begin(), _attributes.end(), name);

        if (position == _attributes.end()) {
            return;
        }

        _attributes.erase(position);

        for (Point& pt : _points) {
            if (pt.attributes.count(name) > 0) {
                pt.attributes.erase(name);
            }
        }
    }

    inline void addPoint(Point const& pt) {
        _points.push_back(pt);
    }

    inline void removePoint(int i) {
        _points.erase(_points.begin() + i);
    }

    inline PointCloudGenericAttribute& globalAttribute(std::string const& name) {
        return _global_attributes[name];
    }

    inline PointCloudGenericAttribute globalAttribute(std::string const& name) const {
        return _global_attributes.at(name);
    }

    inline PointCloudGenericAttribute nthGlobalAttribute(int i) const {
        auto iterator = _global_attributes.begin();
        std::advance(iterator, i);
        return iterator->first;
    }

    inline bool hasGlobalAttribute(std::string const& name) {
        return _global_attributes.count(name) > 0;
    }

    inline void clearGlobalAttribute(std::string const& name) {
        _global_attributes.erase(name);
    }

    inline std::vector<std::string> listGlobalAttributes() const {
        std::vector<std::string> ret;
        ret.reserve(_global_attributes.size());

        for (const auto & [key, val] : _global_attributes) {
            ret.push_back(key);
        }
        return ret;
    }

protected:


    std::map<std::string, PointCloudGenericAttribute> _global_attributes;

    std::vector<std::string> _attributes;
    std::vector<Point> _points;


};

template<typename Geometry_T, typename Color_T>
class GenericPointCloudHeaderInterface : public PointCloudHeaderInterface {
public:

    GenericPointCloudHeaderInterface(GenericPointCloud<Geometry_T, Color_T> const& point_cloud) :
        PointCloudHeaderInterface(),
        _point_cloud(&point_cloud)
    {

    };
    virtual ~GenericPointCloudHeaderInterface() {};

    virtual std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override {
        return _point_cloud->nthGlobalAttribute(id);
    }
    virtual std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override {
        return _point_cloud->globalAttribute(attributeName);
    }

    virtual std::vector<std::string> attributeList() const override {
        return _point_cloud->listGlobalAttributes();
    }

protected:

    const GenericPointCloud<Geometry_T, Color_T>* _point_cloud;

};

template<typename Geometry_T, typename Color_T>
class GenericPointCloudPointAccessInterface : public PointCloudPointAccessInterface {
public:

    inline GenericPointCloudPointAccessInterface(GenericPointCloud<Geometry_T, Color_T> const& point_cloud) :
        PointCloudPointAccessInterface(),
        _point_cloud(&point_cloud)
    {
        _iterator = _point_cloud->begin();
    };
    inline virtual ~GenericPointCloudPointAccessInterface() {};

    virtual PtGeometry<PointCloudGenericAttribute> getPointPosition() const override {
        PtGeometry<PointCloudGenericAttribute> ret;
        ret.x = _iterator->xyz.x;
        ret.y = _iterator->xyz.y;
        ret.z = _iterator->xyz.z;
        return ret;
    }
    virtual std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override {

        if (std::is_same_v<Color_T, void>) {
            return std::nullopt;
        }

        PtColor<PointCloudGenericAttribute> ret;

        //just use this trick, since PtColor<void> containts no fields
        using CT = std::conditional_t<std::is_same_v<Color_T, void>,float,Color_T>;

        const PtColor<CT>* ptr = reinterpret_cast<const PtColor<CT>*>(&_iterator->rgba);

        ret.r = ptr->r;
        ret.g = ptr->g;
        ret.b = ptr->b;
        ret.a = ptr->a;

        return ret;
    }

    virtual std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override {
        std::string attributeName = _point_cloud->attributes()[id];
        return GenericPointCloudPointAccessInterface<Geometry_T, Color_T>::getAttributeByName(attributeName.c_str());
    }
    virtual std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override {
        if (_iterator->attributes.count(attributeName) <= 0) {
            return std::nullopt;
        }
        return _iterator->attributes.at(attributeName);
    }

    virtual std::vector<std::string> attributeList() const override {
        return _point_cloud->attributes();
    }

    virtual bool gotoNext() override {
        _iterator++;

        if (_iterator == _point_cloud->end()) {
            return false;
        }

        return true;
    }

protected:

    typename std::vector<typename GenericPointCloud<Geometry_T, Color_T>::Point>::const_iterator _iterator;
    const GenericPointCloud<Geometry_T, Color_T>* _point_cloud;
};

} // namespace IO
} // namespace StereoVision

#endif // STEREOVISION_IO_POINTCLOUD_IO_H
