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
#include <cstring>
#include <sstream>
#include <tuple>
#include <variant>
#include <optional>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <filesystem>

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
std::variant<
    // basic types
    int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, std::string,
    // list types
    std::vector<int8_t>, std::vector<uint8_t>, std::vector<int16_t>, std::vector<uint16_t>, std::vector<int32_t>,
    std::vector<uint32_t>, std::vector<int64_t>, std::vector<uint64_t>, std::vector<float>, std::vector<double>,
    std::vector<std::string>,
    // packet of bytes
    std::vector<std::byte>>;

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
T castedPointCloudAttribute(PointCloudGenericAttribute const& val);

// template to check if a type is a vector
template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc>> : std::true_type {};

template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

template<typename T_>
T_ castedPointCloudAttribute(PointCloudGenericAttribute const& val) {
    using T = std::decay_t<T_>;
    constexpr bool isSimpleReturnType = std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<std::string, T>;
    constexpr bool isVectorReturnType = is_vector_v<T>;

    static_assert(isSimpleReturnType or isVectorReturnType, "Target type must be a supported type!");

    if (std::holds_alternative<T>(val)) {
        return std::get<T>(val);
    }

    // to string visitor
    // any supported type can be converted to string
    auto toString = [](auto&& attr) {
        std::ostringstream strs;
        using U = std::decay_t<decltype(attr)>;
        if constexpr (std::is_same_v<U, std::string>) {
            strs << attr;
        } else if constexpr (std::is_floating_point_v<U>) {
            constexpr auto maxPrecision{std::numeric_limits<U>::digits10 + 1};
            strs << std::setprecision(maxPrecision) << attr; // write with max precision
        } else if constexpr (std::is_integral_v<U>) {
            if constexpr (std::is_signed_v<U>) { // convert to bigger type to display char type as a number
                strs << static_cast<intmax_t>(attr);
            } else {
                strs << static_cast<uintmax_t>(attr);
            }
        } else if constexpr (is_vector_v<U>) {
            using value_type = typename U::value_type;
            if constexpr (std::is_same_v<value_type, std::string>) {
                for (size_t i = 0; i < attr.size()-1; i++) { strs << attr[i] << " "; }
                if (attr.size() > 0) { strs << attr[attr.size()-1]; }
            } else if constexpr (std::is_floating_point_v<value_type>) {
                constexpr auto maxPrecision{std::numeric_limits<value_type>::digits10 + 1};
                for (size_t i = 0; i < attr.size()-1; i++) { strs << std::setprecision(maxPrecision) << attr[i] << " "; }
                if (attr.size() > 0) { strs << std::setprecision(maxPrecision) << attr[attr.size()-1]; }
            } else if constexpr (std::is_integral_v<value_type>) {
                if constexpr (std::is_signed_v<value_type>) { // convert to bigger type to display char type as a number
                    for (size_t i = 0; i < attr.size()-1; i++) { strs << static_cast<intmax_t>(attr[i]) << " "; }
                    if (attr.size() > 0) { strs << static_cast<intmax_t>(attr[attr.size()-1]); }
                } else {
                    for (size_t i = 0; i < attr.size()-1; i++) { strs << static_cast<uintmax_t>(attr[i]) << " "; }
                    if (attr.size() > 0) { strs << static_cast<uintmax_t>(attr[attr.size()-1]); }
                }
            } else if constexpr (std::is_same_v<value_type, std::byte>) {
                // print them in hex
                if (attr.size() > 0) { strs << "0x"; }
                for (const auto& byte : attr) {
                    strs << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
                }
            } else {
                static_assert(false, "Unsupported vector type");
            }
        } else if constexpr (std::is_same_v<U, std::byte>) {
            strs << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(attr);
        } else {
            static_assert(false, "Unsupported type");
        }
        return strs.str();
    };

    if constexpr (std::is_same_v<T, std::string>) {
        return std::visit(toString, val);
    } else {
        // redundant assertion in case we add other types
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T> || isVectorReturnType,
            "Target type should be an integral, a floating point number or a vector at this stage");

        T ret = std::visit([&] (auto&& arg) {
            using variantHeld_t = std::decay_t<decltype(arg)>; // the type inside the alternative
            constexpr bool isSimpleHeldType = std::is_integral_v<variantHeld_t> || std::is_floating_point_v<variantHeld_t>
                || std::is_same_v<std::string, variantHeld_t>;
            const bool isVectorHeldType = is_vector_v<variantHeld_t>;
            if constexpr (isSimpleHeldType && isSimpleReturnType) { //* simple type => simple type conversion
                if constexpr (std::is_same_v<std::string, variantHeld_t>) {
                    double tmp = std::stod(arg);
                    return static_cast<T>(tmp);
                } else {
                    return static_cast<T>(arg);
                }
            } else if constexpr (isSimpleHeldType && isVectorReturnType) { //* simple type => vector conversion
                using returnVectorValue_t = typename T::value_type;
                if constexpr (std::is_same_v<std::string, variantHeld_t>) {
                    // try to interpret it as a vector
                    T vec;
                    std::stringstream ss(arg);
                    std::string token;
                    while (ss >> token) {
                        if (ss.fail()) {
                            return T{};
                        }
                        if constexpr (std::is_same_v<returnVectorValue_t, std::string>) {
                            vec.push_back(token);
                        } else if constexpr (std::is_arithmetic_v<returnVectorValue_t>) {
                            double tmp = std::stod(token);
                            vec.push_back(static_cast<returnVectorValue_t>(tmp));
                        } else {
                            return T{};
                        }
                    }
                    return vec;
                } else if constexpr (std::is_same_v<returnVectorValue_t, std::string>) {
                    return T{toString(arg)};
                } else if constexpr (std::is_convertible_v<variantHeld_t, returnVectorValue_t>) { // try to wrap it
                        return T{static_cast<returnVectorValue_t>(arg)};
                } else if constexpr(std::is_convertible_v<variantHeld_t, T>) {
                    return static_cast<T>(arg);
                } else {
                    return T{};
                }
            } else if constexpr (isVectorHeldType && isVectorReturnType) { //* vector => vector conversion
                using returnVectorValue_t = typename T::value_type;
                using variantVectorValue_t = std::remove_cv_t<std::remove_reference_t<typename variantHeld_t::value_type>>;
                
                 if constexpr(std::is_convertible_v<variantVectorValue_t, returnVectorValue_t>) {
                    T vec;
                    vec.reserve(arg.size());
                    for(auto&& e: arg) {
                        vec.push_back(static_cast<returnVectorValue_t>(e));
                    }
                    return vec;
                } else if constexpr (std::is_same_v<returnVectorValue_t, std::string>) {
                    T vec;
                    vec.reserve(arg.size());
                    for(auto&& e: arg) {
                        vec.push_back(toString(e));
                    }
                    return vec;
                } else if constexpr (std::is_same_v<std::string, variantVectorValue_t>) {
                    T vec;
                    vec.reserve(arg.size());
                    for (auto&& e : arg) {
                        double tmp = std::stod(e);
                        vec.push_back(static_cast<returnVectorValue_t>(tmp));
                    }
                    return vec;
                } else if constexpr ((std::is_same_v<std::byte, variantVectorValue_t> or std::is_same_v<std::byte, returnVectorValue_t>)
                                     and sizeof(variantVectorValue_t) == sizeof(returnVectorValue_t)
                                     and (std::is_integral_v<variantVectorValue_t> or std::is_integral_v<returnVectorValue_t>)) {
                    // reinterpret the bytes
                    T vec;
                    vec.resize(arg.size());
                    for(auto i = 0; i < arg.size(); i++) {
                        std::memcpy(&vec[i], &arg[i], sizeof(returnVectorValue_t));
                    }
                    return vec;
                } else {
                    return T{}; // empty vector
                }
            } else if constexpr (isVectorHeldType && isSimpleReturnType) { //* vector => simple type conversion
                if (arg.empty()) return T{};
                // case vector => string is already handled above
                using variantVectorValue_t = std::remove_cv_t<std::remove_reference_t<typename variantHeld_t::value_type>>;
                // it might be possible to convert it by taking the first element
                if constexpr (std::is_same_v<T, variantVectorValue_t>) {
                    return arg[0];
                } else if constexpr (std::is_same_v<std::string, variantVectorValue_t>) {
                    // vector => number
                    double tmp = std::stod(arg[0]);
                    return static_cast<T>(tmp);
                } else if constexpr (std::is_convertible_v<variantVectorValue_t, T>) {
                    return static_cast<T>(arg[0]);
                } else {
                    return T{};
                }
            } else {
                // this should not happen
                static_assert(false, "Not all possible types are accepted by the visitor.");
            }
            return T{};
        }, val);
        return ret;
    }
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
    std::optional<PtColor<Color_T>> castedPointColor() const {
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

/**
 * @brief
 *
 * Open a file and returns a FullPointCloudAccessInterface.
 * The extension of the file is used to determine the type of the point cloud.
 *
 * @param filePath The path to the file containing the point cloud
 *
 * @return A FullPointCloudAccessInterface containing the header and the points.
 * If the file can't be opened, an empty optional is returned.
 */
std::optional<FullPointCloudAccessInterface> openPointCloud(const std::filesystem::path& filePath);

} // namespace IO
} // namespace StereoVision

#endif // STEREOVISION_IO_POINTCLOUD_IO_H
