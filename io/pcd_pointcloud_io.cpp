#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <numeric>
#include <optional>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "pcd_pointcloud_io.h"
#include "pointcloud_io.h"


namespace StereoVision {
namespace IO {

    static std::ostream &operator<<(std::ostream &os, const PcdDataStorageType &data);
    static std::istream &operator>>(std::istream &is, PcdDataStorageType &data);
    template <class ContainerT>
    static std::string joinToString(const ContainerT& c);
    static std::unique_ptr<PcdPointCloudHeader> readPcdHeader(std::ifstream& reader);
    static bool getNextHeaderLine(std::ifstream& reader, std::string& line, std::vector<std::string>& lineSplit, std::stringstream& lineStream);

PcdPointCloudPoint::PcdPointCloudPoint(std::unique_ptr<std::ifstream> reader,
    const std::vector<std::string>& attributeNames, const std::vector<size_t>& fieldByteSize,
    const std::vector<uint8_t>& fieldType, const std::vector<size_t>& fieldCount, PcdDataStorageType dataStorageType):
    attributeNames{attributeNames}, fieldByteSize{fieldByteSize},
    fieldOffset(fieldByteSize.size()), fieldType{fieldType}, fieldCount{fieldCount},
    dataStorageType{dataStorageType}, reader{std::move(reader)}
{
    const auto* fieldCountPtr = fieldCount.data();
    const auto* fieldByteSizePtr = fieldByteSize.data();
    // compute the offsets with transform_exclusive_scan
    std::transform_exclusive_scan(fieldByteSize.begin(), fieldByteSize.end(), fieldOffset.begin(), size_t{0}, std::plus<>{},
    [fieldCountPtr, fieldByteSizePtr](const auto& size) {
        auto i = std::distance(fieldByteSizePtr, &size);
        return fieldCountPtr[i] * size;
    });

    // compute the size of a "record", which is the last offset + the last size times the last count
    recordByteSize = fieldOffset.back() + fieldCount.back() * fieldByteSize.back();
    // resize the data
    dataBuffer.resize(recordByteSize);

    // find the index of the rgba field
    auto it = std::find(attributeNames.begin(), attributeNames.end(), "rgba");
    if (it != attributeNames.end()) {
        rgbaIndex = std::distance(attributeNames.begin(), it);
        containsColor = true;
    } else {
        it = std::find(attributeNames.begin(), attributeNames.end(), "rgb");
        if (it != attributeNames.end()) {
            rgbaIndex = std::distance(attributeNames.begin(), it);
            containsColor = true;
        }
    }
    // find the index of the x field
    it = std::find(attributeNames.begin(), attributeNames.end(), "x");
    if (it != attributeNames.end()) {
        xIndex = std::distance(attributeNames.begin(), it);
        containsPosition = true;
    }
    // find the index of the y field
    it = std::find(attributeNames.begin(), attributeNames.end(), "y");
    if (it != attributeNames.end()) {
        yIndex = std::distance(attributeNames.begin(), it);
        containsPosition = true;
    }
    // find the index of the z field
    it = std::find(attributeNames.begin(), attributeNames.end(), "z");
    if (it != attributeNames.end()) {
        zIndex = std::distance(attributeNames.begin(), it);
        containsPosition = true;
    }
}

PtGeometry<PointCloudGenericAttribute> PcdPointCloudPoint::getPointPosition() const
{
    const auto nan = std::nan("");
    if (!containsPosition) {
        return PtGeometry<PointCloudGenericAttribute>{nan, nan, nan};
    } else {
        return PtGeometry<PointCloudGenericAttribute>{getAttributeById(xIndex).value_or(nan), getAttributeById(yIndex).value_or(nan), getAttributeById(zIndex).value_or(nan)};
    }
}

std::optional<PtColor<PointCloudGenericAttribute>> PcdPointCloudPoint::getPointColor() const
{
    if (!containsColor) {
        return std::nullopt;
    }
    auto rgba_opt = getAttributeById(rgbaIndex);
    if (!rgba_opt.has_value()) {
        return std::nullopt;
    }
    const float rgba_float = std::get<float>(rgba_opt.value());
    const uint32_t rgba = *reinterpret_cast<const uint32_t*>(&rgba_float);
    const uint8_t a = (rgba >> 24)  & 0x000000FF;
    const uint8_t r = (rgba >> 16)  & 0x000000FF;
    const uint8_t g = (rgba >> 8)   & 0x000000FF;
    const uint8_t b =  rgba         & 0x000000FF;
    return PtColor<PointCloudGenericAttribute>{r, g, b, a};
}

std::optional<PointCloudGenericAttribute> PcdPointCloudPoint::getAttributeById(int id) const
{
    static_assert(sizeof(float) == 4); // check if float is 4 bytes, should be true on most systems
    static_assert(sizeof(double) == 8); // check if double is 8 bytes

    const auto size = fieldByteSize[id];
    const auto type = fieldType[id];
    const auto offset = fieldOffset[id];
    const auto count = fieldCount[id];

    // test if id is valid
    if (id < 0 || id >= fieldByteSize.size() || size + offset > recordByteSize) {
        return std::nullopt;
    }
    const auto* const position = dataBuffer.data() + fieldOffset[id];
    
    // lambda function that return a vector of type T or T if count == 1
    auto returnVectorOrSingleValue = [](const auto* dataPtr, auto count) -> PointCloudGenericAttribute {
        if (count == 1) return {*dataPtr};
        return std::vector(dataPtr, dataPtr + count);
    };
    // test the type and size
    if (type == 'F') {
        if (size == 4) return returnVectorOrSingleValue(reinterpret_cast<const float*>(position), count);
        if (size == 8) return returnVectorOrSingleValue(reinterpret_cast<const double*>(position), count);
    } else if (type == 'I') {
        if (size == 1) return returnVectorOrSingleValue(reinterpret_cast<const int8_t*>(position), count);
        if (size == 2) return returnVectorOrSingleValue(reinterpret_cast<const int16_t*>(position), count);
        if (size == 4) return returnVectorOrSingleValue(reinterpret_cast<const int32_t*>(position), count);
        if (size == 8) return returnVectorOrSingleValue(reinterpret_cast<const int64_t*>(position), count);
    } else if (type == 'U') {
        if (size == 1) return returnVectorOrSingleValue(reinterpret_cast<const uint8_t*>(position), count);
        if (size == 2) return returnVectorOrSingleValue(reinterpret_cast<const uint16_t*>(position), count);
        if (size == 4) return returnVectorOrSingleValue(reinterpret_cast<const uint32_t*>(position), count);
        if (size == 8) return returnVectorOrSingleValue(reinterpret_cast<const uint64_t*>(position), count);
    }
    return std::nullopt;
}

std::optional<PointCloudGenericAttribute> PcdPointCloudPoint::getAttributeByName(const char *attributeName) const
{
    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    if (it != attributeNames.end()) {
        return getAttributeById(std::distance(attributeNames.begin(), it));
    }
    return std::nullopt; // Attribute not found
}

std::vector<std::string> PcdPointCloudPoint::attributeList() const
{
    return attributeNames;
}

bool PcdPointCloudPoint::gotoNext()
{
    switch (dataStorageType) {
        case PcdDataStorageType::ascii:
            return gotoNextAscii();
        case PcdDataStorageType::binary:
            return gotoNextBinary();
        case PcdDataStorageType::binary_compressed:
            return gotoNextBinaryCompressed();
    }
    return false;
}

bool PcdPointCloudPoint::gotoNextAscii()
{
    static_assert(sizeof(float) == 4 && sizeof(double) == 8);

    auto convertAndCopy = [](auto data, auto* dataPtr, auto size) {
        std::memcpy(dataPtr, &data, size);
    };
    // read a line
    std::string line;
    std::getline(*reader, line);
    if (!reader->good()) return false;
    // parse the line
    std::stringstream ss(line);
    std::string token;
    for (size_t fieldIt = 0; fieldIt < fieldByteSize.size(); fieldIt++) {
        auto count = fieldCount[fieldIt];
        auto size = fieldByteSize[fieldIt];
        auto type = fieldType[fieldIt];
        for (size_t countIt = 0; countIt < count; countIt++) {
            // read the token
            ss >> token;
            if (ss.fail()) return false;
            auto* position = dataBuffer.data() + fieldOffset[fieldIt] + countIt * size;
            try {
                errno = 0;
                // test the type and size
                if (type == 'F') {
                    if (size == 4) convertAndCopy(std::stof(token), position, size);
                    if (size == 8) convertAndCopy(std::stod(token), position, size);
                } else if (type == 'I') {
                    auto data = std::strtoimax(token.c_str(), nullptr, 10);
                    if (size == 1) convertAndCopy(static_cast<int8_t>(data), position, size);
                    if (size == 2) convertAndCopy(static_cast<int16_t>(data), position, size);
                    if (size == 4) convertAndCopy(static_cast<int32_t>(data), position, size);
                    if (size == 8) convertAndCopy(static_cast<int64_t>(data), position, size);
                } else if (type == 'U') {
                    auto data = std::strtoumax(token.c_str(), nullptr, 10);
                    if (size == 1) convertAndCopy(static_cast<uint8_t>(data), position, size);
                    if (size == 2) convertAndCopy(static_cast<uint16_t>(data), position, size);
                    if (size == 4) convertAndCopy(static_cast<uint32_t>(data), position, size);
                    if (size == 8) convertAndCopy(static_cast<uint64_t>(data), position, size);
                }
                if (errno != 0) return false;
            } catch (...) {
                return false;
            }
        }
    }
    return true;
}  

bool PcdPointCloudPoint::gotoNextBinary()
{
    // we just read the next record
    reader->read(reinterpret_cast<char*>(dataBuffer.data()), recordByteSize);
    if (!reader->good()) return false;
    return true;
}

bool PcdPointCloudPoint::gotoNextBinaryCompressed()
{
    return false;
}

PcdPointCloudPoint::~PcdPointCloudPoint()
{
    reader->close();
}

/**
 * @brief Constructs a PcdPointCloudHeader object with the specified parameters for the PCD file format (see the PCL documentation for more information).
 *
 * @param version The version of the PCD file format.
 * @param fields the names of each field in the PCD file
 * @param size the size of each field in the PCD file (in bytes)
 * @param type the type of each field in the PCD file ('F' for floating point, 'I' for integer, 'U' for unsigned integer)
 * @param count the number of elements for each field
 * @param width the width of the point cloud
 * @param height the height of the point cloud (1 if unstructured)
 * @param viewpoint the viewpoint of the point cloud, translation (tx ty tz) + quaternion (qw qx qy qz)
 * @param points the number of points in the point cloud
 * @param data the type of data in the point cloud (ascii, binary_compressed, or binary)
 */
PcdPointCloudHeader::PcdPointCloudHeader(const double version, const std::vector<std::string> &fields,
    const std::vector<size_t> &size, const std::vector<uint8_t> &type, const std::vector<size_t> &count,
    const size_t width, const size_t height, const std::vector<double> &viewpoint, const size_t points,
    const PcdDataStorageType data):
        version{version}, fields{fields}, size{size}, type{type}, count{count}, width{width}, height{height},
        viewpoint{viewpoint}, points{points}, data{data}
{ }

std::optional<PointCloudGenericAttribute> PcdPointCloudHeader::getAttributeById(int id) const
{
    switch (id) {
        case 0:
            return PointCloudGenericAttribute{version};
        case 1:
            return PointCloudGenericAttribute{fields};
        case 2:
            return PointCloudGenericAttribute{size};
        case 3:
            return PointCloudGenericAttribute{type};
        case 4:
            return PointCloudGenericAttribute{count};
        case 5:
            return PointCloudGenericAttribute{width};
        case 6:
            return PointCloudGenericAttribute{height};
        case 7:
            return PointCloudGenericAttribute{viewpoint};
        case 8:
            return PointCloudGenericAttribute{points};
        case 9:
            return PointCloudGenericAttribute{(std::ostringstream() << data).str()};
    }
    return std::nullopt;
}

std::optional<PointCloudGenericAttribute> PcdPointCloudHeader::getAttributeByName(const char *attributeName) const
{
    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    if (it != attributeNames.end()) {
        return getAttributeById(std::distance(attributeNames.begin(), it));
    }
    return std::nullopt;
}

std::vector<std::string> PcdPointCloudHeader::attributeList() const
{
    return attributeNames;
}

// data to string
static std::ostream &operator<<(std::ostream &os, const PcdDataStorageType &data)
{
    switch (data)
    {
        case PcdDataStorageType::ascii:
            os << "ascii";
            break;
        case PcdDataStorageType::binary:
            os << "binary";
            break;
        case PcdDataStorageType::binary_compressed:
            os << "binary_compressed";
            break;
        default:
            os << "unknown";
    }
    return os;
}

// data from string
static std::istream &operator>>(std::istream &is, PcdDataStorageType &data)
{
    std::string s;
    // to lower case
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    is >> s;
    if (s == "ascii")
        data = PcdDataStorageType::ascii;
    else if (s == "binary")
        data = PcdDataStorageType::binary;
    else if (s == "binary_compressed")
        data = PcdDataStorageType::binary_compressed;
    else
        data = PcdDataStorageType::binary; // default, should not happen
    return is;
}

std::optional<FullPointCloudAccessInterface> openPointCloudPcd(const std::filesystem::path &pcdFilePath)
{
   // open the file
    auto reader = std::make_unique<std::ifstream>(pcdFilePath, std::ios_base::binary);
    if (!reader->is_open()) return std::nullopt;

    // read the header
    auto header = readPcdHeader(*reader);
    // test if header ptr is not null
    if (header == nullptr) {
        return std::nullopt;
    }

    // create a point cloud
    auto pointCloud = std::make_unique<PcdPointCloudPoint>(std::move(reader), header->fields, header->size, header->type,
                                                           header->count, header->data);

    // return the point cloud
    if (pointCloud->gotoNext()) {
            FullPointCloudAccessInterface fullPointInterface;
            fullPointInterface.headerAccess = std::move(header);
            fullPointInterface.pointAccess = std::move(pointCloud);
            return fullPointInterface;
    }
    return std::nullopt;
}

/***************************       PRIVATE FUNCTIONS      ************************************/

// joins the elements of a vector/array to a string
template <class ContainerT>
static std::string joinToString(const ContainerT& c) {
    std::ostringstream oss;
    try {
    for (size_t i = 0; i < c.size(); ++i) {
        oss << c[i];
        if (i < c.size() - 1) {
            oss << " ";
        }
    }
    if (oss.fail()) {
        return "";
    }
    } catch (...) {
        return "";
    }
    return oss.str();
}

// return a vector of T from a string
template <typename T>
static std::vector<T> splitFromString(const std::string& str) {
    std::vector<T> vec;
    std::stringstream ss(str);
    T val;
    while (ss >> val) {
        if (ss.fail()) {
            return {};
        }
        vec.push_back(val);
    }
    return vec;
}

// return an std::array from a string
template <typename T, size_t N>
static std::array<T, N> splitFromString(const std::string& str) {
    std::array<T, N> arr;
    std::stringstream ss(str);
    T val;
    for (size_t i = 0; i < N; ++i) {
        ss >> val;
        if (ss.fail()) {
            return {};
        }
        arr[i] = val;
    }
    return arr;
}

// read the header of the PCD file
static std::unique_ptr<PcdPointCloudHeader> readPcdHeader(std::ifstream& reader) {
    
    // data for the header
    double version;
    std::vector<std::string> fields;
    std::vector<size_t> size;
    std::vector<uint8_t> type;
    std::vector<size_t> count;
    size_t width;
    size_t height;
    std::vector<double> viewpoint{0, 0, 0, 1, 0, 0, 0}; // translation (tx ty tz) + quaternion (qw qx qy qz)
    size_t points;
    PcdDataStorageType data = PcdDataStorageType::ascii; // default data type is ascii
    
    try {
        std::string line;
        std::vector<std::string> lineSplit; // variable holding the split line data
        std::stringstream lineStream; // we will also use this stream to easily convert the data to the correct type

        std::string headerEntryName;
        size_t nbFields;

        if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        
        //* --------- version ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "VERSION") {
            lineStream >> version;
            if (lineStream.fail()) return nullptr;
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        }

        //* --------- fields ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "FIELDS") {
            fields = std::vector<std::string>(lineSplit.begin()+1, lineSplit.end());
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        } else {
            // error if fields is not defined properly
            return nullptr;
        }
        // set the data according to the nimber of fields
        nbFields = fields.size();
        size = std::vector<size_t>(nbFields, 4); // default size is 4 bytes
        type = std::vector<uint8_t>(nbFields, 'F'); // default type is float
        count = std::vector<size_t>(nbFields, 1); // default count is 1

        //* --------- size ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "SIZE") {
            if (lineSplit.size()-1 != nbFields) return nullptr; // error if size is not defined properly
            for (size_t i = 0; i < nbFields; ++i) {
                lineStream >> size[i];
            }
            if (lineStream.fail()) return nullptr;
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        } else {
            // error if size is not defined properly
            return nullptr;
        }

        //* --------- type ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "TYPE") {
            if (lineSplit.size()-1 != nbFields) return nullptr; // error if type is not defined properly
            for (size_t i = 0; i < nbFields; ++i) {
                lineStream >> type[i];
            }
            if (lineStream.fail()) return nullptr;
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        } else {
            // error if type is not defined properly
            return nullptr;
        }
        
        //* --------- count ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "COUNT") {
            if (lineSplit.size()-1 != nbFields) return nullptr; // error if count is not defined properly
            for (size_t i = 0; i < nbFields; ++i) {
                lineStream >> count[i];
            }
            if (lineStream.fail()) return nullptr;
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        }

        //* --------- width ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "WIDTH") {
            lineStream >> width;
            if (lineStream.fail()) return nullptr;
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        } else {
            // error if width is not defined properly
            return nullptr;
        }

        //* --------- height ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "HEIGHT") {
            lineStream >> height;
            if (lineStream.fail()) return nullptr;
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        } else {
            // error if height is not defined properly
            return nullptr;
        }

        //* --------- viewpoint ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "VIEWPOINT") {
            for (size_t i = 0; i < 6; ++i) {
                lineStream >> viewpoint[i];
            }
            if (lineStream.fail()) return nullptr;
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        }

        //* --------- points ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "POINTS") {
            lineStream >> points;
            if (lineStream.fail()) return nullptr;
            if (!getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        } else {
            points = width * height;
        }

        //* --------- data ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "DATA") {
            lineStream >> data;
            if (lineStream.fail()) return nullptr;
        }
    } catch(const std::exception& e) {
        return nullptr;
    }

    return std::make_unique<PcdPointCloudHeader>(version, fields, size, type, count, width, height, viewpoint, points, data);
}

static bool getNextHeaderLine(std::ifstream& reader, std::string& line, std::vector<std::string>& lineSplit, std::stringstream& lineStream) {
    try {
        while(reader.good()) { // If no error occurs, the while loop should not reach the exit condition.
            // read a line
            std::getline(reader, line);

            //* ignore empty lines */
            // if the line is empty, skip it
            if (line.empty()) {
                continue;
            }

            // split the line into tokens
            lineStream.str(line);
            lineStream.clear(); // reset the error state

            std::string token;

            lineSplit.clear();
            while (lineStream >> token) {
                lineSplit.push_back(token);
            }

            //* ignore comments */
            if (lineSplit[0][0] == '#') {
                continue;
            }

            // reset the lineStream to the beginning of the line
            lineStream.clear(); // reset the error state
            lineStream.seekg(0, std::ios::beg);

            return true; // return true if the line is not empty
        }
    } catch(const std::exception& e) {
        return false;
    }
    return false;
}

} // namespace IO
} // namespace StereoVision