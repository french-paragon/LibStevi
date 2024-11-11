#include <cstdint>
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
    static std::unique_ptr<PcdPointCloudHeader> readHeader(std::ifstream& reader);
    static bool getNextHeaderLine(std::ifstream& reader, std::string& line, std::vector<std::string>& lineSplit, std::stringstream& lineStream);

PcdPointCloudPoint::PcdPointCloudPoint(std::unique_ptr<std::ifstream> reader, size_t recordByteSize):
    reader{std::move(reader)}, recordByteSize{recordByteSize}, fieldData(recordByteSize)
{
}

PtGeometry<PointCloudGenericAttribute> PcdPointCloudPoint::getPointPosition() const
{
    return PtGeometry<PointCloudGenericAttribute>();
}

std::optional<PtColor<PointCloudGenericAttribute>> PcdPointCloudPoint::getPointColor() const
{
    return std::optional<PtColor<PointCloudGenericAttribute>>();
}

std::optional<PointCloudGenericAttribute> PcdPointCloudPoint::getAttributeById(int id) const
{
    static_assert(sizeof(float) == 4); // check if float is 4 bytes, should be true on most systems
    static_assert(sizeof(double) == 8); // check if double is 8 bytes
    // test if id is valid
    if (id < 0 || id >= fieldByteSize.size() || fieldByteSize[id] + fieldOffset[id] > recordByteSize) {
        return std::nullopt;
    }
    const auto* const position = fieldData.data() + fieldOffset[id];
    // test the type and size
    if (fieldType[id] == 'F') {
        if (fieldByteSize[id] == 4) return *reinterpret_cast<const float*>(position);
        if (fieldByteSize[id] == 8) return *reinterpret_cast<const double*>(position);
    } else if (fieldType[id] == 'I') {
        if (fieldByteSize[id] == 1) return *reinterpret_cast<const int8_t*>(position);
        if (fieldByteSize[id] == 2) return *reinterpret_cast<const int16_t*>(position);
        if (fieldByteSize[id] == 4) return *reinterpret_cast<const int32_t*>(position);
        if (fieldByteSize[id] == 8) return *reinterpret_cast<const int64_t*>(position);
    } else if (fieldType[id] == 'U') {
        if (fieldByteSize[id] == 1) return *reinterpret_cast<const uint8_t*>(position);
        if (fieldByteSize[id] == 2) return *reinterpret_cast<const uint16_t*>(position);
        if (fieldByteSize[id] == 4) return *reinterpret_cast<const uint32_t*>(position);
        if (fieldByteSize[id] == 8) return *reinterpret_cast<const uint64_t*>(position);
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
 * @param viewpoint the viewpoint of the point cloud
 * @param points the number of points in the point cloud
 * @param data the type of data in the point cloud (ascii, binary_compressed, or binary)
 */
PcdPointCloudHeader::PcdPointCloudHeader(const double version, const std::vector<std::string> &fields,
    const std::vector<size_t> &size, const std::vector<char> &type, const std::vector<size_t> &count,
    const size_t width, const size_t height, const std::array<double, 7> &viewpoint, const size_t points,
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
            return PointCloudGenericAttribute{joinToString(fields)};
        case 2:
            return PointCloudGenericAttribute{joinToString(size)};
        case 3:
            return PointCloudGenericAttribute{joinToString(type)};
        case 4:
            return PointCloudGenericAttribute{joinToString(count)};
        case 5:
            return PointCloudGenericAttribute{width};
        case 6:
            return PointCloudGenericAttribute{height};
        case 7:
            return PointCloudGenericAttribute{joinToString(viewpoint)};
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
    FullPointCloudAccessInterface fullPcInterface;
    // open the file
    auto reader = std::make_unique<std::ifstream>(pcdFilePath, std::ios_base::binary);
    if (!reader->is_open()) return std::nullopt;

    // read the header
    fullPcInterface.headerAccess = readHeader(*reader);
    // test if header ptr is not null
    if (fullPcInterface.headerAccess == nullptr) {
        std::cout << "header ptr is null" << std::endl;
    }
    return fullPcInterface;
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
static std::unique_ptr<PcdPointCloudHeader> readHeader(std::ifstream& reader) {
    
    // data for the header
    double version;
    std::vector<std::string> fields;
    std::vector<size_t> size;
    std::vector<char> type;
    std::vector<size_t> count;
    size_t width;
    size_t height;
    std::array<double, 7> viewpoint{0, 0, 0, 1, 0, 0, 0}; // translation (tx ty tz) + quaternion (qw qx qy qz)
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
        type = std::vector<char>(nbFields, 'F'); // default type is float
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