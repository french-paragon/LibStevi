#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <numeric>
#include <optional>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>
#include "pcd_pointcloud_io.h"
#include "pointcloud_io.h"
#include "fstreamCustomBuffer.h"

namespace StereoVision {
namespace IO {

// constants
// buffersize when reading pcd files
constexpr static size_t pcdFileReaderBufferSize = 1 << 16;
// write
constexpr static size_t pcdFileWriterBufferSize_binary = 1 << 16;
constexpr static size_t pcdFileWriterBufferSize_ascii = 1 << 16;

static std::ostream &operator<<(std::ostream &os, const PcdDataStorageType &data);
static std::istream &operator>>(std::istream &is, PcdDataStorageType &data);

/// @brief From any pointcloud point, get the names of the attributes, their byte size, their type and their count
/// @param pointcloudPointAccessInterface The pointcloud point access interface
/// @return A tuple containing the attributes names, their byte size, their type and their count
static std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<uint8_t>, std::vector<size_t>>
    getPcdDataLayoutFromPointcloudPoint(const PointCloudPointAccessInterface& pointcloudPointAccessInterface);

/// @brief adapter class to obtain a PcdPointCloudPoint from any PointCloudPointAccessInterface
class PcdPointCloudPointAdapter : public PcdPointCloudPoint
{
protected:
    PointCloudPointAccessInterface* pointCloudPointAccessInterface = nullptr;
public:
    PcdPointCloudPointAdapter(PointCloudPointAccessInterface* pointCloudPointAccessInterface);

    bool gotoNext() override;

    // destructor
    ~PcdPointCloudPointAdapter() override;

protected:
    PcdPointCloudPointAdapter(PointCloudPointAccessInterface* pointCloudPointAccessInterface,
        const std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<uint8_t>, std::vector<size_t>>& attributeInformations);

    /**
     * @brief fill the internal data buffer with the values of the attributes of the current point.
     * 
     * @return true if the internal state was properly adapted, false otherwise
     */
    bool adaptInternalState();
};

/// @brief adapter class to obtain a PcdPointCloudHeader from any PointCloudHeaderInterface
class PcdPointCloudHeaderAdapter : public PcdPointCloudHeader
{
protected:
    PointCloudHeaderInterface* pointCloudHeaderInterface = nullptr;
public:
    PcdPointCloudHeaderAdapter(PointCloudHeaderInterface* pointCloudHeaderInterface);

    // destructor
    ~PcdPointCloudHeaderAdapter() override;

private:
    /**
     * @brief set the internal state of the adapter
     *
     * @return true if the internal state was properly adapted, false otherwise
     */
    bool adaptInternalState();
};

PcdPointCloudPoint::PcdPointCloudPoint(std::unique_ptr<std::istream> reader,
    const std::vector<std::string>& attributeNames, const std::vector<size_t>& fieldByteSize,
    const std::vector<uint8_t>& fieldType, const std::vector<size_t>& fieldCount, PcdDataStorageType dataStorageType):
    attributeNames{attributeNames}, fieldByteSize{fieldByteSize},
    fieldOffset(fieldByteSize.size()), fieldType{fieldType}, fieldCount{fieldCount},
    dataStorageType{dataStorageType}, reader{std::move(reader)}
{
    // compute the offsets
    const auto* fieldCountPtr = fieldCount.data();
    const auto* fieldByteSizePtr = fieldByteSize.data();
    // compute the offsets with transform_exclusive_scan
    std::transform_exclusive_scan(fieldByteSize.begin(), fieldByteSize.end(), fieldOffset.begin(), size_t{0}, std::plus<>{},
    [fieldCountPtr, fieldByteSizePtr](const auto& size) {
        auto i = std::distance(fieldByteSizePtr, &size);
        return fieldCountPtr[i] * size;
    });
    // compute the size of a "record", which is the last offset + the last size times the last count
    if (!(fieldOffset.empty() || fieldCount.empty() || fieldByteSize.empty())) {
        recordByteSize = fieldOffset.back() + fieldCount.back() * fieldByteSize.back();
    }
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

bool PcdPointCloudPoint::gotoNextAscii()
{
    static_assert(sizeof(float) == 4 && sizeof(double) == 8);

    auto convertAndCopy = [](auto data, auto* dataPtr, auto size) {
        std::memcpy(dataPtr, &data, size);
    };

    // if the reader is at the end of the file or in a bad state, return false
    if (!reader->good()) return false;
    
    std::string line;
    // read a line until the line is not empty or cannot read more (EOF or read error)
    do {
        std::getline(*reader, line);
    } while (reader->good() && line.empty());
    if (reader->fail() || line.empty()) return false;

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

std::shared_ptr<PcdPointCloudPoint> PcdPointCloudPoint::createAdapter(PointCloudPointAccessInterface *pointCloudPointAccessInterface)
{
    // test if nullptr. If so, return nullptr
    if (pointCloudPointAccessInterface == nullptr) {return nullptr;}
    
    // we test if the interface is already a PcdPointCloudPoint with a dynamic cast
    PcdPointCloudPoint* pcdPoint = dynamic_cast<PcdPointCloudPoint*>(pointCloudPointAccessInterface);
    if (pcdPoint != nullptr) {
        // if it is already a PcdPointCloudPointAdapter, we return a shared pointer with no ownership
        return std::shared_ptr<PcdPointCloudPoint>{std::shared_ptr<PcdPointCloudPoint>{}, pcdPoint};
    } else {
        // create a new PcdPointCloudPointAdapter
        return std::make_shared<PcdPointCloudPointAdapter>(pointCloudPointAccessInterface);
    }
}

PcdPointCloudPoint::~PcdPointCloudPoint(){}

PcdPointCloudHeader::PcdPointCloudHeader(){}

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

// read the header of the PCD filead
std::unique_ptr<PcdPointCloudHeader> PcdPointCloudHeader::readHeader(std::istream& reader) {
    
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

        constexpr auto maxPrecision{std::numeric_limits<long double>::max_digits10 + 1};
        lineStream << std::setprecision(maxPrecision);

        if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        
        //* --------- version ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "VERSION") {
            lineStream >> version;
            if (lineStream.fail()) return nullptr;
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        }

        //* --------- fields ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "FIELDS") {
            fields = std::vector<std::string>(lineSplit.begin()+1, lineSplit.end());
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
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
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
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
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
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
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        }

        //* --------- width ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "WIDTH") {
            lineStream >> width;
            if (lineStream.fail()) return nullptr;
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        } else {
            // error if width is not defined properly
            return nullptr;
        }

        //* --------- height ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "HEIGHT") {
            lineStream >> height;
            if (lineStream.fail()) return nullptr;
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        } else {
            // error if height is not defined properly
            return nullptr;
        }

        //* --------- viewpoint ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "VIEWPOINT") {
            for (size_t i = 0; i < viewpoint.size(); ++i) {
                lineStream >> viewpoint[i];
            }
            if (lineStream.fail()) return nullptr;
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
        }

        //* --------- points ----------------
        lineStream >> headerEntryName;
        if (headerEntryName == "POINTS") {
            lineStream >> points;
            if (lineStream.fail()) return nullptr;
            if (!PcdPointCloudHeader::getNextHeaderLine(reader, line, lineSplit, lineStream)) return nullptr;
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

bool PcdPointCloudHeader::getNextHeaderLine(std::istream& reader, std::string& line, std::vector<std::string>& lineSplit, std::stringstream& lineStream) {
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

bool PcdPointCloudHeader::writeHeader(std::ostream &writer, const PcdPointCloudHeader &header, std::streampos &headerWidthPos,
                    std::streampos &headerHeightPos, std::streampos &headerPointsPos)
{
    if (!writer.good()) return false;

    // convert the bigest size_t to a string and get its length
    static const size_t maxSizeStr = std::to_string(std::numeric_limits<size_t>::max()).length();

    // resize the width, height and points attributes to the length of maxSizeStr
    std::string widthStr = std::to_string(header.width);
    std::string heightStr = std::to_string(header.height);
    std::string pointsStr = std::to_string(header.points);
    widthStr.resize(maxSizeStr, ' ');
    heightStr.resize(maxSizeStr, ' ');
    pointsStr.resize(maxSizeStr, ' ');

    // vector to string
    auto vectorToString = [](auto&& c) {
        std::stringstream ss;
        using U = std::decay_t<decltype(c)>;
        if constexpr (std::is_floating_point_v<U>) {
            constexpr auto maxPrecision{std::numeric_limits<U>::digits10 + 1};
            ss << std::setprecision(maxPrecision); // write with max precision
        }
        for (size_t i = 0; i < c.size() - 1; ++i) {ss << c[i] << " ";}
        if (c.size() > 0) {ss << c[c.size() - 1];}
        return ss.str();
    };

    // write the data
    writer << "VERSION" << " " << header.version << std::endl
           << "FIELDS" << " " << vectorToString(header.fields) << std::endl
           << "SIZE" << " " << vectorToString(header.size) << std::endl
           << "TYPE" << " " << vectorToString(header.type) << std::endl
           << "COUNT" << " " << vectorToString(header.count) << std::endl
           << "WIDTH" << " ";
    headerWidthPos = writer.tellp();
    writer << widthStr << std::endl
            << "HEIGHT" << " "; 
    headerHeightPos = writer.tellp();
    writer << heightStr << std::endl
           << "VIEWPOINT" << " " << vectorToString(header.viewpoint) << std::endl
           << "POINTS" << " ";
    headerPointsPos = writer.tellp();
    writer << pointsStr << std::endl
           << "DATA" << " " << header.data << std::endl;

    return true;
}

std::shared_ptr<PcdPointCloudHeader> PcdPointCloudHeader::createAdapter(PointCloudHeaderInterface *pointCloudHeaderInterface)
{
    // test if nullptr. If so, return nullptr
    if (pointCloudHeaderInterface == nullptr) {return nullptr;}
    
    // we test if the interface is already a PcdPointCloudHeader with a dynamic cast
    PcdPointCloudHeader* pcdPoint = dynamic_cast<PcdPointCloudHeader*>(pointCloudHeaderInterface);
    if (pcdPoint != nullptr) {
        // if it is already a PcdPointCloudHeaderAdapter, we return a shared pointer with no ownership
        return std::shared_ptr<PcdPointCloudHeader>{std::shared_ptr<PcdPointCloudHeader>{}, pcdPoint};
    } else {
        // create a new PcdPointCloudHeaderAdapter
        return std::make_shared<PcdPointCloudHeaderAdapter>(pointCloudHeaderInterface);
    }
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

std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<uint8_t>, std::vector<size_t>>
    getPcdDataLayoutFromPointcloudPoint(const PointCloudPointAccessInterface &pointcloudPointAccessInterface)
{
    // try to guess header parameters from the point cloud directly
    auto attributeNames = std::vector<std::string>{};
    auto size = std::vector<size_t>{};
    auto type = std::vector<uint8_t>{};
    auto count = std::vector<size_t>{};

    size_t nbAttributes = 0;
    for (auto&& attributeName : pointcloudPointAccessInterface.attributeList()) {
        // try to find the attribute name in the header
        auto attrOpt = pointcloudPointAccessInterface.getAttributeByName(attributeName.c_str()); 

        if (attrOpt) {
            const auto attr = attrOpt.value();
            // visit the variant to get the size, type, count and data
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                // simple types
                if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>) {
                    attributeNames.push_back(attributeName);
                    nbAttributes++;
                    size.push_back(sizeof(T));
                    count.push_back(1);
                    if constexpr (std::is_floating_point_v<T>) {
                        type.push_back('F');
                    } else if constexpr (std::is_unsigned_v<T>) {
                        type.push_back('U');
                    } else {
                        type.push_back('I');
                    }
                } else if constexpr (std::is_same_v<T, std::string>) {
                    // PCD cannot store strings...We ignore it
                } else if constexpr (is_vector_v<T>) { // vector types
                    // get the contained type
                    using V_t = std::decay_t<typename T::value_type>;
                    if constexpr (std::is_floating_point_v<V_t> || std::is_integral_v<V_t>) {
                        attributeNames.push_back(attributeName);
                        nbAttributes++;
                        size.push_back(sizeof(V_t));
                        count.push_back(arg.size());
                        if constexpr (std::is_floating_point_v<V_t>) {
                            type.push_back('F');
                        } else if constexpr (std::is_unsigned_v<V_t>) {
                            type.push_back('U');
                        } else {
                            type.push_back('I');
                        }
                    } else if constexpr (std::is_same_v<V_t, std::string>) {
                        // PCD cannot store strings...We ignore it
                    } else {
                        static_assert(false, "All types in the variant must be handled");
                    }
                } else {
                    static_assert(false, "All types in the variant must be handled");
                }
            }, attr);
        }
    }
    return {attributeNames, size, type, count};
}

std::optional<FullPointCloudAccessInterface> openPointCloudPcd(const std::filesystem::path &pcdFilePath)
{
   // open the file
    auto reader = std::make_unique<ifstreamCustomBuffer<pcdFileReaderBufferSize>>();

    reader->open(pcdFilePath, std::ios_base::binary);

    if (!reader->is_open()) return std::nullopt;

    // read the header
    auto header = PcdPointCloudHeader::readHeader(*reader);
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

bool writePointCloudPcd(const std::filesystem::path &pcdFilePath, FullPointCloudAccessInterface &pointCloud,
    std::optional<PcdDataStorageType> dataStorageType)
{
    // position of some header elements that might change
    std::streampos headerWidthPos;
    std::streampos headerHeightPos;
    std::streampos headerPointsPos;

    // default header data
    double headerVersion = 0.7;
    size_t headerWidth = 0;
    size_t headerHeight = 0;
    std::vector<double> headerViewpoint = {0, 0, 0, 1, 0, 0, 0};
    size_t headerPoints = 0;
    PcdDataStorageType headerData = PcdDataStorageType::ascii;

    auto pcdHeaderAccessAdapter = PcdPointCloudHeader::createAdapter(pointCloud.headerAccess.get());
    if (pcdHeaderAccessAdapter) {
        headerVersion = pcdHeaderAccessAdapter->version;
        headerWidth = pcdHeaderAccessAdapter->width;
        headerHeight = pcdHeaderAccessAdapter->height;
        headerViewpoint = pcdHeaderAccessAdapter->viewpoint;
        headerPoints = pcdHeaderAccessAdapter->points;
        headerData = pcdHeaderAccessAdapter->data;
    }

    // get the pointcloud point adapter
    auto pcdPointAccessAdapter = PcdPointCloudPoint::createAdapter(pointCloud.pointAccess.get());
    if (!(pcdPointAccessAdapter)) {return false;}

    auto usedDataStorageType = headerData;
    // set the data storage type
    if (dataStorageType.has_value()) {
        usedDataStorageType = dataStorageType.value();
    }

    // we generate a totally new header because if there is a mismatch between the informations of the point cloud and the header,
    // we use the informations guessed from the point cloud point.
    // the only information that we keep from the header is the data storage type (if not set by the user), the number of points,
    // the width and the height (even though they may be wrong and will be corrected later after writing the points).
    PcdPointCloudHeader newHeader{headerVersion, pcdPointAccessAdapter->attributeList(),
        pcdPointAccessAdapter->getFieldByteSize(), pcdPointAccessAdapter->getFieldType(),
        pcdPointAccessAdapter->getFieldCount(), headerWidth, headerHeight,
        headerViewpoint, headerPoints, usedDataStorageType};


   // open the file
    auto writer = std::unique_ptr<std::fstream>{nullptr};

    // set the buffer size
    if (usedDataStorageType == PcdDataStorageType::ascii) {
        writer = std::make_unique<fstreamCustomBuffer<pcdFileWriterBufferSize_ascii>>();
    } else if (usedDataStorageType == PcdDataStorageType::binary) {
        writer = std::make_unique<fstreamCustomBuffer<pcdFileWriterBufferSize_binary>>();
    }  else {
        // default
        writer = std::make_unique<fstreamCustomBuffer<pcdFileWriterBufferSize_binary>>();
    }

    writer->open(pcdFilePath, std::ios_base::in | std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

    if (!writer->is_open()) return false;

    // write the header
    if (!PcdPointCloudHeader::writeHeader(*writer, newHeader, headerWidthPos, headerHeightPos, headerPointsPos)) {
        return false;
    }


    size_t nbPoints = 0;
    // write the points
    switch (usedDataStorageType) {
        case PcdDataStorageType::ascii:
            do {
                if (!PcdPointCloudPointAdapter::writePointAscii(*writer, *pcdPointAccessAdapter)) return false;
                nbPoints++;
            } while (pcdPointAccessAdapter->gotoNext());
            break;
        case PcdDataStorageType::binary:
            do {
                if (!PcdPointCloudPointAdapter::writePointBinary(*writer, *pcdPointAccessAdapter)) return false;
                nbPoints++;
            } while (pcdPointAccessAdapter->gotoNext());
            break;
        case PcdDataStorageType::binary_compressed:
            return false;
            break;
        default:
            return false;
            break;
    }

    // convert the bigest size_t to a string and get its length
    static const size_t maxSizeStr = std::to_string(std::numeric_limits<size_t>::max()).length();
    // test if the number of points is the same as the number of points in the header. Otherwise, we have to modify it
    // in the header
    if (nbPoints != newHeader.points || nbPoints != newHeader.width * newHeader.height) {
        // we cannot guess the width and height if the size of the point cloud has changed. Therefore, we have to
        // set the width to nbPoints and the height to 1

        // convert everything to a string
        std::string pointsStr = std::to_string(nbPoints);
        std::string widthStr = pointsStr;
        std::string heightStr = std::to_string(1);
        // modify the length of the string to the maximum length
        pointsStr.resize(maxSizeStr, ' ');
        widthStr.resize(maxSizeStr, ' ');
        heightStr.resize(maxSizeStr, ' ');
        // rewrite them
        writer->seekp(headerPointsPos);
        writer->write(pointsStr.c_str(), maxSizeStr);
        writer->seekp(headerWidthPos);
        writer->write(widthStr.c_str(), maxSizeStr);
        writer->seekp(headerHeightPos);
        writer->write(heightStr.c_str(), maxSizeStr);
        // seek to the end of the file
        writer->seekp(0, std::ios_base::end);
    }

    return true;
}

PcdPointCloudPointAdapter::PcdPointCloudPointAdapter(PointCloudPointAccessInterface *pointCloudPointAccessInterface,
    const std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<uint8_t>, std::vector<size_t>>& attributeInformations) :
    PcdPointCloudPoint{nullptr, std::get<0>(attributeInformations), std::get<1>(attributeInformations), std::get<2>(attributeInformations),
        std::get<3>(attributeInformations), PcdDataStorageType::ascii}, pointCloudPointAccessInterface(pointCloudPointAccessInterface)
{
    adaptInternalState(); // set the internal state for the first time
}

PcdPointCloudPointAdapter::PcdPointCloudPointAdapter(PointCloudPointAccessInterface *pointCloudPointAccessInterface):
    PcdPointCloudPointAdapter(pointCloudPointAccessInterface, getPcdDataLayoutFromPointcloudPoint(*pointCloudPointAccessInterface)) {}

bool PcdPointCloudPointAdapter::gotoNext() {
    return pointCloudPointAccessInterface->gotoNext() && adaptInternalState();
}

PcdPointCloudPointAdapter::~PcdPointCloudPointAdapter(){}

bool PcdPointCloudPointAdapter::adaptInternalState() {
    static_assert(sizeof(float) == 4 && sizeof(double) == 8);

    if (pointCloudPointAccessInterface == nullptr) return false;

    auto convertAndCopy = [](auto data, auto* dataPtr, auto size) {
        std::memcpy(dataPtr, &data, size);
    };

    auto convertAndCopyVector = [](const auto& vector, auto* dataPtr, auto elementByteSize) {
        std::memcpy(dataPtr, vector.data(), vector.size() * elementByteSize);
    };

    for (size_t fieldIt = 0; fieldIt < fieldByteSize.size(); fieldIt++) {
        auto count = fieldCount[fieldIt];
        auto size = fieldByteSize[fieldIt];
        auto type = fieldType[fieldIt];

        // try to get the attribute
        auto attrOpt = pointCloudPointAccessInterface->getAttributeById(fieldIt);
        if (!attrOpt.has_value()) return false;
        const auto& attr = attrOpt.value();

            auto* position = dataBuffer.data() + fieldOffset[fieldIt];
            try {
                if (!isAttributeList(attr)) {
                    if (count != 1) return false; // should not happen
                    // test the type and size
                    if (type == 'F') {
                        if (size == 4) convertAndCopy(castedPointCloudAttribute<float>(attr), position, size);
                        if (size == 8) convertAndCopy(castedPointCloudAttribute<double>(attr), position, size);
                    } else if (type == 'I') {
                        if (size == 1) convertAndCopy(castedPointCloudAttribute<int8_t>(attr), position, size);
                        if (size == 2) convertAndCopy(castedPointCloudAttribute<int16_t>(attr), position, size);
                        if (size == 4) convertAndCopy(castedPointCloudAttribute<int32_t>(attr), position, size);
                        if (size == 8) convertAndCopy(castedPointCloudAttribute<int64_t>(attr), position, size);
                    } else if (type == 'U') {
                        if (size == 1) convertAndCopy(castedPointCloudAttribute<uint8_t>(attr), position, size);
                        if (size == 2) convertAndCopy(castedPointCloudAttribute<uint16_t>(attr), position, size);
                        if (size == 4) convertAndCopy(castedPointCloudAttribute<uint32_t>(attr), position, size);
                        if (size == 8) convertAndCopy(castedPointCloudAttribute<uint64_t>(attr), position, size);
                    } else {
                        return false;
                    }
                } else { // same thing but with vectors
                    if (type == 'F') {
                        if (size == 4) convertAndCopyVector(castedPointCloudAttribute<std::vector<float>>(attr), position, size);
                        if (size == 8) convertAndCopyVector(castedPointCloudAttribute<std::vector<double>>(attr), position, size);
                    } else if (type == 'I') {
                        if (size == 1) convertAndCopyVector(castedPointCloudAttribute<std::vector<int8_t>>(attr), position, size);
                        if (size == 2) convertAndCopyVector(castedPointCloudAttribute<std::vector<int16_t>>(attr), position, size);
                        if (size == 4) convertAndCopyVector(castedPointCloudAttribute<std::vector<int32_t>>(attr), position, size);
                        if (size == 8) convertAndCopyVector(castedPointCloudAttribute<std::vector<int64_t>>(attr), position, size);
                    } else if (type == 'U') {
                        if (size == 1) convertAndCopyVector(castedPointCloudAttribute<std::vector<uint8_t>>(attr), position, size);
                        if (size == 2) convertAndCopyVector(castedPointCloudAttribute<std::vector<uint16_t>>(attr), position, size);
                        if (size == 4) convertAndCopyVector(castedPointCloudAttribute<std::vector<uint32_t>>(attr), position, size);
                        if (size == 8) convertAndCopyVector(castedPointCloudAttribute<std::vector<uint64_t>>(attr), position, size);
                    } else {
                        return false;
                    }
                }
            } catch (...) {
                return false;
            }
            
    }
    return true;
}

PcdPointCloudHeaderAdapter::PcdPointCloudHeaderAdapter(PointCloudHeaderInterface *pointCloudHeaderInterface) :
    PcdPointCloudHeader{}
{
    this->pointCloudHeaderInterface = pointCloudHeaderInterface;

    // try to adapt the header
    adaptInternalState();
}

PcdPointCloudHeaderAdapter::~PcdPointCloudHeaderAdapter(){}

bool PcdPointCloudHeaderAdapter::adaptInternalState()
{
    // test if the list of attributes is valid (contains all the required attributes)
    for (const auto& attr : attributeNames) {
        auto it = std::find(pointCloudHeaderInterface->attributeList().begin(), pointCloudHeaderInterface->attributeList().end(), attr);
        if (it == pointCloudHeaderInterface->attributeList().end() && attr != "viewpoint" && attr != "count") {
            return false;
        }
    }

    // version
    auto versionOpt = pointCloudHeaderInterface->getAttributeByName("version");
    if (!versionOpt.has_value()) return false;
    version = castedPointCloudAttribute<double>(versionOpt.value());
    // fields
    auto fieldsOpt = pointCloudHeaderInterface->getAttributeByName("fields");
    if (!fieldsOpt.has_value()) return false;
    fields = castedPointCloudAttribute<std::vector<std::string>>(fieldsOpt.value());
    // size
    auto sizeOpt = pointCloudHeaderInterface->getAttributeByName("size");
    if (!sizeOpt.has_value()) return false;
    size = castedPointCloudAttribute<std::vector<size_t>>(sizeOpt.value());
    // type
    auto typeOpt = pointCloudHeaderInterface->getAttributeByName("type");
    if (!typeOpt.has_value()) return false;
    type = castedPointCloudAttribute<std::vector<uint8_t>>(typeOpt.value());
    // count
    auto countOpt = pointCloudHeaderInterface->getAttributeByName("count");
    if (!countOpt.has_value()) {
        count = std::vector<size_t>(fields.size(), 1);
    } else {
        count = castedPointCloudAttribute<std::vector<size_t>>(countOpt.value());   
    }
    // width
    auto widthOpt = pointCloudHeaderInterface->getAttributeByName("width");
    if (!widthOpt.has_value()) return false;
    width = castedPointCloudAttribute<size_t>(widthOpt.value());
    // height
    auto heightOpt = pointCloudHeaderInterface->getAttributeByName("height");
    if (!heightOpt.has_value()) return false;
    height = castedPointCloudAttribute<size_t>(heightOpt.value());
    // viewpoint
    auto viewpointOpt = pointCloudHeaderInterface->getAttributeByName("viewpoint");
    if (!viewpointOpt.has_value()) {
        viewpoint = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    } else {
        viewpoint = castedPointCloudAttribute<std::vector<double>>(viewpointOpt.value());
    }
    // points
    auto pointsOpt = pointCloudHeaderInterface->getAttributeByName("points");
    if (!pointsOpt.has_value()) return false;
    points = castedPointCloudAttribute<size_t>(pointsOpt.value());
    // data
    auto dataOpt = pointCloudHeaderInterface->getAttributeByName("data");
    if (!dataOpt.has_value()) return false;
        std::istringstream(castedPointCloudAttribute<std::string>(dataOpt.value())) >> data;

    // do some tests
    // the size of the fields should match
    size_t nbFields = fields.size();
    if (nbFields == 0 || size.size() != nbFields || type.size() != nbFields || count.size() != nbFields) {
        return false;
    }

    // the number of points should match the width * height
    if (width * height != points) {
        return false;
    }

    // the size of the viewpoint should be 7
    if (viewpoint.size() != 7) {
        return false;
    }
    return true;
}

bool PcdPointCloudPoint::writePoint(std::ostream &writer, const PcdPointCloudPoint &point, PcdDataStorageType dataStorageType )
{
    switch (dataStorageType) {
        case PcdDataStorageType::ascii:
            return writePointAscii(writer, point);
        case PcdDataStorageType::binary:
            return writePointBinary(writer, point);
        case PcdDataStorageType::binary_compressed:
            return false;
        default:
            return false;
    }
}


bool PcdPointCloudPoint::writePointBinary(std::ostream &writer, const PcdPointCloudPoint &point)
{
    // simply write the bytes
    writer.write(reinterpret_cast<const char*>(point.dataBuffer.data()), point.dataBuffer.size());
    if (!writer.good()) return false;
    return true;
}

bool PcdPointCloudPoint::writePointAscii(std::ostream &writer, const PcdPointCloudPoint &point)
{
    // visit the variant and write each field as a string
    auto visitor = [&writer](auto&& attr) {
        using T = std::decay_t<decltype(attr)>;
        if constexpr (std::is_same_v<T, std::string>) {
            writer << attr;
        } else if constexpr (std::is_floating_point_v<T>) {
            constexpr auto maxPrecision{std::numeric_limits<T>::digits10 + 1};
            writer << std::setprecision(maxPrecision) << attr; // write with max precision
        } else if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_signed_v<T>) { // convert to bigger type to display char type as a number
                writer << static_cast<intmax_t>(attr);
            } else {
                writer << static_cast<uintmax_t>(attr);
            }
        } else if constexpr (is_vector_v<T>) {
            using value_type = typename T::value_type;
            if constexpr (std::is_same_v<value_type, std::string>) {
                for (size_t i = 0; i < attr.size()-1; i++) { writer << attr[i] << " "; }
                if (attr.size() > 0) { writer << attr[attr.size()-1]; }
            } else if constexpr (std::is_floating_point_v<value_type>) {
                constexpr auto maxPrecision{std::numeric_limits<value_type>::digits10 + 1};
                for (size_t i = 0; i < attr.size()-1; i++) { writer << std::setprecision(maxPrecision) << attr[i] << " "; }
                if (attr.size() > 0) { writer << std::setprecision(maxPrecision) << attr[attr.size()-1]; }
            } else if constexpr (std::is_integral_v<value_type>) {
                if constexpr (std::is_signed_v<value_type>) { // convert to bigger type to display char type as a number
                    for (size_t i = 0; i < attr.size()-1; i++) { writer << static_cast<intmax_t>(attr[i]) << " "; }
                    if (attr.size() > 0) { writer << static_cast<intmax_t>(attr[attr.size()-1]); }
                } else {
                    for (size_t i = 0; i < attr.size()-1; i++) { writer << static_cast<uintmax_t>(attr[i]) << " "; }
                    if (attr.size() > 0) { writer << static_cast<uintmax_t>(attr[attr.size()-1]); }
                }
            } else {
                static_assert(false, "Unsupported vector type");
            }
        } else {
            static_assert(false, "Unsupported type");
        }
    };
    // write each field
    for (size_t i = 0; i < point.fieldCount.size()-1; i++) {
        std::visit(visitor, point.getAttributeById(i).value());
        writer << " ";
    }
    std::visit(visitor, point.getAttributeById(point.fieldCount.size() - 1).value()); // last field
    writer << std::endl;
    return writer.good();
}

} // namespace IO
} // namespace StereoVision