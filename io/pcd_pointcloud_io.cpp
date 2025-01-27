#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <charconv>
#include <cmath>
#include <numeric>
#include <optional>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>
#include "pcd_pointcloud_io.h"
#include "sdc_pointcloud_io.h"
#include "fstreamCustomBuffer.h"
#include "bit_manipulations.h"

namespace StereoVision {
namespace IO {

using PcdDataLayout = std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<size_t>,
    std::vector<uint8_t>, std::vector<size_t>>;

// constants
// buffersize when reading pcd files
constexpr static size_t pcdFileReaderBufferSize = 1 << 16;
// write
constexpr static size_t pcdFileWriterBufferSize = 1 << 16;

static std::ostream &operator<<(std::ostream &os, const PcdDataStorageType &data);
static std::istream &operator>>(std::istream &is, PcdDataStorageType &data);

/** @brief From any pointcloud point, get the sanitized attributes names, the original names of the attributes,
 * their byte size, their type and their count
 * @param pointcloudPointAccessInterface The pointcloud point access interface
 * @return A tuple containing the sanitized attributes names, original attributes names, their byte size, their type and their count
 */
static PcdDataLayout getPcdDataLayoutFromPointcloudPoint(PointCloudPointAccessInterface* pointcloudPointAccessInterface);

/**
 * @brief sanitize a string to be a valid pcd attribute name. Only alphanumeric characters and underscores are allowed.
 * 
 * @param str the string to sanitize
 * @return std::string the sanitized string
 */
static std::string sanitizeAttributeNamePcd(const std::string& str);

/// @brief adapter class to obtain a PcdPointCloudPoint from any PointCloudPointAccessInterface
class PcdPointCloudPointBasicAdapter : public PcdPointCloudPoint {
protected:
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr = nullptr;
    PointCloudPointAccessInterface* pointCloudPointAccessInterface = nullptr;

    const std::vector<std::string> originalAttributeNames;
public:
    /**
     * @brief Adapter class to obtain a PcdPointCloudPoint from any PointCloudPointAccessInterface
     *
     * @param pointCloudPointAccessInterfaceUniquePtr a unique pointer to a PointCloudPointAccessInterface object.
     * @param pointCloudPointAccessInterface a pointer to a PointCloudPointAccessInterface object. If
     * pointCloudPointAccessInterfaceUniquePtr is not nullptr, this should point to the same object.
     */
    PcdPointCloudPointBasicAdapter(std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr,
        PointCloudPointAccessInterface* pointCloudPointAccessInterface );

    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;
    
    bool gotoNext() override;

protected:
    PcdPointCloudPointBasicAdapter(std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr,
        PointCloudPointAccessInterface* pointCloudPointAccessInterface,
        const PcdDataLayout& attributeInformations);

    /**
     * @brief fill the internal data buffer with the values of the attributes of the current point.
     * 
     * @return true if the internal state was properly adapted, false otherwise
     */
    bool adaptInternalState();
};

/// @brief adapter class to obtain a PcdPointCloudPoint from a SdcPointCloudPoint
class PcdPointCloudPointFromSdcAdapter : public PcdPointCloudPoint {
protected:
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointInterface = nullptr;
    SdcPointCloudPoint* castedSdcPointCloudPoint = nullptr;
public:
    /**
     * @brief Construct a new Pcd Point Cloud Point From a PointCloudPointAccessInterface object that must be a SdcPointCloudPoint.
     * 
     * @param pointCloudPointInterface a unique pointer to a PointCloudPointAccessInterface object that must be a SdcPointCloudPoint
     * @param sdcPointCloudPoint a pointer to the SdcPointCloudPoint. If sdcPointCloudPointUniquePtr is not nullptr,
     * this pointer should point to the same object. If sdcPointCloudPointUniquePtr is nullptr, it may still point to
     * a valid SdcPointCloudPoint. This can be useful when we don't want the pass the ownership of the SdcPointCloudPoint
     * to the adapter.
     */
    PcdPointCloudPointFromSdcAdapter(std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointInterface,
        SdcPointCloudPoint* castedSdcPointCloudPoint);

    bool gotoNext() override;

protected:
    PcdPointCloudPointFromSdcAdapter(std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointInterface,
        SdcPointCloudPoint* castedSdcPointCloudPoint, const PcdDataLayout& attributeInformations);
};

/// @brief adapter class to obtain a PcdPointCloudHeader from any PointCloudHeaderInterface
class PcdPointCloudHeaderBasicAdapter : public PcdPointCloudHeader {
protected:
    std::unique_ptr<PointCloudHeaderInterface> pointCloudHeaderInterface = nullptr;
public:
    PcdPointCloudHeaderBasicAdapter(std::unique_ptr<PointCloudHeaderInterface> pointCloudHeaderInterface);

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
    const std::vector<uint8_t>& fieldType, const std::vector<size_t>& fieldCount, PcdDataStorageType dataStorageType) :
    PcdPointCloudPoint(std::move(reader), attributeNames, fieldByteSize, fieldType, fieldCount, dataStorageType, nullptr)
{
    dataBufferContainer.resize(recordByteSize);
    dataBuffer = dataBufferContainer.data();
}

PcdPointCloudPoint::PcdPointCloudPoint(std::unique_ptr<std::istream> reader,
    const std::vector<std::string>& attributeNames, const std::vector<size_t>& fieldByteSize,
    const std::vector<uint8_t>& fieldType, const std::vector<size_t>& fieldCount, PcdDataStorageType dataStorageType,
    char* dataBuffer) :
    attributeNames{attributeNames}, fieldByteSize{fieldByteSize},
    fieldOffset(fieldByteSize.size()), fieldType{fieldType}, fieldCount{fieldCount},
    dataStorageType{dataStorageType}, reader{std::move(reader)}, dataBuffer{dataBuffer}
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
    const uint32_t rgba = bit_cast<uint32_t>(rgba_float);
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
    const auto* const position = dataBuffer + fieldOffset[id];
    
    // lambda function that return a vector of type T or T if count == 1
    auto returnVectorOrSingleValue = [](const auto* dataPtr, auto count) -> PointCloudGenericAttribute {
        if (count == 1) return {*dataPtr};
        return std::vector(dataPtr, dataPtr + count);
    };
    if (count == 1) { // simple value
        // test the type and size
        if (type == 'F') {
            if (size == 4) return fromBytes<float>(position);
            if (size == 8) return fromBytes<double>(position);
        } else if (type == 'I') {
            if (size == 1) return fromBytes<int8_t>(position);
            if (size == 2) return fromBytes<int16_t>(position);
            if (size == 4) return fromBytes<int32_t>(position);
            if (size == 8) return fromBytes<int64_t>(position);
        } else if (type == 'U') {
            if (size == 1) return fromBytes<uint8_t>(position);
            if (size == 2) return fromBytes<uint16_t>(position);
            if (size == 4) return fromBytes<uint32_t>(position);
            if (size == 8) return fromBytes<uint64_t>(position);
        }
    } else { // vector
        // test the type and size
        if (type == 'F') {
            if (size == 4) return vectorFromBytes<float>(position, count);
            if (size == 8) return vectorFromBytes<double>(position, count);
        } else if (type == 'I') {
            if (size == 1) return vectorFromBytes<int8_t>(position, count);
            if (size == 2) return vectorFromBytes<int16_t>(position, count);
            if (size == 4) return vectorFromBytes<int32_t>(position, count);
            if (size == 8) return vectorFromBytes<int64_t>(position, count);
        } else if (type == 'U') {
            if (size == 1) return vectorFromBytes<uint8_t>(position, count);
            if (size == 2) return vectorFromBytes<uint16_t>(position, count);
            if (size == 4) return vectorFromBytes<uint32_t>(position, count);
            if (size == 8) return vectorFromBytes<uint64_t>(position, count);
        }
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

bool PcdPointCloudPoint::gotoNextAscii() {
    static_assert(sizeof(float) == 4 && sizeof(double) == 8);

    // If the reader is at the end of the file or in a bad state, return false
    if (!reader->good()) return false;

    std::string line;
    // Read a non-empty line or return false if no valid lines are found
    do {
        std::getline(*reader, line);
    } while (reader->good() && line.empty());
    if (reader->fail() || line.empty()) return false;

    std::stringstream ss(line);
    std::string token;

    try {
        for (size_t fieldIt = 0; fieldIt < fieldByteSize.size(); ++fieldIt) {
            size_t count = fieldCount[fieldIt];
            size_t size = fieldByteSize[fieldIt];
            char type = fieldType[fieldIt];
            size_t baseOffset = fieldOffset[fieldIt];

            for (size_t countIt = 0; countIt < count; ++countIt) {
                // Read the token
                ss >> token;
                if (ss.fail()) return false;

                auto* position = dataBuffer + baseOffset + countIt * size;

                // Parse and copy data using `std::from_chars`
                if (type == 'F') {
                    if (size == 4) {
                        float value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    } else if (size == 8) {
                        double value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    }
                } else if (type == 'I') {
                    if (size == 1) {
                        int8_t value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    } else if (size == 2) {
                        int16_t value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    } else if (size == 4) {
                        int32_t value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    } else if (size == 8) {
                        int64_t value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    }
                } else if (type == 'U') {
                    if (size == 1) {
                        uint8_t value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    } else if (size == 2) {
                        uint16_t value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    } else if (size == 4) {
                        uint32_t value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    } else if (size == 8) {
                        uint64_t value;
                        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (ec != std::errc{}) return false;
                        std::memcpy(position, &value, size);
                    }
                }
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool PcdPointCloudPoint::gotoNextBinary()
{
    // we just read the next record
    reader->read(dataBuffer, recordByteSize);
    if (!reader->good()) return false;
    return true;
}

bool PcdPointCloudPoint::gotoNextBinaryCompressed()
{
    return false;
}

std::unique_ptr<PointCloudPointAccessInterface> PcdPointCloudPoint::createAdapter(
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterface) {

    // test if nullptr. If so, return nullptr
    if (pointCloudPointAccessInterface == nullptr) {return nullptr;}

    // we test if the interface is already a PcdPointCloudPoint with a dynamic cast
    PcdPointCloudPoint* pcdPoint = dynamic_cast<PcdPointCloudPoint*>(pointCloudPointAccessInterface.get());
    if (pcdPoint != nullptr) {
        // if it is already a PcdPointCloudPointAdapter, we return it
        return std::move(pointCloudPointAccessInterface);
    }

    // cast to SdcPointCloudPoint
    SdcPointCloudPoint* sdcPoint = dynamic_cast<SdcPointCloudPoint*>(pointCloudPointAccessInterface.get());
    if (sdcPoint != nullptr) {
        // create a new PcdPointCloudPointAdapter
        return std::make_unique<PcdPointCloudPointFromSdcAdapter>(std::move(pointCloudPointAccessInterface), sdcPoint);
    }

    // create a new PcdPointCloudPointBasicAdapter
    auto pointCloudPtr = pointCloudPointAccessInterface.get();
    return std::make_unique<PcdPointCloudPointBasicAdapter>(std::move(pointCloudPointAccessInterface), pointCloudPtr);
}

PcdPointCloudHeader::PcdPointCloudHeader(){ }

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
        lineStream.copyfmt(reader);

        std::string headerEntryName;
        size_t nbFields;

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
                    std::streampos &headerHeightPos, std::streampos &headerPointsPos) {
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
    auto vectorToString = [&writer](auto&& c) {
        std::ostringstream ss;
        ss.copyfmt(writer);

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

std::unique_ptr<PointCloudHeaderInterface> PcdPointCloudHeader::createAdapter(std::unique_ptr<PointCloudHeaderInterface> pointCloudHeaderInterface) {
    
    // test if nullptr. If so, return nullptr
    if (pointCloudHeaderInterface == nullptr) {return nullptr;}

    // we test if the interface is already a PcdPointCloudHeader with a dynamic cast
    PcdPointCloudHeader* pcdHeader = dynamic_cast<PcdPointCloudHeader*>(pointCloudHeaderInterface.get());
    if (pcdHeader != nullptr) {
        // if it is already a PcdPointCloudHeaderAdapter, we return a shared pointer with no ownership
        return std::move(pointCloudHeaderInterface);
    } else {
        // create a new PcdPointCloudHeaderAdapter
        return std::make_unique<PcdPointCloudHeaderBasicAdapter>(std::move(pointCloudHeaderInterface));
    }
}

std::vector<std::string> PcdPointCloudHeader::attributeList() const {
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

PcdDataLayout getPcdDataLayoutFromPointcloudPoint(PointCloudPointAccessInterface* pointcloudPointAccessInterface) {

    if (pointcloudPointAccessInterface == nullptr) {return PcdDataLayout{};}

    // try to guess header parameters from the point cloud directly
    auto originalAttributeNames = std::vector<std::string>{};
    auto sanitizedAttributeNames = std::vector<std::string>{};
    auto size = std::vector<size_t>{};
    auto type = std::vector<uint8_t>{};
    auto count = std::vector<size_t>{};
    
    size_t nbAttributes = 0;
    for (auto&& attributeName : pointcloudPointAccessInterface->attributeList()) {
        // try to find the attribute name in the header
        auto attrOpt = pointcloudPointAccessInterface->getAttributeByName(attributeName.c_str()); 

        if (attrOpt) {
            const auto attr = attrOpt.value();
            // visit the variant to get the size, type, count and data
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                // simple types
                if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>) {
                    originalAttributeNames.push_back(attributeName);
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
                    if constexpr (std::is_floating_point_v<V_t> || std::is_integral_v<V_t> || std::is_same_v<V_t, std::byte>) {
                        originalAttributeNames.push_back(attributeName);
                        nbAttributes++;
                        size.push_back(sizeof(V_t));
                        count.push_back(arg.size());
                        if constexpr (std::is_floating_point_v<V_t>) {
                            type.push_back('F');
                        } else if constexpr (std::is_unsigned_v<V_t>) {
                            type.push_back('U');
                        } else if constexpr (std::is_signed_v<V_t>) {
                            type.push_back('I');
                        } else if constexpr (std::is_same_v<V_t, std::byte>) {
                            type.push_back('U'); // byte as uint8_t
                        } else {
                            static_assert(std::is_floating_point_v<V_t> or
                                    std::is_unsigned_v<V_t> or
                                    std::is_signed_v<V_t> or
                                    std::is_same_v<V_t, std::byte>, "All types in the variant must be handled");
                        }
                    } else if constexpr (std::is_same_v<V_t, std::string>) {
                        // PCD cannot store strings...We ignore it
                    } else {
                        static_assert(std::is_same_v<V_t, std::string> or
                                std::is_floating_point_v<V_t> or
                                std::is_integral_v<V_t> or
                                std::is_same_v<V_t, std::byte>, "All types in the variant must be handled");
                    }
                } else {
                    static_assert(is_vector_v<T> or
                            std::is_same_v<T, std::string> or
                            std::is_floating_point_v<T> or
                            std::is_integral_v<T>, "All types in the variant must be handled");
                }
            }, attr);
        }
    }

    // sanitize attribute names and remove duplicates
    std::vector<size_t> toRemove;
    for (size_t i = 0; i < nbAttributes; i++) {
        auto sanitizedAttributeName = sanitizeAttributeNamePcd(originalAttributeNames[i]);
        auto it = std::find(sanitizedAttributeNames.begin(), sanitizedAttributeNames.end(), sanitizedAttributeName);
        if (it != sanitizedAttributeNames.end()) {
            toRemove.push_back(i);
        } else {
            sanitizedAttributeNames.push_back(sanitizedAttributeName);
        }
    }

    // remove duplicates
    std::reverse(toRemove.begin(), toRemove.end());
    for (auto i : toRemove) {
        originalAttributeNames.erase(originalAttributeNames.begin() + i);
        size.erase(size.begin() + i);
        type.erase(type.begin() + i);
        count.erase(count.begin() + i);
    }

    return {sanitizedAttributeNames, originalAttributeNames, size, type, count};
}

std::string sanitizeAttributeNamePcd(const std::string &str) {
    std::string sanitizedStr;

    for (char c : str) {
        if (c == '_'|| std::isalnum(static_cast<unsigned char>(c))) {
            sanitizedStr += c;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            // Replace spaces with underscores
            sanitizedStr += '_';
        }
        // Other characters are ignored
    }

    return sanitizedStr;
}

std::optional<FullPointCloudAccessInterface> openPointCloudPcd(const std::filesystem::path &pcdFilePath) {
   // open the file
    auto reader = std::make_unique<ifstreamCustomBuffer<pcdFileReaderBufferSize>>();

    constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
    reader->precision(maxPrecision); // set the precision for the reader

    reader->open(pcdFilePath, std::ios_base::binary);

    if (!reader->is_open()) return std::nullopt;

    return openPointCloudPcd(std::move(reader));
}

std::optional<FullPointCloudAccessInterface> openPointCloudPcd(std::unique_ptr<std::istream> reader) {
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
    std::optional<PcdDataStorageType> dataStorageType) {
    
   // open the file
    auto writer = std::make_unique<fstreamCustomBuffer<pcdFileWriterBufferSize>>();

    writer->open(pcdFilePath, std::ios_base::in | std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

    if (!writer->is_open()) return false;

    // set the precision to the maximum
    constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
    *writer << std::setprecision(maxPrecision);

    return writePointCloudPcd(*writer, pointCloud, dataStorageType);
}

bool writePointCloudPcd(std::ostream &writer, FullPointCloudAccessInterface &pointCloud,
    std::optional<PcdDataStorageType> dataStorageType) {
    
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

    auto tempHeader = PcdPointCloudHeader::createAdapter(std::move(pointCloud.headerAccess));

    pointCloud.headerAccess = std::move(tempHeader);

    // safe to static cast because we know that the point cloud is a pcd point cloud
    auto pcdHeaderAccessAdapter = static_cast<PcdPointCloudHeader*>(pointCloud.headerAccess.get());

    if (pcdHeaderAccessAdapter != nullptr) {
        headerVersion = pcdHeaderAccessAdapter->version;
        headerWidth = pcdHeaderAccessAdapter->width;
        headerHeight = pcdHeaderAccessAdapter->height;
        headerViewpoint = pcdHeaderAccessAdapter->viewpoint;
        headerPoints = pcdHeaderAccessAdapter->points;
        headerData = pcdHeaderAccessAdapter->data;
    }

    // get the pointcloud point adapter
    pointCloud.pointAccess = PcdPointCloudPoint::createAdapter(std::move(pointCloud.pointAccess));

    // safe to static cast
    auto pcdPointAccessAdapter = static_cast<PcdPointCloudPoint*>(pointCloud.pointAccess.get());
    if (pcdPointAccessAdapter == nullptr) {return false;}

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

    // write the header
    if (!PcdPointCloudHeader::writeHeader(writer, newHeader, headerWidthPos, headerHeightPos, headerPointsPos)) {
        return false;
    }

    size_t nbPoints = 0;
    // write the points
    switch (usedDataStorageType) {
        case PcdDataStorageType::ascii:
            do {
                if (!PcdPointCloudPoint::writePointAscii(writer, *pcdPointAccessAdapter)) return false;
                nbPoints++;
            } while (pcdPointAccessAdapter->gotoNext());
            break;
        case PcdDataStorageType::binary:
            do {
                if (!PcdPointCloudPoint::writePointBinary(writer, *pcdPointAccessAdapter)) return false;
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
        writer.seekp(headerPointsPos);
        writer.write(pointsStr.c_str(), maxSizeStr);
        writer.seekp(headerWidthPos);
        writer.write(widthStr.c_str(), maxSizeStr);
        writer.seekp(headerHeightPos);
        writer.write(heightStr.c_str(), maxSizeStr);
        // seek to the end of the file
        writer.seekp(0, std::ios_base::end);
    }

    return true;
}

PcdPointCloudPointBasicAdapter::PcdPointCloudPointBasicAdapter(
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr,
    PointCloudPointAccessInterface* pointCloudPointAccessInterface,
    const PcdDataLayout& attributeInformations) :
        originalAttributeNames{std::get<1>(attributeInformations)},
        PcdPointCloudPoint{nullptr, std::get<0>(attributeInformations),
            std::get<2>(attributeInformations), std::get<3>(attributeInformations),
            std::get<4>(attributeInformations), PcdDataStorageType::ascii},
        pointCloudPointAccessInterfaceUniquePtr{std::move(pointCloudPointAccessInterfaceUniquePtr)},
        pointCloudPointAccessInterface{pointCloudPointAccessInterface} {
    
    adaptInternalState(); // set the internal state for the first time
}

PcdPointCloudPointBasicAdapter::PcdPointCloudPointBasicAdapter(
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr,
    PointCloudPointAccessInterface* pointCloudPointAccessInterface) :
        PcdPointCloudPointBasicAdapter(std::move(pointCloudPointAccessInterfaceUniquePtr),
        pointCloudPointAccessInterface,
        getPcdDataLayoutFromPointcloudPoint(pointCloudPointAccessInterface)) {
}

PtGeometry<PointCloudGenericAttribute> PcdPointCloudPointBasicAdapter::getPointPosition() const {
    return pointCloudPointAccessInterface->getPointPosition();
}

std::optional<PtColor<PointCloudGenericAttribute>> PcdPointCloudPointBasicAdapter::getPointColor() const {
    return pointCloudPointAccessInterface->getPointColor();
}

std::optional<PointCloudGenericAttribute> PcdPointCloudPointBasicAdapter::getAttributeById(int id) const {
    if (id < 0 || id >= originalAttributeNames.size()) return std::nullopt;
    return pointCloudPointAccessInterface->getAttributeByName(originalAttributeNames[id].c_str());
}

std::optional<PointCloudGenericAttribute> PcdPointCloudPointBasicAdapter::getAttributeByName(const char *attributeName) const {
    // find the attribute in the list of sanitized attribute names
    auto it = std::find(originalAttributeNames.begin(), originalAttributeNames.end(), attributeName);
    if (it == originalAttributeNames.end()) return std::nullopt;
    return pointCloudPointAccessInterface->getAttributeByName((*it).c_str());
}

std::vector<std::string> PcdPointCloudPointBasicAdapter::attributeList() const {
    return attributeNames;
}

bool PcdPointCloudPointBasicAdapter::gotoNext() {
    return pointCloudPointAccessInterface->gotoNext() && adaptInternalState();
}

bool PcdPointCloudPointBasicAdapter::adaptInternalState() {
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
        auto attrOpt = getAttributeById(fieldIt);
        if (!attrOpt.has_value()) return false;
        const auto& attr = attrOpt.value();

            auto* position = dataBuffer + fieldOffset[fieldIt];
            try {
                if (!isAttributeList(attr) && !std::holds_alternative<std::vector<std::byte>>(attr)) {
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

PcdPointCloudHeaderBasicAdapter::PcdPointCloudHeaderBasicAdapter(
    std::unique_ptr<PointCloudHeaderInterface> pointCloudHeaderInterface) :
    PcdPointCloudHeader{} {
    this->pointCloudHeaderInterface = std::move(pointCloudHeaderInterface);

    // try to adapt the header
    adaptInternalState();
}

bool PcdPointCloudHeaderBasicAdapter::adaptInternalState()
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
    writer.write(point.dataBuffer, point.recordByteSize);
    if (!writer.good()) return false;
    return true;
}

bool PcdPointCloudPoint::writePointAscii(std::ostream &writer, const PcdPointCloudPoint &point)
{
    std::ostringstream buffer; // Accumulate output to a buffer for the point
    // copy the formatting:
    buffer.copyfmt(writer);

    // Visitor for handling each field type
    auto visitor = [&buffer](auto &&attr) {
        using T = std::decay_t<decltype(attr)>;
        if constexpr (std::is_same_v<T, std::string> || std::is_floating_point_v<T>) {
            buffer << attr;
        } else if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_signed_v<T>) {
                buffer << static_cast<intmax_t>(attr);
            } else {
                buffer << static_cast<uintmax_t>(attr);
            }
        } else if constexpr (is_vector_v<T>) {
            for (size_t i = 0; i < attr.size(); ++i) {
                if constexpr (std::is_same_v<typename T::value_type, std::string> || 
                              std::is_floating_point_v<typename T::value_type>) {
                    buffer << attr[i];
                } else if constexpr (std::is_integral_v<typename T::value_type>) {
                    if constexpr (std::is_signed_v<typename T::value_type>) {
                        buffer << static_cast<intmax_t>(attr[i]);
                    } else {
                        buffer << static_cast<uintmax_t>(attr[i]);
                    }
                }
                if (i < attr.size() - 1) { 
                    buffer << " "; // Add space after all but the last element
                }
            }
        } else {
            static_assert(is_vector_v<T> or
                    std::is_integral_v<T> or
                    std::is_same_v<T, std::string> or
                    std::is_floating_point_v<T>, "Unsupported type");
        }
    };

    // Precompute size to avoid multiple size() calls
    size_t fieldCount = point.fieldCount.size();
    
    // Write each field
    for (size_t i = 0; i < fieldCount; ++i) {
        std::visit(visitor, point.getAttributeById(i).value());
        if (i < fieldCount - 1) {
            buffer << " "; // Add space between fields
        }
    }

    buffer << std::endl;

    // Write accumulated output to writer
    writer << buffer.str();
    return writer.good();
}

PcdPointCloudPointFromSdcAdapter::PcdPointCloudPointFromSdcAdapter(
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointInterface, SdcPointCloudPoint* castedSdcPointCloudPoint):
    PcdPointCloudPointFromSdcAdapter(std::move(pointCloudPointInterface), castedSdcPointCloudPoint,
    getPcdDataLayoutFromPointcloudPoint(castedSdcPointCloudPoint)) {}


bool PcdPointCloudPointFromSdcAdapter::gotoNext() {
   return castedSdcPointCloudPoint->gotoNext();
}

PcdPointCloudPointFromSdcAdapter::PcdPointCloudPointFromSdcAdapter(
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointInterface,
        SdcPointCloudPoint* castedSdcPointCloudPoint, const PcdDataLayout& attributeInformations) :
        castedSdcPointCloudPoint{castedSdcPointCloudPoint},
        PcdPointCloudPoint{nullptr, std::get<0>(attributeInformations), std::get<2>(attributeInformations),
            std::get<3>(attributeInformations), std::get<4>(attributeInformations), PcdDataStorageType::ascii,
            castedSdcPointCloudPoint->getRecordDataBuffer()},
            pointCloudPointInterface{std::move(pointCloudPointInterface)} {
    
}

} // namespace IO
} // namespace StereoVision
