#include <cstring>
#include <array>
#include <iostream>
#include <cstddef>
#include <cmath>
#include "las_pointcloud_io.h"
#include "bit_manipulations.h"
#include "fstreamCustomBuffer.h"

namespace StereoVision {
namespace IO {

using namespace std::literals::string_view_literals;

// constants
// buffersize when reading las files
constexpr static size_t lasFileReaderBufferSize = 1 << 16;
// write
constexpr static size_t lasFileWriterBufferSize = 1 << 16;

// private constexpr function to compute the offset from an array of sizes
template <size_t N, typename T = size_t, size_t N_MAX = N, const std::array<T, N_MAX>& sizes, const std::array<bool, N_MAX>& usePriorDataOffset>
static constexpr size_t computeOffset() {
    if constexpr (N == 0) { // base case
        return 0; 
    } else if constexpr (std::get<N>(usePriorDataOffset)) { // if usePriorDataOffset is true, same offset than previous data
        return computeOffset<N-1, T, N_MAX, sizes, usePriorDataOffset>();
    } else {
        return std::get<N - 1>(sizes) + computeOffset<N-1, T, N_MAX, sizes, usePriorDataOffset>(); // add size to the last offset
    }
}

template <size_t N, const auto& sizes, const auto& usePriorDataOffset>
static constexpr size_t computeOffset() {
    return computeOffset<N, size_t, sizes.size(), sizes, usePriorDataOffset>();
}

/**
 * @brief Factory function that generate a LasPointCloudPoint object given a istream, the recordByteSize, the record format number (0-11) and the dataBuffer.
 * @param reader the istream
 * @param recordByteSize the size of the point record
 * @param nbPoints the total number of points in the pointcloud
 * @param recordFormatNumber the format number
 * @param extraAttributesNames the names of the extra attributes
 * @param extraAttributesTypes the types of the extra attributes
 * @param extraAttributesSizes the sizes of the extra attributes
 * @param extraAttributesOffsets the offsets of the extra attributes
 * @param hideColorAndGeometricAttributes if true, the color and geometric attributes will be hidden to the user
 * @param dataBuffer the data buffer to use to read the point to
 * 
 * @return std::unique_ptr<LasPointCloudPoint>
 * extraAttributesNames, extraAttributesTypes, extraAttributesSizes, extraAttributesOffsets
 */
static std::unique_ptr<LasPointCloudPoint> createLasPointCloudPoint(std::unique_ptr<std::istream> reader,
    size_t recordByteSize, size_t nbPoints, size_t recordFormatNumber,
    const std::vector<std::string>& extraAttributesNames = {}, const std::vector<uint8_t>& extraAttributesTypes = {},
    const std::vector<size_t>& extraAttributesSizes = {}, const std::vector<size_t>& extraAttributesOffsets = {},
    bool hideColorAndGeometricAttributes = false, double xScaleFactor = 0.01, double yScaleFactor = 0.01,
    double zScaleFactor = 0.01, double xOffset = 0, double yOffset = 0, double zOffset = 0, char* dataBuffer = nullptr);

template <size_t N = size_t{0}>
static std::unique_ptr<LasPointCloudPoint> createLasPointCloudPoint_helper(std::unique_ptr<std::istream> reader,
    size_t recordByteSize, size_t nbPoints, size_t recordFormatNumber, const std::vector<std::string> &extraAttributesNames,
    const std::vector<uint8_t> &extraAttributesTypes, const std::vector<size_t> &extraAttributesSizes,
    const std::vector<size_t> &extraAttributesOffsets, bool hideColorAndGeometricAttributes, double xScaleFactor,
    double yScaleFactor, double zScaleFactor, double xOffset, double yOffset, double zOffset, char* dataBuffer);

struct LasDataLayout {
    size_t format;
    size_t minimumNumberOfAttributes; // the minimum number of attributes for this format
    size_t recordByteSize;
    std::vector<std::string> attributeNames;
    std::vector<uint8_t> fieldType;
    std::vector<size_t> fieldByteSize;
    std::vector<size_t> fieldOffset;
    std::vector<bool> usePriorDataOffset;
    std::vector<bool> isBitfield;
    std::vector<size_t> bitfieldSize;
    std::vector<size_t> bitfieldOffset;
};

template <class D>
class LasPointCloudPoint_Base : public LasPointCloudPoint {
public:
    // structure to get the return type for each attribute from the derived class
    template <class Derived, size_t N>
    struct GetReturnType {
        using type = typename std::tuple_element<N, typename Derived::returnTypeList>::type;
    };

    // structure to get the record format number
    template<class Derived>
    struct GetRecordFormatNumber;

    template<template<size_t> class DerivedTemplate, size_t RecordFormatNumber>
    struct GetRecordFormatNumber<DerivedTemplate<RecordFormatNumber>> {
        static constexpr size_t value = RecordFormatNumber;
    };

    static constexpr size_t recordFormatNumber = GetRecordFormatNumber<D>::value;

    // tell if the format is a legacy format
    static constexpr bool isLegacyFormat = recordFormatNumber <= 5;

    static constexpr bool containsColor = recordFormatNumber == 2 || recordFormatNumber == 3 || recordFormatNumber == 5
        || recordFormatNumber == 7 || recordFormatNumber == 8 || recordFormatNumber == 10;

    static constexpr bool containsGPS = recordFormatNumber != 0 && recordFormatNumber != 2;

    static constexpr bool containsWavePacket = recordFormatNumber == 4 || recordFormatNumber == 5
        || recordFormatNumber == 9 || recordFormatNumber == 10;

    static constexpr bool containsNIR = recordFormatNumber == 8 || recordFormatNumber == 10;

    // we use std::tuple to store the return type for each attribute
    template<size_t N>
    using returnType = typename GetReturnType<D, N>::type;

    // get the byte size of the field's data in the buffer
    template<size_t N>
    static constexpr size_t size = std::get<N>(D::fieldByteSize);

    // get the offset of the data in the buffer
    template<size_t N>
    static constexpr size_t offset = computeOffset<N, D::fieldByteSize, D::usePriorDataOffset>();

    // if the type is a vector, get the number of elements. If it's a string, get the maximum number of characters for the string.
    template<size_t N>
    static constexpr size_t count = std::get<N>(D::fieldCount);

    // tell if the attribute is in a bitfield
    template<size_t N>
    static constexpr bool isBitfield = std::get<N>(D::isBitfield);

    // the size of the bitfield
    template<size_t N>
    static constexpr size_t bitfieldSize = std::get<N>(D::bitfieldSize);

    // the offset in bits of the bitfield
    template<size_t N>
    static constexpr size_t bitfieldOffset = std::get<N>(D::bitfieldOffset);

    static constexpr size_t minimumRecordByteSize = size<D::nbAttributes - 1> + offset<D::nbAttributes - 1>;
protected:
    const std::unique_ptr<std::istream> reader;
    std::vector<std::string> extraAttributesNames;
    std::vector<uint8_t> extraAttributesTypes;
    std::vector<size_t> extraAttributesSizes;
    std::vector<size_t> extraAttributesOffsets;
    
    const size_t nbPoints;
    size_t currentPointIdOneBased = 0; // zero for non initialization, one for the first point

    // attribute names exposed to the user. For example, we might hide the color and geometric attributes.
    std::vector<std::string> exposedAttributeNames;
    // the full list of attribute names including the one hidden to the user
    std::vector<std::string> attributeNamesFullListInternal;
    std::vector<size_t> exposedIdToInternalId; // map the exposed id of an attribute to its internal id
public:
    LasPointCloudPoint_Base(std::unique_ptr<std::istream> reader, size_t recordByteSize, size_t nbPoints,
        const std::vector<std::string> &extraAttributesNames, const std::vector<uint8_t> &extraAttributesTypes,
        const std::vector<size_t> &extraAttributesSizes, const std::vector<size_t> &extraAttributesOffsets,
        bool hideColorAndGeometricAttributes, double xScaleFactor, double yScaleFactor, double zScaleFactor, double xOffset,
        double yOffset, double zOffset, char* dataBuffer) :
            reader{std::move(reader)},
            nbPoints{nbPoints},
            extraAttributesNames{extraAttributesNames},
            extraAttributesTypes{extraAttributesTypes},
            extraAttributesSizes{extraAttributesSizes},
            extraAttributesOffsets{extraAttributesOffsets},
            LasPointCloudPoint(recordByteSize, xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset,
                dataBuffer) {
        //do some checks on the extra parameters and correct them is needed
        verifyAndCorrectParameters(hideColorAndGeometricAttributes);
    }

    LasPointCloudPoint_Base(std::unique_ptr<std::istream> reader, size_t recordByteSize, size_t nbPoints,
        const std::vector<std::string> &extraAttributesNames, const std::vector<uint8_t> &extraAttributesTypes,
        const std::vector<size_t> &extraAttributesSizes,
        const std::vector<size_t> &extraAttributesOffsets, bool hideColorAndGeometricAttributes,
        double xScaleFactor, double yScaleFactor, double zScaleFactor, double xOffset, double yOffset, double zOffset) :
            reader{std::move(reader)},
            nbPoints{nbPoints},
            extraAttributesNames{extraAttributesNames},
            extraAttributesTypes{extraAttributesTypes},
            extraAttributesSizes{extraAttributesSizes},
            extraAttributesOffsets{extraAttributesOffsets},
            LasPointCloudPoint(recordByteSize, xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset) {
        //do some checks on the extra parameters and correct them is needed
        verifyAndCorrectParameters(hideColorAndGeometricAttributes);
    }

    template <size_t N = 0>
    static uint8_t getLasDataType(size_t id);

    template <typename T>
    static uint8_t getLasDataType();

    template <size_t N = 0>
    static size_t getFieldOffset(size_t id);


    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    bool gotoNext() override;

    size_t getFormat() const override { return recordFormatNumber; }

    size_t getMinimumNumberOfAttributes() const override { return D::nbAttributes; }

    LasExtraAttributesInfos getExtraAttributesInfos() const override;
private:

    /**
     * @brief helper functions to get the attribute by id.
     * 
     * If N > nbAttributes, return nullopt.
     * If N == id, return the attribute.
     * If N != id, recursively call the function with N+1.
     */
    template<size_t N = size_t{0}>
    std::optional<PointCloudGenericAttribute> getAttributeById_helper(size_t id) const;

    /**
     * @brief do some checks on the parameters and correct them if needed. It can also hide the color and geometric
     * attributes (x, y, z, red, green, blue)
     * 
     * @param hideColorAndGeometricAttributes If true, does not expose the attributes x, y, z, red, green, blue
     */

    void verifyAndCorrectParameters(bool hideColorAndGeometricAttributes);
    
};

// the different record formats

template<size_t RecordFormatNumber>
class LasPointCloudPoint_Format;

template<>
class LasPointCloudPoint_Format<0> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<0>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<0>>::LasPointCloudPoint_Base;

    static constexpr size_t nbAttributes = 15;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, int8_t, uint8_t, uint16_t>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
        "keyPointFlag"sv, "withheldFlag"sv, "scanAngleRank"sv, "userData"sv, "pointSourceID"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 3, 3, 1, 1, 5, 1, 1, 1, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 3, 6, 7, 0, 5, 6, 7, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;
};

template<>
class LasPointCloudPoint_Format<1> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<1>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<1>>::LasPointCloudPoint_Base;

    static constexpr size_t nbAttributes = 16;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, int8_t, uint8_t, uint16_t, double>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
        "keyPointFlag"sv, "withheldFlag"sv, "scanAngleRank"sv, "userData"sv, "pointSourceID"sv, "GPSTime"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 3, 3, 1, 1, 5, 1, 1, 1, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 3, 6, 7, 0, 5, 6, 7, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;
};

template<>
class LasPointCloudPoint_Format<2> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<2>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<2>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 18;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, int8_t, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
        "keyPointFlag"sv, "withheldFlag"sv, "scanAngleRank"sv, "userData"sv, "pointSourceID"sv, "red"sv, "green"sv,
        "blue"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 3, 3, 1, 1, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 3, 6, 7, 0, 5, 6, 7, 0, 0, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;

    static constexpr auto r_id = 15;
    static constexpr auto g_id = 16;
    static constexpr auto b_id = 17;
};

template<>
class LasPointCloudPoint_Format<3> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<3>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<3>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 19;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, int8_t, uint8_t, uint16_t, double, uint16_t, uint16_t, uint16_t>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
        "keyPointFlag"sv, "withheldFlag"sv, "scanAngleRank"sv, "userData"sv, "pointSourceID"sv, "GPSTime"sv, "red"sv,
        "green"sv, "blue"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 2, 2, 2};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 3, 3, 1, 1, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 3, 6, 7, 0, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;

    static constexpr auto r_id = 16;
    static constexpr auto g_id = 17;
    static constexpr auto b_id = 18;
};

template<>
class LasPointCloudPoint_Format<4> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<4>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<4>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 23;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, int8_t, uint8_t, uint16_t, double, uint8_t, uint64_t, uint32_t, float,
        float, float, float>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
        "keyPointFlag"sv, "withheldFlag"sv, "scanAngleRank"sv, "userData"sv, "pointSourceID"sv, "GPSTime"sv,
        "wavePacketDescriptorIndex"sv, "byteOffsetToWaveformData"sv, "waveformPacketSizeInBytes"sv,
        "returnPointWaveformLocation"sv, "parametricDx"sv, "parametricDy"sv, "parametricDz"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 1, 8, 4, 4, 4, 4, 4};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 3, 3, 1, 1, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 3, 6, 7, 0, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;
};

template<>
class LasPointCloudPoint_Format<5> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<5>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<5>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 26;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, int8_t, uint8_t, uint16_t, double, uint16_t, uint16_t, uint16_t, uint8_t,
        uint64_t, uint32_t, float, float, float, float>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
        "keyPointFlag"sv, "withheldFlag"sv, "scanAngleRank"sv, "userData"sv, "pointSourceID"sv, "GPSTime"sv, "red"sv,
        "green"sv, "blue"sv, "wavePacketDescriptorIndex"sv, "byteOffsetToWaveformData"sv, "waveformPacketSizeInBytes"sv,
        "returnPointWaveformLocation"sv, "parametricDx"sv, "parametricDy"sv, "parametricDz"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 2, 2, 2, 1, 8, 4, 4, 4, 4, 4};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 3, 3, 1, 1, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 3, 6, 7, 0, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;

    static constexpr auto r_id = 16;
    static constexpr auto g_id = 17;
    static constexpr auto b_id = 18;
};

template<>
class LasPointCloudPoint_Format<6> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<6>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<6>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 18;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, int16_t, uint16_t, double>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
        "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "userData"sv, "scanAngle"sv,
        "pointSourceID"sv, "GPSTime"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 8};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 4, 4, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 4, 0, 1, 2, 3, 4, 6, 7, 0, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;
};

template<>
class LasPointCloudPoint_Format<7> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<7>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<7>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 21;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, int16_t, uint16_t, double, uint16_t, uint16_t, uint16_t>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
        "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "userData"sv, "scanAngle"sv,
        "pointSourceID"sv, "GPSTime"sv, "red"sv, "green"sv, "blue"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 8, 2, 2, 2};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 4, 4, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 4, 0, 1, 2, 3, 4, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;

    static constexpr size_t r_id = 18;
    static constexpr size_t g_id = 19;
    static constexpr size_t b_id = 20;
};

template<>
class LasPointCloudPoint_Format<8> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<8>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<8>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 22;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, int16_t, uint16_t, double, uint16_t, uint16_t, uint16_t,
        uint16_t>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
        "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "userData"sv, "scanAngle"sv,
        "pointSourceID"sv, "GPSTime"sv, "red"sv, "green"sv, "blue"sv, "NIR"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 8, 2, 2, 2, 2};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 4, 4, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 4, 0, 1, 2, 3, 4, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0};


    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;

    static constexpr size_t r_id = 18;
    static constexpr size_t g_id = 19;
    static constexpr size_t b_id = 20;
};

template<>
class LasPointCloudPoint_Format<9> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<9>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<9>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 25;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, int16_t, uint16_t, double, uint8_t, uint64_t, uint32_t,
        float, float, float, float>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
        "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "userData"sv, "scanAngle"sv,
        "pointSourceID"sv, "GPSTime"sv, "wavePacketDescriptorIndex"sv,
        "byteOffsetToWaveformData"sv, "waveformPacketSizeInBytes"sv, "returnPointWaveformLocation"sv, "parametricDx"sv,
        "parametricDy"sv, "parametricDz"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 8, 1, 8, 4, 4, 4, 4, 4};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 4, 4, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 4, 0, 1, 2, 3, 4, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;
};

template<>
class LasPointCloudPoint_Format<10> : public LasPointCloudPoint_Base<LasPointCloudPoint_Format<10>> {
public:
    using LasPointCloudPoint_Base<LasPointCloudPoint_Format<10>>::LasPointCloudPoint_Base;
    
    static constexpr size_t nbAttributes = 29;

    using returnTypeList = std::tuple<int32_t, int32_t, int32_t, uint16_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, int16_t, uint16_t, double, uint16_t, uint16_t, uint16_t,
        uint16_t, uint8_t, uint64_t, uint32_t, float, float, float, float>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "returnNumber"sv,
        "numberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
        "scanDirectionFlag"sv, "edgeOfFlightLineFlag"sv, "classification"sv, "userData"sv, "scanAngle"sv,
        "pointSourceID"sv, "GPSTime"sv, "red"sv, "green"sv, "blue"sv, "NIR"sv, "wavePacketDescriptorIndex"sv,
        "byteOffsetToWaveformData"sv, "waveformPacketSizeInBytes"sv, "returnPointWaveformLocation"sv, "parametricDx"sv,
        "parametricDy"sv, "parametricDz"sv};

    constexpr static std::array<size_t, nbAttributes> fieldByteSize =    {4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 8, 2, 2, 2, 2, 1, 8, 4, 4, 4, 4, 4};
    constexpr static std::array<size_t, nbAttributes> fieldCount =       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // infos related to bitfields
    constexpr static std::array<bool, nbAttributes> usePriorDataOffset = {0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<bool, nbAttributes> isBitfield =         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldSize =     {0, 0, 0, 0, 4, 4, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<size_t, nbAttributes> bitfieldOffset =   {0, 0, 0, 0, 0, 4, 0, 1, 2, 3, 4, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr size_t x_id = 0;
    static constexpr size_t y_id = 1;
    static constexpr size_t z_id = 2;

    static constexpr size_t r_id = 18;
    static constexpr size_t g_id = 19;
    static constexpr size_t b_id = 20;
};

class LasPointCloudPointBasicAdapter : public LasPointCloudPoint {
protected:
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr = nullptr;
    PointCloudPointAccessInterface* pointCloudPointAccessInterface = nullptr;

    const size_t format;
    const size_t minimumNumberOfAttributes;
    const std::vector<std::string> attributeNames;
    const std::vector<uint8_t> fieldType;
    const std::vector<size_t> fieldByteSize;
    const std::vector<size_t> fieldOffset;
    const std::vector<bool> usePriorDataOffset;
    const std::vector<bool> isBitfield;
    const std::vector<size_t> bitfieldSize;
    const std::vector<size_t> bitfieldOffset;

    size_t xIndex;
    size_t yIndex;
    size_t zIndex;

    bool containsColor;

    size_t redIndex;
    size_t greenIndex;
    size_t blueIndex;

    // stored position and color
    PtGeometry<PointCloudGenericAttribute> pointPosition;
    PtGeometry<int32_t> pointPositionScaledInteger;
    std::optional<PtColor<PointCloudGenericAttribute>> pointColor;
public:
    /**
     * @brief adapt a point cloud point access interface to the LasPointCloudPoint interface.
     * 
     * @param pointCloudPointAccessInterfaceUniquePtr A unique pointer to the interface.
     * @param pointCloudPointAccessInterface  A pointer to the interface. If pointCloudPointAccessInterfaceUniquePtr
     * is not null, it should point to the same object.
     */
    LasPointCloudPointBasicAdapter(
        std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr,
        PointCloudPointAccessInterface* pointCloudPointAccessInterface, double xScaleFactor, double yScaleFactor,
        double zScaleFactor, double xOffset, double yOffset, double zOffset);

    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override;

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;
    
    size_t getFormat() const override { return format; }

    size_t getMinimumNumberOfAttributes() const override { return minimumNumberOfAttributes; }

    LasExtraAttributesInfos getExtraAttributesInfos() const override;

    bool gotoNext() override;

    /**
     * @brief Function that return the format number suiteable for the given pointCloudPointAccessInterface
     * 
     * @param pointCloudPointAccessInterface The point cloud point access interface
     * @param defaultFormat The default format to use if none is found
     * @return const size_t The format
     */
    static size_t getSuitableFormat(const PointCloudPointAccessInterface& pointCloudPointAccessInterface,
        const std::optional<size_t>& defaultFormat = std::nullopt);

    /** @brief From any pointcloud point, get the sanitized attributes names, the original names of the attributes,
     * their byte size, their type and their count
     * @param pointcloudPointAccessInterface The pointcloud point access interface
     * @return A struct containing the sanitized attributes names, original attributes names, their byte size, their type and their count
     */
    static LasDataLayout getLasDataLayoutFromPointcloudPoint(PointCloudPointAccessInterface* pointcloudPointAccessInterface);
protected:

    LasPointCloudPointBasicAdapter(
        std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr,
        PointCloudPointAccessInterface* pointCloudPointAccessInterface, double xScaleFactor, double yScaleFactor,
        double zScaleFactor, double xOffset, double yOffset, double zOffset,
        const LasDataLayout& attributeInformations);

    /**
     * @brief cast a PointCloudGenericAttribute to the given lasType as defined in hte extra bytes vlr descriptor
     * 
     * @param attribute the attribute to cast 
     * @param lasType the las type
     * @param numberUndocumentedBytes the number of undocumented bytes of the attribute if the type is a sequence of bytes
     * @return * PointCloudGenericAttribute the casted attribute
     */
    static PointCloudGenericAttribute castAttributeToLasType(const PointCloudGenericAttribute& attribute, uint8_t lasType,
    std::optional<size_t> numberUndocumentedBytes = std::nullopt);
    
    /**
     * @brief fill the internal state with the values of the attributes of the current point.
     * 
     * @return true if the internal state was properly adapted, false otherwise
     */
    bool adaptInternalState();

private:
    template<size_t Format = 0>
    static LasDataLayout getLasDataLayoutFromPointcloudPoint_helper(const PointCloudPointAccessInterface& pointcloudPointAccessInterface, size_t format);
};

std::optional<PointCloudGenericAttribute> LasPointCloudHeader::getAttributeById(int id) const {
    // TODO: add support for VLRs and EVLRs
    if (id < 0) {
        return std::nullopt;
    } else if (id < publicHeaderBlock.publicHeaderAttributeList().size()) {
            return publicHeaderBlock.getPublicHeaderAttributeById(id);
    } else {
        return std::nullopt;
    }
}

std::optional<PointCloudGenericAttribute> LasPointCloudHeader::getAttributeByName(const char *attributeName) const {
    // TODO: add support for VLRs and EVLRs
    auto it = std::find(publicHeaderBlock.publicHeaderAttributeList().begin(), publicHeaderBlock.publicHeaderAttributeList().end(), attributeName);
    if (it != publicHeaderBlock.publicHeaderAttributeList().end()) {
        auto id = std::distance(publicHeaderBlock.publicHeaderAttributeList().begin(), it);
        return publicHeaderBlock.getPublicHeaderAttributeById(id);
    } else {
        return std::nullopt;
    }
}
std::vector<std::string> LasPointCloudHeader::attributeList() const {
    // TODO: add support for VLRs and EVLRs
    return publicHeaderBlock.publicHeaderAttributeList();
}
LasExtraAttributesInfos LasPointCloudHeader::getPointwiseExtraAttributesInfos(bool ignoreUndocumentedExtraBytes) const {
    // generate the list of extra bytes descriptors
    auto extraBytesDescriptors = extraBytesDescriptorsFromVlrs(variableLengthRecords, extendedVariableLengthRecords);
    // generate the list of extra bytes names, data types, sizes and offsets
    return generateExtraBytesInfo(extraBytesDescriptors, ignoreUndocumentedExtraBytes);
}

std::unique_ptr<LasPointCloudHeader> LasPointCloudHeader::readHeader(std::istream &reader) {
    auto header = std::make_unique<LasPointCloudHeader>();
    auto publicHeader = LasPublicHeaderBlock::readPublicHeader(reader);
    if (!publicHeader) return nullptr;
    header->publicHeaderBlock = std::move(*publicHeader);
    auto vlrs = LasVariableLengthRecord::readVariableLengthRecords(reader, publicHeader->numberOfVariableLengthRecords);
    if (!vlrs) return nullptr;
    header->variableLengthRecords = std::move(*vlrs);
    // test if there is EVLRs
    if (publicHeader->numberOfExtendedVariableLengthRecords > 0) {
        // move the reader to the start of the EVLRs
        reader.seekg(publicHeader->startOfFirstExtendedVariableLengthRecord);
        if (reader.fail()) return nullptr;
        // read the EVLRs
        auto evlrs = LasExtendedVariableLengthRecord::readVariableLengthRecords(reader, publicHeader->numberOfExtendedVariableLengthRecords);
        if (!evlrs) return nullptr;
        header->extendedVariableLengthRecords = std::move(*evlrs);
    }

    // seek the reader to the start of the points
    reader.seekg(publicHeader->offsetToPointData);
    if (reader.fail()) return nullptr;

    return header;
}

bool LasPointCloudHeader::writeVLRs(std::ostream &writer) const {
    for (const auto &vlr : variableLengthRecords) {
        // write the VLRs header and data
        writer.write(reinterpret_cast<const char*>(vlr.getHeaderData().data()), vlr.getHeaderData().size());
        writer.write(reinterpret_cast<const char*>(vlr.getData().data()), vlr.getData().size());
    }
    return writer.good();
}

bool LasPointCloudHeader::writeEVLRs(std::ostream &writer) const {
    for (const auto &evlr : extendedVariableLengthRecords) {
        // write the EVLRs header and data
        writer.write(reinterpret_cast<const char*>(evlr.getHeaderData().data()), evlr.getHeaderData().size());
        writer.write(reinterpret_cast<const char*>(evlr.getData().data()), evlr.getData().size());
    }
    return writer.good();
}

std::vector<LasExtraBytesDescriptor> LasPointCloudHeader::extraBytesDescriptorsFromVlrs(
    const std::vector<LasVariableLengthRecord> &variableLengthRecords,
    const std::vector<LasExtendedVariableLengthRecord> &extendedVariableLengthRecords) {

    std::vector<LasExtraBytesDescriptor> extraBytesDescriptors;
    
    bool extraBytesVlrFound = false;
    size_t nbExtraBytesDescriptors = 0;

    const std::byte* vlrData;
    for (const auto &vlr : variableLengthRecords) {
        if (vlr.getUserId() == "LASF_Spec" && vlr.getRecordId() == 4) {
            extraBytesVlrFound = true;
            vlrData = vlr.getData().data();
            nbExtraBytesDescriptors = vlr.getRecordLengthAfterHeader() / 192;
            break;
        }
    }

    for (const auto &evlr : extendedVariableLengthRecords) {
        if (evlr.getUserId() == "LASF_Spec" && evlr.getRecordId() == 4) {
            extraBytesVlrFound = true;
            vlrData = evlr.getData().data();
            nbExtraBytesDescriptors = evlr.getRecordLengthAfterHeader() / 192;
            break;
        }
    }

    if (extraBytesVlrFound) {
        for (size_t i = 0; i < nbExtraBytesDescriptors; i++) {
            extraBytesDescriptors.push_back(LasExtraBytesDescriptor{vlrData + i * 192});
        }
    }

    return extraBytesDescriptors;
}

LasExtraAttributesInfos LasPointCloudHeader::generateExtraBytesInfo(
    const std::vector<LasExtraBytesDescriptor> &extraBytesDescriptors,
    bool ignoreUndocumentedExtraBytes) {

    std::vector<std::string> extraBytesNames;
    std::vector<uint8_t> extraBytesTypes;
    std::vector<size_t> extraBytesSizes;
    std::vector<size_t> extraBytesOffsets;

    size_t currentOffset = 0;
    // iterate over the extra bytes descriptors
    for (const auto &descriptor : extraBytesDescriptors) {
        size_t size;
        // test if the descriptor is valid
        if (descriptor.data_type > 10) { // deprecated of reserved. Since we do not know the size of the data, we cannot read the next descriptors
            break;
        } else if (descriptor.data_type == 0) { // undocumented data types
            size = descriptor.options;
        } else {
            switch (descriptor.data_type) {
                case 1: // uint8_t
                    size = 1;
                    break;
                case 2: // int8_t
                    size = 1;
                    break;
                case 3: // uint16_t
                    size = 2;
                    break;
                case 4: // int16_t
                    size = 2;
                    break;
                case 5: // uint32_t
                    size = 4;
                    break;
                case 6: // int32_t
                    size = 4;
                    break;
                case 7: // uint64_t
                    size = 8;
                    break;
                case 8: // int64_t
                    size = 8;
                    break;
                case 9: // float
                    size = 4;
                    break;
                case 10: // double
                    size = 8;
                    break;
                default:
                    size = 1;
            }
        }

        if (descriptor.data_type != 0 || !ignoreUndocumentedExtraBytes) {
            auto endName = std::find(descriptor.name.begin(), descriptor.name.end(), '\0'); // find the end of the attribute name
            extraBytesNames.push_back(std::string{descriptor.name.begin(), endName});
            extraBytesTypes.push_back(descriptor.data_type);
            extraBytesSizes.push_back(size);
            extraBytesOffsets.push_back(currentOffset);
        }

        currentOffset += size;
    }

    return {extraBytesNames, extraBytesTypes, extraBytesSizes, extraBytesOffsets};
}

std::vector<std::byte> LasPointCloudHeader::generateExtraBytesVlrData(const LasExtraAttributesInfos &extraAttributesInfos){
    // make sure the vector size is valid
    auto nbExtraAttributes = extraAttributesInfos.name.size();
    if (   nbExtraAttributes != extraAttributesInfos.type.size() 
        || nbExtraAttributes!= extraAttributesInfos.size.size()
        || nbExtraAttributes!= extraAttributesInfos.offset.size()) {
        
        return {};
    }

    std::vector<std::byte> data;
    data.reserve(nbExtraAttributes * 192);

    size_t prevOffset{};
    size_t prevSize{};

    for (size_t i = 0; i < nbExtraAttributes; i++) {
        size_t expectedOffset = prevOffset + prevSize;
        if (expectedOffset != extraAttributesInfos.offset[i]) {
            if (expectedOffset > extraAttributesInfos.offset[i]) return {};
            // empty extra byte descriptor to fill the gap
            auto descriptorData = LasExtraBytesDescriptor{0, "", extraAttributesInfos.offset[i] - expectedOffset}.toBytes();
            data.insert(data.end(), descriptorData.begin(), descriptorData.end());
        }
        // write the extra byte descriptor
        auto descriptorData = LasExtraBytesDescriptor{extraAttributesInfos.type[i], extraAttributesInfos.name[i],
            extraAttributesInfos.size[i]}.toBytes();
        data.insert(data.end(), descriptorData.begin(), descriptorData.end());
        prevOffset = extraAttributesInfos.offset[i];
        prevSize = extraAttributesInfos.size[i];
    }

    return data;
}

LasPublicHeaderBlock::LasPublicHeaderBlock() {
    fileSignature = {'L','A','S','F'};
    versionMajor = 1;
    versionMinor = 4;
    systemIdentifier = {"OTHER"};
    generatingSoftware = {"LibStevi"};
    headerSize = 375;
    offsetToPointData = headerSize;
    pointDataRecordFormat = 6;
    pointDataRecordLength = LasPointCloudPoint_Format<6>::minimumRecordByteSize;
    startOfFirstExtendedVariableLengthRecord = offsetToPointData;
}

std::optional<LasPublicHeaderBlock> LasPublicHeaderBlock::readPublicHeader(std::istream &reader)
{
    LasPublicHeaderBlock header;
    reader.read(header.fileSignature.data(), fileSignature_size);
    reader.read(reinterpret_cast<char*>(&header.fileSourceID), fileSourceID_size);
    reader.read(reinterpret_cast<char*>(&header.globalEncoding), globalEncoding_size);
    reader.read(reinterpret_cast<char*>(&header.projectID_GUID_Data1), projectID_GUID_Data1_size);
    reader.read(reinterpret_cast<char*>(&header.projectID_GUID_Data2), projectID_GUID_Data2_size);
    reader.read(reinterpret_cast<char*>(&header.projectID_GUID_Data3), projectID_GUID_Data3_size);
    reader.read(reinterpret_cast<char*>(header.projectID_GUID_Data4.data()), projectID_GUID_Data4_size);
    reader.read(reinterpret_cast<char*>(&header.versionMajor), versionMajor_size);
    reader.read(reinterpret_cast<char*>(&header.versionMinor), versionMinor_size);
    reader.read(reinterpret_cast<char*>(header.systemIdentifier.data()), systemIdentifier_size);
    reader.read(reinterpret_cast<char*>(header.generatingSoftware.data()), generatingSoftware_size);
    reader.read(reinterpret_cast<char*>(&header.fileCreationDayOfYear), fileCreationDayOfYear_size);
    reader.read(reinterpret_cast<char*>(&header.fileCreationYear), fileCreationYear_size);
    reader.read(reinterpret_cast<char*>(&header.headerSize), headerSize_size);
    reader.read(reinterpret_cast<char*>(&header.offsetToPointData), offsetToPointData_size);
    reader.read(reinterpret_cast<char*>(&header.numberOfVariableLengthRecords), numberOfVariableLengthRecords_size);
    reader.read(reinterpret_cast<char*>(&header.pointDataRecordFormat), pointDataRecordFormat_size);
    reader.read(reinterpret_cast<char*>(&header.pointDataRecordLength), pointDataRecordLength_size);
    reader.read(reinterpret_cast<char*>(&header.legacyNumberOfPointRecords), legacyNumberOfPointRecords_size);
    reader.read(reinterpret_cast<char*>(header.legacyNumberOfPointsByReturn.data()), legacyNumberOfPointsByReturn_size);
    reader.read(reinterpret_cast<char*>(&header.xScaleFactor), xScaleFactor_size);
    reader.read(reinterpret_cast<char*>(&header.yScaleFactor), yScaleFactor_size);
    reader.read(reinterpret_cast<char*>(&header.zScaleFactor), zScaleFactor_size);
    reader.read(reinterpret_cast<char*>(&header.xOffset), xOffset_size);
    reader.read(reinterpret_cast<char*>(&header.yOffset), yOffset_size);
    reader.read(reinterpret_cast<char*>(&header.zOffset), zOffset_size);
    reader.read(reinterpret_cast<char*>(&header.maxX), maxX_size);
    reader.read(reinterpret_cast<char*>(&header.minX), minX_size);
    reader.read(reinterpret_cast<char*>(&header.maxY), maxY_size);
    reader.read(reinterpret_cast<char*>(&header.minY), minY_size);
    reader.read(reinterpret_cast<char*>(&header.maxZ), maxZ_size);
    reader.read(reinterpret_cast<char*>(&header.minZ), minZ_size);
    reader.read(reinterpret_cast<char*>(&header.startOfWaveformDataPacketRecord), startOfWaveformDataPacketRecord_size);
    // version >= 1.4
    if ((header.versionMajor == 1 && header.versionMinor >= 4) || header.versionMajor > 1) {
        reader.read(reinterpret_cast<char*>(&header.startOfFirstExtendedVariableLengthRecord), startOfFirstExtendedVariableLengthRecord_size);
        reader.read(reinterpret_cast<char*>(&header.numberOfExtendedVariableLengthRecords), numberOfExtendedVariableLengthRecords_size);
        reader.read(reinterpret_cast<char*>(&header.numberOfPointRecords), numberOfPointRecords_size);
        reader.read(reinterpret_cast<char*>(header.numberOfPointsByReturn.data()), numberOfPointsByReturn_size);
    } else {
        // copy data from legacy
        header.startOfFirstExtendedVariableLengthRecord = 0;
        header.numberOfExtendedVariableLengthRecords = 0;
        header.numberOfPointRecords = header.legacyNumberOfPointRecords;
        for (size_t i = 0; i < header.legacyNumberOfPointsByReturn.size(); i++) {
            header.numberOfPointsByReturn[i] = header.legacyNumberOfPointsByReturn[i];
        }
    }

    if (reader.fail()) return std::nullopt;

    return header;
}

bool LasPublicHeaderBlock::writePublicHeader(std::ostream &writer, const LasPublicHeaderBlock &header) {
    writer.write(header.fileSignature.data(), fileSignature_size);
    writer.write(reinterpret_cast<const char*>(&header.fileSourceID), fileSourceID_size);
    writer.write(reinterpret_cast<const char*>(&header.globalEncoding), globalEncoding_size);
    writer.write(reinterpret_cast<const char*>(&header.projectID_GUID_Data1), projectID_GUID_Data1_size);
    writer.write(reinterpret_cast<const char*>(&header.projectID_GUID_Data2), projectID_GUID_Data2_size);
    writer.write(reinterpret_cast<const char*>(&header.projectID_GUID_Data3), projectID_GUID_Data3_size);
    writer.write(reinterpret_cast<const char*>(header.projectID_GUID_Data4.data()), projectID_GUID_Data4_size);
    writer.write(reinterpret_cast<const char*>(&header.versionMajor), versionMajor_size);
    writer.write(reinterpret_cast<const char*>(&header.versionMinor), versionMinor_size);
    writer.write(reinterpret_cast<const char*>(header.systemIdentifier.data()), systemIdentifier_size);
    writer.write(reinterpret_cast<const char*>(header.generatingSoftware.data()), generatingSoftware_size);
    writer.write(reinterpret_cast<const char*>(&header.fileCreationDayOfYear), fileCreationDayOfYear_size);
    writer.write(reinterpret_cast<const char*>(&header.fileCreationYear), fileCreationYear_size);
    writer.write(reinterpret_cast<const char*>(&header.headerSize), headerSize_size);
    writer.write(reinterpret_cast<const char*>(&header.offsetToPointData), offsetToPointData_size);
    writer.write(reinterpret_cast<const char*>(&header.numberOfVariableLengthRecords), numberOfVariableLengthRecords_size);
    writer.write(reinterpret_cast<const char*>(&header.pointDataRecordFormat), pointDataRecordFormat_size);
    writer.write(reinterpret_cast<const char*>(&header.pointDataRecordLength), pointDataRecordLength_size);
    writer.write(reinterpret_cast<const char*>(&header.legacyNumberOfPointRecords), legacyNumberOfPointRecords_size);
    writer.write(reinterpret_cast<const char*>(header.legacyNumberOfPointsByReturn.data()), legacyNumberOfPointsByReturn_size);
    writer.write(reinterpret_cast<const char*>(&header.xScaleFactor), xScaleFactor_size);
    writer.write(reinterpret_cast<const char*>(&header.yScaleFactor), yScaleFactor_size);
    writer.write(reinterpret_cast<const char*>(&header.zScaleFactor), zScaleFactor_size);
    writer.write(reinterpret_cast<const char*>(&header.xOffset), xOffset_size);
    writer.write(reinterpret_cast<const char*>(&header.yOffset), yOffset_size);
    writer.write(reinterpret_cast<const char*>(&header.zOffset), zOffset_size);
    writer.write(reinterpret_cast<const char*>(&header.maxX), maxX_size);
    writer.write(reinterpret_cast<const char*>(&header.minX), minX_size);
    writer.write(reinterpret_cast<const char*>(&header.maxY), maxY_size);
    writer.write(reinterpret_cast<const char*>(&header.minY), minY_size);
    writer.write(reinterpret_cast<const char*>(&header.maxZ), maxZ_size);
    writer.write(reinterpret_cast<const char*>(&header.minZ), minZ_size);
    writer.write(reinterpret_cast<const char*>(&header.startOfWaveformDataPacketRecord), startOfWaveformDataPacketRecord_size);
    // version >= 1.4
    if ((header.versionMajor == 1 && header.versionMinor >= 4) || header.versionMajor > 1) {
        writer.write(reinterpret_cast<const char*>(&header.startOfFirstExtendedVariableLengthRecord), startOfFirstExtendedVariableLengthRecord_size);
        writer.write(reinterpret_cast<const char*>(&header.numberOfExtendedVariableLengthRecords), numberOfExtendedVariableLengthRecords_size);
        writer.write(reinterpret_cast<const char*>(&header.numberOfPointRecords), numberOfPointRecords_size);
        writer.write(reinterpret_cast<const char*>(header.numberOfPointsByReturn.data()), numberOfPointsByReturn_size);
    }

    return writer.good();
}

std::optional<PointCloudGenericAttribute> LasPublicHeaderBlock::getPublicHeaderAttributeById(int id) const {
    switch (id) {
    case 0:
        return std::string{fileSignature.begin(), fileSignature.end()};
    case 1:
        return fileSourceID;
    case 2:
        return globalEncoding;
    case 3:
        return projectID_GUID_Data1;
    case 4:
        return projectID_GUID_Data2;
    case 5:
        return projectID_GUID_Data3;
    case 6:
        return std::vector<uint8_t>{projectID_GUID_Data4.begin(), projectID_GUID_Data4.end()};
    case 7:
        return versionMajor;
    case 8:
        return versionMinor;
    case 9:
        return std::string{systemIdentifier.begin(), systemIdentifier.end()};
    case 10:
        return std::string{generatingSoftware.begin(), generatingSoftware.end()};
    case 11:
        return fileCreationDayOfYear;
    case 12:
        return fileCreationYear;
    case 13:
        return headerSize;
    case 14:
        return offsetToPointData;
    case 15:
        return numberOfVariableLengthRecords;
    case 16:
        return pointDataRecordFormat;
    case 17:
        return pointDataRecordLength;
    case 18:
        return legacyNumberOfPointRecords;
    case 19:
        return std::vector<uint32_t>{legacyNumberOfPointsByReturn.begin(), legacyNumberOfPointsByReturn.end()};
    case 20:
        return xScaleFactor;
    case 21:
        return yScaleFactor;
    case 22:
        return zScaleFactor;
    case 23:
        return xOffset;
    case 24:
        return yOffset;
    case 25:
        return zOffset;
    case 26:
        return maxX;
    case 27:
        return minX;
    case 28:
        return maxY;
    case 29:
        return minY;
    case 30:
        return maxZ;
    case 31:
        return minZ;
    case 32:
        return startOfWaveformDataPacketRecord;
    case 33:
        return startOfFirstExtendedVariableLengthRecord;
    case 34:
        return numberOfExtendedVariableLengthRecords;
    case 35:
        return numberOfPointRecords;
    case 36:
        return std::vector<uint64_t>{numberOfPointsByReturn.begin(), numberOfPointsByReturn.end()};
    default:
        return std::nullopt;
    }
}

template <class D>
PtGeometry<PointCloudGenericAttribute> LasPointCloudPoint_Base<D>::getPointPosition() const {
    // the maximum value for the int32 type is used to represent a nan value
    static const auto nan = std::nan("");
    auto xRaw = fromBytes<returnType<D::x_id>>(getRecordDataBuffer() + offset<D::x_id>);
    auto yRaw = fromBytes<returnType<D::y_id>>(getRecordDataBuffer() + offset<D::y_id>);
    auto zRaw = fromBytes<returnType<D::z_id>>(getRecordDataBuffer() + offset<D::z_id>);
    return PtGeometry<PointCloudGenericAttribute>{
        xRaw == std::numeric_limits<returnType<D::x_id>>::max() ? PointCloudGenericAttribute{nan} :
            PointCloudGenericAttribute{xScaleFactor * xRaw + xOffset},
        yRaw == std::numeric_limits<returnType<D::y_id>>::max() ? PointCloudGenericAttribute{nan} :
            PointCloudGenericAttribute{yScaleFactor * yRaw + yOffset},
        zRaw == std::numeric_limits<returnType<D::z_id>>::max() ? PointCloudGenericAttribute{nan} :
            PointCloudGenericAttribute{zScaleFactor * zRaw + zOffset}
    };
}

template <class D>
inline std::optional<PtColor<PointCloudGenericAttribute>> LasPointCloudPoint_Base<D>::getPointColor() const {
    if constexpr (containsColor) {
        return PtColor<PointCloudGenericAttribute>{
            fromBytes<returnType<D::r_id>>(getRecordDataBuffer() + offset<D::r_id>),
            fromBytes<returnType<D::g_id>>(getRecordDataBuffer() + offset<D::g_id>),
            fromBytes<returnType<D::b_id>>(getRecordDataBuffer() + offset<D::b_id>),
            EmptyParam{} // no alpha channel
            };
    } else {
        return std::nullopt;
    }
}

template <class D>
std::optional<PointCloudGenericAttribute> LasPointCloudPoint_Base<D>::getAttributeById(int exposedId) const {
    
    if (exposedId < 0 || exposedId >= exposedIdToInternalId.size()) return std::nullopt;
    size_t id = exposedIdToInternalId[exposedId];
    
    if (id < D::nbAttributes) { // default attribute
        return getAttributeById_helper(id);   
    } else if (id < D::nbAttributes + extraAttributesNames.size()) { // extra attribute
        const size_t extra_id = id - D::nbAttributes;
        // test the type
        switch (extraAttributesTypes[extra_id]) {
            case 1: // uint8_t
                return fromBytes<uint8_t>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 2: // int8_t
                return fromBytes<int8_t>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 3: // uint16_t
                return fromBytes<uint16_t>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 4: // int16_t
                return fromBytes<int16_t>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 5: // uint32_t
                return fromBytes<uint32_t>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 6: // int32_t
                return fromBytes<int32_t>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 7: // uint64_t
                return fromBytes<uint64_t>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 8: // int64_t
                return fromBytes<int64_t>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 9: // float
                return fromBytes<float>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            case 10: // double
                return fromBytes<double>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id]);
                break;
            default:
                // vector of bytes
                return vectorFromBytes<std::byte>(getRecordDataBuffer() + minimumRecordByteSize + extraAttributesOffsets[extra_id], extraAttributesSizes[extra_id]);
        }
    } else {
        return std::nullopt;
    }
}

template <class D>
std::optional<PointCloudGenericAttribute> LasPointCloudPoint_Base<D>::getAttributeByName(const char *attributeName) const {
    // search in default attributes
    auto it = std::find(exposedAttributeNames.begin(), exposedAttributeNames.end(), attributeName);
    if (it != exposedAttributeNames.end()) return getAttributeById(std::distance(exposedAttributeNames.begin(), it));
    // not found
    return std::nullopt; // Attribute not found
}

template <class D>
std::vector<std::string> LasPointCloudPoint_Base<D>::attributeList() const {
    return exposedAttributeNames;
}

template <class Derived>
bool LasPointCloudPoint_Base<Derived>::gotoNext() {
    if (nbPoints == 0 || currentPointIdOneBased >= nbPoints) return false;
    currentPointIdOneBased++;
    reader->read(dataBuffer, recordByteSize);
    return reader->good();
}

template <class D>
LasExtraAttributesInfos LasPointCloudPoint_Base<D>::getExtraAttributesInfos() const {
    LasExtraAttributesInfos infos;
    infos.name = extraAttributesNames;
    infos.type = extraAttributesTypes;
    infos.size = extraAttributesSizes;
    infos.offset = extraAttributesOffsets;
    return infos;
}

template <class D>
void LasPointCloudPoint_Base<D>::verifyAndCorrectParameters(bool hideColorAndGeometricAttributes) {
    // test the size of the extra attributes
    bool isSizeValid =
        extraAttributesNames.size() == extraAttributesTypes.size() &&
        extraAttributesNames.size() == extraAttributesSizes.size() &&
        extraAttributesNames.size() == extraAttributesOffsets.size();
    
    if (!isSizeValid) { // if the size of the extra attributes is not valid, clear the vectors
        extraAttributesNames.clear();
        extraAttributesTypes.clear();
        extraAttributesSizes.clear();
        extraAttributesOffsets.clear();
    } else {
        // test the size and offset of the extra attributes
        std::vector<size_t> toRemove = {};
        for (size_t i = 0; i < extraAttributesNames.size(); i++) {
            
            bool isToRemove = extraAttributesSizes[i] == 0 || // size is 0 or too large or size is 0
                minimumRecordByteSize + extraAttributesOffsets[i] + extraAttributesSizes[i] > recordByteSize;
            
            // check if the name is not already present or empty
            for (const auto& extraName : extraAttributesNames) {
                auto it = std::find(D::attributeNames.begin(), D::attributeNames.begin() + D::nbAttributes, extraName);
                if (it != D::attributeNames.begin() + D::nbAttributes || extraName.empty()) {
                    isToRemove = true;
                    break;
                }
            }

            if (isToRemove) {
                toRemove.push_back(i);
            }     
        }

        // remove the elements
        std::reverse(toRemove.begin(), toRemove.end());
        for (const auto& i : toRemove) {
            extraAttributesNames.erase(extraAttributesNames.begin() + i);
            extraAttributesTypes.erase(extraAttributesTypes.begin() + i);
            extraAttributesSizes.erase(extraAttributesSizes.begin() + i);
            extraAttributesOffsets.erase(extraAttributesOffsets.begin() + i);
        }
    }

    // compute the full list of attributes
    attributeNamesFullListInternal = std::vector<std::string>(
        D::attributeNames.begin(), D::attributeNames.begin() + D::nbAttributes);
    std::copy(extraAttributesNames.begin(), extraAttributesNames.end(),
        std::back_inserter(attributeNamesFullListInternal));

    std::vector<std::string> attributesToHide = {};

    if (hideColorAndGeometricAttributes) {
        attributesToHide = {"x", "y", "z", "red", "green", "blue"};
    }
    for (size_t i = 0; i < attributeNamesFullListInternal.size(); i++) {
        // only add the attribute if it is not in the list of attributes to hide
        if (std::find(attributesToHide.begin(), attributesToHide.end(), attributeNamesFullListInternal[i]) == attributesToHide.end()) {
            exposedIdToInternalId.push_back(i);   
            exposedAttributeNames.push_back(attributeNamesFullListInternal[i]);
        }
    }
}

LasPointCloudPoint::LasPointCloudPoint(size_t recordByteSize, double xScaleFactor, double yScaleFactor,
    double zScaleFactor, double xOffset, double yOffset, double zOffset) :
    LasPointCloudPoint(recordByteSize, xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset, nullptr) {
    dataBufferContainer.resize(recordByteSize);
    dataBuffer = dataBufferContainer.data();
}

LasPointCloudPoint::LasPointCloudPoint(size_t recordByteSize, double xScaleFactor, double yScaleFactor,
    double zScaleFactor, double xOffset, double yOffset, double zOffset, char *dataBuffer) :
    recordByteSize{recordByteSize}, xScaleFactor{xScaleFactor}, yScaleFactor{yScaleFactor}, zScaleFactor{zScaleFactor},
    xOffset{xOffset}, yOffset{yOffset}, zOffset{zOffset}, dataBuffer{dataBuffer} { }

bool LasPointCloudPoint::write(std::ostream &writer) const {
    writer.write(dataBuffer, recordByteSize);
    return writer.good();
}

std::unique_ptr<PointCloudPointAccessInterface> LasPointCloudPoint::createAdapter(
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterface, double xScaleFactor,
    double yScaleFactor, double zScaleFactor, double xOffset, double yOffset, double zOffset) {
    // test if nullptr. If so, return nullptr
    if (pointCloudPointAccessInterface == nullptr) {return nullptr;}

    // try to cast the point cloud to a las point cloud
    auto lasPoint = dynamic_cast<LasPointCloudPoint*>(pointCloudPointAccessInterface.get());
    if (lasPoint != nullptr) {
        // return same object
        return std::move(pointCloudPointAccessInterface);
    }
    // create adapter
    auto pointPtr = pointCloudPointAccessInterface.get();
    return std::make_unique<LasPointCloudPointBasicAdapter>(std::move(pointCloudPointAccessInterface), pointPtr,
        xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset);
}

std::optional<FullPointCloudAccessInterface> openPointCloudLas(const std::filesystem::path &lasFilePath) {
    // open the file
    auto reader = std::make_unique<ifstreamCustomBuffer<lasFileReaderBufferSize>>();

    constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
    reader->precision(maxPrecision); // set the precision for the reader

    reader->open(lasFilePath, std::ios_base::binary);

    if (!reader->is_open()) return std::nullopt;

    return openPointCloudLas(std::move(reader));
}

std::optional<FullPointCloudAccessInterface> openPointCloudLas(std::unique_ptr<std::istream> reader) {
    // read the header
    auto header = LasPointCloudHeader::readHeader(*reader);
    // test if header ptr is not null
    if (header == nullptr) {
        return std::nullopt;
    }
    // get the extra attributes
    auto [extraAttributesNames, extraAttributesTypes, extraAttributesSizes, extraAttributesOffsets]
        = header->getPointwiseExtraAttributesInfos();
    // create a point cloud
    auto pointCloud = createLasPointCloudPoint(std::move(reader), header->publicHeaderBlock.getPointDataRecordLength(),
        std::max((size_t) header->publicHeaderBlock.legacyNumberOfPointRecords,
            header->publicHeaderBlock.numberOfPointRecords),
        header->publicHeaderBlock.getPointDataRecordFormat(), extraAttributesNames, extraAttributesTypes,
        extraAttributesSizes, extraAttributesOffsets, true, header->publicHeaderBlock.getXScaleFactor(),
        header->publicHeaderBlock.getYScaleFactor(), header->publicHeaderBlock.getZScaleFactor(),
        header->publicHeaderBlock.getXOffset(), header->publicHeaderBlock.getYOffset(),
        header->publicHeaderBlock.getZOffset(), nullptr);

    if (pointCloud == nullptr) {
        return std::nullopt;
    }

    // return the point cloud full access interface
    FullPointCloudAccessInterface fullPointInterface;
    fullPointInterface.headerAccess = std::move(header);
    fullPointInterface.pointAccess = std::move(pointCloud);
    return fullPointInterface;

    return std::nullopt;
}

bool writePointCloudLas(std::ostream &writer, FullPointCloudAccessInterface &pointCloud) {
    // test if point cloud ptr is not null
    if (pointCloud.pointAccess == nullptr) {
        return false;
    }

    double xScaleFactor = 0.01;
    double yScaleFactor = 0.01;
    double zScaleFactor = 0.01;
    double xOffset = 0;
    double yOffset = 0;
    double zOffset = 0;

    if (pointCloud.headerAccess != nullptr) {
        // try to get the xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset
        auto xScaleFactorOpt = pointCloud.headerAccess->getAttributeByName("xScaleFactor");
        auto yScaleFactorOpt = pointCloud.headerAccess->getAttributeByName("yScaleFactor");
        auto zScaleFactorOpt = pointCloud.headerAccess->getAttributeByName("zScaleFactor");
        auto xOffsetOpt = pointCloud.headerAccess->getAttributeByName("xOffset");
        auto yOffsetOpt = pointCloud.headerAccess->getAttributeByName("yOffset");
        auto zOffsetOpt = pointCloud.headerAccess->getAttributeByName("zOffset");

        if (xScaleFactorOpt.has_value())
            xScaleFactor = castedPointCloudAttribute<double>(xScaleFactorOpt.value());
        if (yScaleFactorOpt.has_value())
            yScaleFactor = castedPointCloudAttribute<double>(yScaleFactorOpt.value());
        if (zScaleFactorOpt.has_value())
            zScaleFactor = castedPointCloudAttribute<double>(zScaleFactorOpt.value());
        if (xOffsetOpt.has_value())
            xOffset = castedPointCloudAttribute<double>(xOffsetOpt.value());
        if (yOffsetOpt.has_value())
            yOffset = castedPointCloudAttribute<double>(yOffsetOpt.value());
        if (zOffsetOpt.has_value())
            zOffset = castedPointCloudAttribute<double>(zOffsetOpt.value());
    }
    
    pointCloud.pointAccess = LasPointCloudPoint::createAdapter(std::move(pointCloud.pointAccess), xScaleFactor,
        yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset);
    // safe cast
    auto lasPointAccessAdapter = static_cast<LasPointCloudPoint*>(pointCloud.pointAccess.get());
    if (lasPointAccessAdapter == nullptr) return false;

    // try to cast the header to a las header
    auto lasHeader = dynamic_cast<LasPointCloudHeader*>(pointCloud.headerAccess.get());
    auto newHeader = std::make_unique<LasPointCloudHeader>();
    if (lasHeader != nullptr) {
        newHeader->publicHeaderBlock = lasHeader->publicHeaderBlock;
        newHeader->variableLengthRecords = lasHeader->variableLengthRecords;
        newHeader->extendedVariableLengthRecords = lasHeader->extendedVariableLengthRecords;
    }

    lasHeader = newHeader.get();

    // add the extra byte data
    auto extraAttributesInfos = lasPointAccessAdapter->getExtraAttributesInfos();
    if (extraAttributesInfos.name.size() > 0) {
        auto extraBytesVlrData = LasPointCloudHeader::generateExtraBytesVlrData(extraAttributesInfos);
        LasVariableLengthRecord extraBytesVlr{"LASF_Spec", 4, extraBytesVlrData};
        // remove the VLR or EVLR if already present
        for (size_t i_supp = lasHeader->variableLengthRecords.size(); i_supp > 0; i_supp--) {
            size_t i = i_supp - 1;
            if (lasHeader->variableLengthRecords[i].getUserId() == "LASF_Spec" && lasHeader->variableLengthRecords[i].getRecordId() == 4) {
                lasHeader->variableLengthRecords.erase(lasHeader->variableLengthRecords.begin() + i);
                lasHeader->publicHeaderBlock.numberOfVariableLengthRecords--;
            }
        }
        for (size_t i_supp = lasHeader->extendedVariableLengthRecords.size(); i_supp > 0; i_supp--) {
            size_t i = i_supp - 1;
            if (lasHeader->extendedVariableLengthRecords[i].getUserId() == "LASF_Spec" && lasHeader->extendedVariableLengthRecords[i].getRecordId() == 4) {
                lasHeader->extendedVariableLengthRecords.erase(lasHeader->extendedVariableLengthRecords.begin() + i);
                lasHeader->publicHeaderBlock.numberOfExtendedVariableLengthRecords--;
            }
        }
        // add it to the VLRs
        lasHeader->variableLengthRecords.push_back(extraBytesVlr);
        lasHeader->publicHeaderBlock.numberOfVariableLengthRecords++;
    }
    // properly set the xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset for the header
    lasHeader->publicHeaderBlock.xScaleFactor = xScaleFactor;
    lasHeader->publicHeaderBlock.yScaleFactor = yScaleFactor;
    lasHeader->publicHeaderBlock.zScaleFactor = zScaleFactor;
    lasHeader->publicHeaderBlock.xOffset = xOffset;
    lasHeader->publicHeaderBlock.yOffset = yOffset;
    lasHeader->publicHeaderBlock.zOffset = zOffset;

    // set version to 1.4 + header size
    lasHeader->publicHeaderBlock.versionMajor = 1;
    lasHeader->publicHeaderBlock.versionMinor = 4;
    lasHeader->publicHeaderBlock.headerSize = 375;

    // set the point data record format and record length
    lasHeader->publicHeaderBlock.pointDataRecordFormat = lasPointAccessAdapter->getFormat();
    lasHeader->publicHeaderBlock.pointDataRecordLength = lasPointAccessAdapter->getRecordByteSize();
    
    // write the public header + the VLRs
    auto beginPos = writer.tellp();
    lasHeader->writePublicHeader(writer);
    
    auto vlrPos = writer.tellp();
    lasHeader->writeVLRs(writer);

    auto pointPos = writer.tellp();
    size_t nbPoints = 0;
    // reset to zero
    lasHeader->publicHeaderBlock.numberOfPointsByReturn = {};
    // write the point cloud
    do {
        lasPointAccessAdapter->write(writer);
        if (writer.fail()) return false;
        // get the return number. carreful: the return number starts at 1
        auto returnNumber
            = castedPointCloudAttribute<size_t>(lasPointAccessAdapter->getAttributeByName("returnNumber").value_or(1));
        // force the return number to be between 1 and the number of returns
        returnNumber = std::max(size_t{1}, std::min(returnNumber, lasHeader->publicHeaderBlock.numberOfPointsByReturn.size()));
        lasHeader->publicHeaderBlock.numberOfPointsByReturn[returnNumber-1]++;
        nbPoints++;
    } while (lasPointAccessAdapter->gotoNext());


    auto evlrPos = writer.tellp();
    
    // update the public header and rewrite it
    lasHeader->publicHeaderBlock.offsetToPointData = pointPos - beginPos;
    if (lasHeader->publicHeaderBlock.numberOfExtendedVariableLengthRecords > 0) {
        lasHeader->publicHeaderBlock.startOfFirstExtendedVariableLengthRecord = evlrPos - beginPos;   
    }
    // set the number of points
    lasHeader->publicHeaderBlock.numberOfPointRecords = nbPoints;
    constexpr auto maxValueLegacy =
        std::numeric_limits<decltype(lasHeader->publicHeaderBlock.legacyNumberOfPointRecords)>::max();
    if (nbPoints <= maxValueLegacy) {
        // set the legacy number of points
        lasHeader->publicHeaderBlock.legacyNumberOfPointRecords = nbPoints;
    }
    // try to set the legacy number of points by return from the number of points by return
    lasHeader->publicHeaderBlock.legacyNumberOfPointsByReturn = {};
    auto maxLegacyReturnNumber = lasHeader->publicHeaderBlock.legacyNumberOfPointsByReturn.size();
    auto maxReturnNumber = lasHeader->publicHeaderBlock.numberOfPointsByReturn.size();
    for (size_t i = 0; i < maxReturnNumber; i++) {
        auto legacyReturnNumber = std::min(maxLegacyReturnNumber, i);
        // if the return number is bigger than the legacy one, we add it to the value of the max legacy return number
        // test if the number of points for this return number is bigger than the maximum value allowed by the type
        size_t nbPointsByReturn = lasHeader->publicHeaderBlock.numberOfPointsByReturn[i]
            + lasHeader->publicHeaderBlock.legacyNumberOfPointsByReturn[legacyReturnNumber];
        if (nbPointsByReturn > maxValueLegacy) {
            // cannot have a valid value
            lasHeader->publicHeaderBlock.legacyNumberOfPointsByReturn = {};
            break;
        } else {
            lasHeader->publicHeaderBlock.legacyNumberOfPointsByReturn[legacyReturnNumber] = nbPointsByReturn;
        }
    }

    // rewrite the public header:
    writer.seekp(beginPos);
    lasHeader->writePublicHeader(writer);
    writer.seekp(evlrPos);

    // write the evlrs
    return lasHeader->writeEVLRs(writer);
}

bool writePointCloudLas(const std::filesystem::path &lasFilePath, FullPointCloudAccessInterface &pointCloud) {
   // open the file
    auto writer = std::make_unique<fstreamCustomBuffer<lasFileWriterBufferSize>>();

    writer->open(lasFilePath, std::ios_base::in | std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

    if (!writer->is_open()) return false;

    // set the precision to the maximum
    constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
    *writer << std::setprecision(maxPrecision);

    auto success = writePointCloudLas(*writer, pointCloud);
    writer->close();

    return success;
}

template <size_t N>
std::unique_ptr<LasPointCloudPoint> createLasPointCloudPoint_helper(std::unique_ptr<std::istream> reader,
    size_t recordByteSize, size_t nbPoints, size_t recordFormatNumber, const std::vector<std::string> &extraAttributesNames,
    const std::vector<uint8_t> &extraAttributesTypes, const std::vector<size_t> &extraAttributesSizes,
    const std::vector<size_t> &extraAttributesOffsets, bool hideColorAndGeometricAttributes, double xScaleFactor,
    double yScaleFactor, double zScaleFactor, double xOffset, double yOffset, double zOffset, char *dataBuffer) {

    if constexpr (N > 10) {
        return nullptr;
    } else if (N == recordFormatNumber) {
        if (dataBuffer == nullptr) {
            return std::make_unique<LasPointCloudPoint_Format<N>>(std::move(reader), recordByteSize, nbPoints,
                extraAttributesNames, extraAttributesTypes, extraAttributesSizes, extraAttributesOffsets,
                hideColorAndGeometricAttributes, xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset);
        } else  {
            return std::make_unique<LasPointCloudPoint_Format<N>>(std::move(reader), recordByteSize, nbPoints,
                extraAttributesNames, extraAttributesTypes, extraAttributesSizes, extraAttributesOffsets,
                hideColorAndGeometricAttributes, xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset,
                dataBuffer);
        }
    } else {
        return createLasPointCloudPoint_helper<N+1>(std::move(reader), recordByteSize, nbPoints,
            recordFormatNumber, extraAttributesNames, extraAttributesTypes, extraAttributesSizes,
            extraAttributesOffsets, hideColorAndGeometricAttributes, xScaleFactor, yScaleFactor, zScaleFactor, xOffset,
            yOffset, zOffset,dataBuffer);
    }
}

std::unique_ptr<LasPointCloudPoint> createLasPointCloudPoint(std::unique_ptr<std::istream> reader, size_t recordByteSize,
    size_t nbPoints, size_t recordFormatNumber, const std::vector<std::string> &extraAttributesNames,
    const std::vector<uint8_t> &extraAttributesTypes, const std::vector<size_t> &extraAttributesSizes,
    const std::vector<size_t> &extraAttributesOffsets, bool hideColorAndGeometricAttributes, double xScaleFactor,
    double yScaleFactor, double zScaleFactor, double xOffset, double yOffset, double zOffset, char *dataBuffer) {

    auto lasPointCloudPoint = createLasPointCloudPoint_helper(std::move(reader), recordByteSize, nbPoints,
        recordFormatNumber, extraAttributesNames, extraAttributesTypes, extraAttributesSizes, extraAttributesOffsets,
        hideColorAndGeometricAttributes, xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset,
        dataBuffer);

    // try to go to first point
    if (lasPointCloudPoint != nullptr) {
        if (!lasPointCloudPoint->gotoNext()) {
            return nullptr;
        }
    } else {
        return nullptr;
    }
    return std::move(lasPointCloudPoint);

}

LasExtraBytesDescriptor::LasExtraBytesDescriptor(uint8_t data_type, const std::string & name, size_t size,
    const std::optional<std::string>& description,
    const std::optional<std::variant<uint64_t, int64_t, double>>& no_data,
    const std::optional<std::variant<uint64_t, int64_t, double>>& min,
    const std::optional<std::variant<uint64_t, int64_t, double>>& max,
    std::optional<double> scale, std::optional<double> offset):
        data_type{data_type} {
    
    constexpr std::byte no_data_bit {0b00000001};
    constexpr std::byte min_bit     {0b00000010};
    constexpr std::byte max_bit     {0b00000100};
    constexpr std::byte scale_bit   {0b00001000};
    constexpr std::byte offset_bit  {0b00010000};

    std::byte options{0b0000000};

    size_t name_nbChars = std::min(name.length(), this->name.size());
    for (size_t i = 0; i < name_nbChars; i++) {
        this->name[i] = name[i];
    }

    if (description.has_value()) {
        size_t description_nbChars = std::min(description->length(), this->description.size());
        for (size_t i = 0; i < description_nbChars; i++) {
            this->description[i] = description->at(i);
        }
    }

    if (no_data.has_value()) {
        options |= no_data_bit;
        if (std::holds_alternative<uint64_t>(*no_data)) {
            auto value = std::get<uint64_t>(*no_data);
            this->no_data = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        } else if (std::holds_alternative<int64_t>(*no_data)) {
            auto value = std::get<int64_t>(*no_data);
            this->no_data = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        } else if (std::holds_alternative<double>(*no_data)) {
            auto value = std::get<double>(*no_data);
            this->no_data = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        }
    }

    if (min.has_value()) {
        options |= min_bit;
        if (std::holds_alternative<uint64_t>(*min)) {
            auto value = std::get<uint64_t>(*min);
            this->min = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        } else if (std::holds_alternative<int64_t>(*min)) {
            auto value = std::get<int64_t>(*min);
            this->min = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        } else if (std::holds_alternative<double>(*min)) {
            auto value = std::get<double>(*min);
            this->min = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        }
    }

    if (max.has_value()) {
        options |= max_bit;
        if (std::holds_alternative<uint64_t>(*max)) {
            auto value = std::get<uint64_t>(*max);
            this->max = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        } else if (std::holds_alternative<int64_t>(*max)) {
            auto value = std::get<int64_t>(*max);
            this->max = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        } else if (std::holds_alternative<double>(*max)) {
            auto value = std::get<double>(*max);
            this->max = arrayFromBytes<std::byte, 8>(reinterpret_cast<const std::byte*>(&value));
        }
    }

    if (scale.has_value()) {
        options |= scale_bit;
        this->scale = *scale;
    }

    if (offset.has_value()) {
        options |= offset_bit;
        this->offset = *offset;
    }

    this->options = bit_cast<uint8_t>(options);

    // if data type is 0, we have a sequence of bytes and the size is contained in the options
    if (data_type == 0) {
        this->options = static_cast<uint8_t>(size);
    }
}

LasExtraBytesDescriptor::LasExtraBytesDescriptor(const std::array<uint8_t, 2> &reserved, uint8_t data_type,
    uint8_t options, const std::array<char, 32> &name, const std::array<uint8_t, 4> &unused,
    const std::array<std::byte, 8> &no_data, const std::array<uint8_t, 16> &deprecated1,
    const std::array<std::byte, 8> &min, const std::array<uint8_t, 16> &deprecated2,
    const std::array<std::byte, 8> &max, const std::array<uint8_t, 16> &deprecated3,
    double scale, const std::array<uint8_t, 16> &deprecated4, double offset,
    const std::array<uint8_t, 16> &deprecated5, const std::array<char, 32> &description):
        reserved{reserved}, data_type{data_type}, options{options}, name{name}, unused{unused},
        no_data{no_data}, deprecated1{deprecated1}, min{min}, deprecated2{deprecated2}, max{max},
        deprecated3{deprecated3}, scale{scale}, deprecated4{deprecated4}, offset{offset}, deprecated5{deprecated5},
        description{description}
{ }

LasExtraBytesDescriptor::LasExtraBytesDescriptor(const char *buffer) :
    LasExtraBytesDescriptor(arrayFromBytes<uint8_t, 2>(buffer), fromBytes<uint8_t>(buffer + 2),
        fromBytes<uint8_t>(buffer + 3), arrayFromBytes<char, 32>(buffer + 4),
        arrayFromBytes<uint8_t, 4>(buffer + 36), arrayFromBytes<std::byte, 8>(buffer + 40),
        arrayFromBytes<uint8_t, 16>(buffer + 48), arrayFromBytes<std::byte, 8>(buffer + 64),
        arrayFromBytes<uint8_t, 16>(buffer + 72), arrayFromBytes<std::byte, 8>(buffer + 88),
        arrayFromBytes<uint8_t, 16>(buffer + 96), fromBytes<double>(buffer + 112),
        arrayFromBytes<uint8_t, 16>(buffer + 120), fromBytes<double>(buffer + 136),
        arrayFromBytes<uint8_t, 16>(buffer + 144), arrayFromBytes<char, 32>(buffer + 160))
{ }

void LasExtraBytesDescriptor::toBytes(char *buffer) const {
    // bytestream
    std::ostringstream writer;
    writer.rdbuf()->pubsetbuf(buffer, 192);
    // write the data
    writer.write(reinterpret_cast<const char*>(reserved.data()), 2);
    writer.write(reinterpret_cast<const char*>(&data_type), 1);
    writer.write(reinterpret_cast<const char*>(&options), 1);
    writer.write(reinterpret_cast<const char*>(name.data()), 32);
    writer.write(reinterpret_cast<const char*>(unused.data()), 4);
    writer.write(reinterpret_cast<const char*>(no_data.data()), 8);
    writer.write(reinterpret_cast<const char*>(deprecated1.data()), 16);
    writer.write(reinterpret_cast<const char*>(min.data()), 8);
    writer.write(reinterpret_cast<const char*>(deprecated2.data()), 16);
    writer.write(reinterpret_cast<const char*>(max.data()), 8);
    writer.write(reinterpret_cast<const char*>(deprecated3.data()), 16);
    writer.write(reinterpret_cast<const char*>(&scale), 8);
    writer.write(reinterpret_cast<const char*>(deprecated4.data()), 16);
    writer.write(reinterpret_cast<const char*>(&offset), 8);
    writer.write(reinterpret_cast<const char*>(deprecated5.data()), 16);
    writer.write(reinterpret_cast<const char*>(description.data()), 32);
}

std::vector<std::byte> LasExtraBytesDescriptor::toBytes() const {
    std::vector<std::byte> bytes;
    bytes.resize(192);
    this->toBytes(reinterpret_cast<char*>(bytes.data()));
    return bytes;
}

LasPointCloudPointBasicAdapter::LasPointCloudPointBasicAdapter(
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr,
    PointCloudPointAccessInterface* pointCloudPointAccessInterface, double xScaleFactor, double yScaleFactor,
    double zScaleFactor, double xOffset, double yOffset, double zOffset) :
        LasPointCloudPointBasicAdapter{std::move(pointCloudPointAccessInterfaceUniquePtr),
            pointCloudPointAccessInterface, xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset,
            getLasDataLayoutFromPointcloudPoint(pointCloudPointAccessInterface)}
{ }

PtGeometry<PointCloudGenericAttribute> LasPointCloudPointBasicAdapter::getPointPosition() const {
    return pointPosition;
}

std::optional<PtColor<PointCloudGenericAttribute>> LasPointCloudPointBasicAdapter::getPointColor() const {
    return pointColor;
}

std::optional<PointCloudGenericAttribute> LasPointCloudPointBasicAdapter::getAttributeById(int id) const {
    if (id < 0 || id >= attributeNames.size()) return std::nullopt;

    PointCloudGenericAttribute attValue;
    
    // special case for points and color
    if (id == xIndex) {
        attValue = pointPositionScaledInteger.x;
    } else if (id == yIndex) {
        attValue = pointPositionScaledInteger.y;
    } else if (id == zIndex) {
        attValue = pointPositionScaledInteger.z;
    } else if (id == redIndex && containsColor) {
        if (pointColor.has_value()) {
            attValue = (*pointColor).r;
        }
    } else if (id == greenIndex && containsColor) {
        if (pointColor.has_value()) {
            attValue = (*pointColor).g;
        }
    } else if (id == blueIndex && containsColor) {
        if (pointColor.has_value()) {
            attValue = (*pointColor).b;
        }
    } else { // default case
        attValue = pointCloudPointAccessInterface->getAttributeByName(attributeNames[id].c_str()).value_or(0);
    }

    return castAttributeToLasType(attValue, fieldType[id], fieldByteSize[id]);
}

std::optional<PointCloudGenericAttribute> LasPointCloudPointBasicAdapter::getAttributeByName(
        const char *attributeName) const {
    // search for the attribute name to only accept known attributes
    auto it = std::find(attributeNames.begin(), attributeNames.end(), attributeName);
    return getAttributeById(it - attributeNames.begin());
}

std::vector<std::string> LasPointCloudPointBasicAdapter::attributeList() const {
    return attributeNames;
}

LasExtraAttributesInfos LasPointCloudPointBasicAdapter::getExtraAttributesInfos() const {
    LasExtraAttributesInfos infos;
   
    auto minNbAttr = getMinimumNumberOfAttributes();
    infos.name = std::vector<std::string>(attributeNames.begin() + minNbAttr, attributeNames.end());
    infos.type = std::vector<uint8_t>(fieldType.begin() + minNbAttr, fieldType.end());
    infos.size = std::vector<size_t>(fieldByteSize.begin() + minNbAttr, fieldByteSize.end()); 
    infos.offset = std::vector<size_t>(fieldOffset.size() - minNbAttr);

    for (size_t i = minNbAttr; i < fieldOffset.size(); i++) {
        infos.offset[i-minNbAttr] = fieldOffset[i] - fieldOffset[minNbAttr];
    }

    return infos;
}

bool LasPointCloudPointBasicAdapter::gotoNext()
{
    return pointCloudPointAccessInterface->gotoNext() ? adaptInternalState() : false;
}

size_t LasPointCloudPointBasicAdapter::getSuitableFormat(const PointCloudPointAccessInterface &pointCloudPointAccessInterface,
    const std::optional<size_t> &defaultFormat) {

    auto isDefaultFormat = defaultFormat.has_value();
    size_t defaultFormatValue = defaultFormat.value_or(6);

    defaultFormatValue = defaultFormatValue <= 10 ? defaultFormatValue : 6;

    bool isLegacyFormat = pointCloudPointAccessInterface.getAttributeByName("scanAngleRank").has_value();
    bool containsColor = pointCloudPointAccessInterface.getPointColor().has_value();
    bool containsGPS = pointCloudPointAccessInterface.getAttributeByName("gpsTime").has_value();
    bool containsWavePacket = pointCloudPointAccessInterface.getAttributeByName("wavePacketDescriptorIndex").has_value();
    bool containsNIR = pointCloudPointAccessInterface.getAttributeByName("NIR").has_value();

    size_t format;
    if (isLegacyFormat) {
        // 1: gps
        // 2: rgb
        // 3: rgb + gps
        // 4: gps + wp
        // 5: rgb + gps + wp
        if (containsColor) { // 2,3,5
            if (containsGPS) { // 3,5
                if (containsWavePacket) { // 5
                    format = 5;
                } else { // 3
                    format = 3;
                }
            } else { // 2
                format = 2;
            }
        } else { // 0,1,4
            if (containsGPS) { // 1,4
                if (containsWavePacket) { // 4
                    format = 4;
                } else { // 1
                    format = 1;
                }
            } else { // 0
                format = 0;
            }
        }
    } else {
        format = 10;
        // 7: rgb
        // 8: rgb + nir
        // 9: wp
        // 10: rgb + nir + wp
        if (containsColor) { // 7,8,10
            if (containsNIR) { // 8,10
                if (containsWavePacket) { // 10
                    format = 10;
                } else { // 8
                    format = 8;
                }
            } else { // 7
                format = 7;
            }
        } else { // 6,9
            if (containsWavePacket) { // 9
                format = 9;
            } else { // 6
                format = 6;
            }
        }
    }

    return format;
}

LasDataLayout LasPointCloudPointBasicAdapter::getLasDataLayoutFromPointcloudPoint(
    PointCloudPointAccessInterface* pointcloudPointAccessInterface) {
    
    if (pointcloudPointAccessInterface == nullptr) {
        return {};
    }
    auto format = getSuitableFormat(*pointcloudPointAccessInterface);
    return getLasDataLayoutFromPointcloudPoint_helper(*pointcloudPointAccessInterface, format);
}


LasPointCloudPointBasicAdapter::LasPointCloudPointBasicAdapter(
    std::unique_ptr<PointCloudPointAccessInterface> pointCloudPointAccessInterfaceUniquePtr,
    PointCloudPointAccessInterface* pointCloudPointAccessInterface, double xScaleFactor, double yScaleFactor,
    double zScaleFactor, double xOffset, double yOffset, double zOffset, const LasDataLayout &attributeInformations) :
        pointCloudPointAccessInterfaceUniquePtr{std::move(pointCloudPointAccessInterfaceUniquePtr)},
        pointCloudPointAccessInterface{pointCloudPointAccessInterface}, format{attributeInformations.format},
        minimumNumberOfAttributes{attributeInformations.minimumNumberOfAttributes},
        attributeNames{attributeInformations.attributeNames},
        fieldType{attributeInformations.fieldType}, fieldByteSize{attributeInformations.fieldByteSize},
        fieldOffset{attributeInformations.fieldOffset}, usePriorDataOffset{attributeInformations.usePriorDataOffset},
        isBitfield{attributeInformations.isBitfield}, bitfieldSize{attributeInformations.bitfieldSize},
        bitfieldOffset{attributeInformations.bitfieldOffset},
        LasPointCloudPoint{attributeInformations.recordByteSize,
            xScaleFactor, yScaleFactor, zScaleFactor, xOffset, yOffset, zOffset} {

    // default value is out of range
    xIndex = yIndex = zIndex = redIndex = greenIndex = blueIndex = attributeNames.size();
    containsColor = formatContainsColor(format);
    // find the indices for the corresponding elements
    for (size_t i = 0; i < attributeNames.size(); ++i) {
        if (attributeNames[i] == "x") {
            xIndex = i;
        } else if (attributeNames[i] == "y") {
            yIndex = i;
        } else if (attributeNames[i] == "z") {
            zIndex = i;
        } else if (attributeNames[i] == "red") {
            redIndex = i;
        } else if (attributeNames[i] == "green") {
            greenIndex = i;
        } else if (attributeNames[i] == "blue") {
            blueIndex = i;
        }
    }

    adaptInternalState();
}

PointCloudGenericAttribute LasPointCloudPointBasicAdapter::castAttributeToLasType(
    const PointCloudGenericAttribute &attribute, uint8_t lasType, std::optional<size_t> numberUndocumentedBytes) {
    switch (lasType) {
        case 1: // uint8_t
            return castedPointCloudAttribute<uint8_t>(attribute);
            break;
        case 2: // int8_t
            return castedPointCloudAttribute<int8_t>(attribute);
            break;
        case 3: // uint16_t
            return castedPointCloudAttribute<uint16_t>(attribute);
            break;
        case 4: // int16_t
            return castedPointCloudAttribute<int16_t>(attribute);
            break;
        case 5: // uint32_t
            return castedPointCloudAttribute<uint32_t>(attribute);
            break;
        case 6: // int32_t
            return castedPointCloudAttribute<int32_t>(attribute);
            break;
        case 7: // uint64_t
            return castedPointCloudAttribute<uint64_t>(attribute);
            break;
        case 8: // int64_t
            return castedPointCloudAttribute<int64_t>(attribute);
            break;
        case 9: // float
            return castedPointCloudAttribute<float>(attribute);
            break;
        case 10: // double
            return castedPointCloudAttribute<double>(attribute);
            break;
        default:
            // vector of bytes
            auto v = castedPointCloudAttribute<std::vector<std::byte>>(attribute);
            if (numberUndocumentedBytes.has_value()) {
                v.resize(numberUndocumentedBytes.value());
            }
            return v;
    }
}

bool LasPointCloudPointBasicAdapter::adaptInternalState() {
    static_assert(sizeof(float) == 4 && sizeof(double) == 8);

    if (pointCloudPointAccessInterface == nullptr) return false;

    // get position and color
    pointPosition = pointCloudPointAccessInterface->getPointPosition();
    auto pointPositionDouble = pointCloudPointAccessInterface->castedPointGeometry<double>();

    pointColor = pointCloudPointAccessInterface->getPointColor();

    pointPositionScaledInteger = {
        static_cast<int32_t>(std::isnan(pointPositionDouble.x) ? std::numeric_limits<int32_t>::max() :
            std::round(((pointPositionDouble.x) - xOffset) / xScaleFactor)),
        static_cast<int32_t>(std::isnan(pointPositionDouble.y) ? std::numeric_limits<int32_t>::max() :
            std::round(((pointPositionDouble.y) - yOffset) / yScaleFactor)),
        static_cast<int32_t>(std::isnan(pointPositionDouble.z) ? std::numeric_limits<int32_t>::max() :
            std::round(((pointPositionDouble.z) - zOffset) / zScaleFactor))
    };
    
    for (size_t fieldIt = 0; fieldIt < fieldByteSize.size(); fieldIt++) {
        auto size = fieldByteSize[fieldIt];
        auto type = fieldType[fieldIt];

        // try to get the attribute
        auto attrOpt = getAttributeById(fieldIt);
        if (attrOpt.has_value()) {
            const auto& attr = attrOpt.value();

            auto* position = getRecordDataBuffer() + fieldOffset[fieldIt];

            // if we have a bitfield, save the byte
            std::byte bitField = bit_cast<std::byte>(*position);
            
            switch (type) {
                case 1: { // uint8_t
                    auto attrData = std::get<uint8_t>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 2: { // int8_t
                    auto attrData = std::get<int8_t>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 3: { // uint16_t
                    auto attrData = std::get<uint16_t>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 4: { // int16_t
                    auto attrData = std::get<int16_t>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 5: { // uint32_t
                    auto attrData = std::get<uint32_t>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 6: { // int32_t
                    auto attrData = std::get<int32_t>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 7: { // uint64_t
                    auto attrData = std::get<uint64_t>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 8: { // int64_t
                    auto attrData = std::get<int64_t>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 9: { // float
                    auto attrData = std::get<float>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                case 10: { // double
                    auto attrData = std::get<double>(attr);
                    std::memcpy(position, &attrData, size);
                    break;
                }
                default:
                    // vector of bytes
                    auto attrData = std::get<std::vector<std::byte>>(attr);
                    std::memcpy(position, attrData.data(), size);
            }

            //case with bitfield
            if (isBitfield[fieldIt]) {
                // add the bit flag
                bool flag = bit_cast<std::byte>(*position) != std::byte{0x00};
                if (flag) {
                    bitField |= std::byte{0x01} << bitfieldOffset[fieldIt];
                }
            }
        }
    }
    return true;
}

template <size_t Format>
LasDataLayout LasPointCloudPointBasicAdapter::getLasDataLayoutFromPointcloudPoint_helper(
        const PointCloudPointAccessInterface &pointcloudPointAccessInterface, size_t format) {
    
    if constexpr (Format > 10) { // default case
        return getLasDataLayoutFromPointcloudPoint_helper<10>(pointcloudPointAccessInterface, 10);
    } else if (format == Format) {

        LasDataLayout dataLayout;
        
        // visit the variant
        auto visitorCustomAttributes = [](auto &&attr) {
            using T = std::decay_t<decltype(attr)>;
            uint8_t type;
            size_t size;
            if constexpr (std::is_same_v<T, uint8_t>) {
                type = 1; // uint8_t
                size = 1;
            } else if constexpr (std::is_same_v<T, int8_t>) {
                type = 2; // int8_t
                size = 1;
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                type = 3; // uint16_t
                size = 2;
            } else if constexpr (std::is_same_v<T, int16_t>) {
                type = 4; // int16_t
                size = 2;
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                type = 5; // uint32_t
                size = 4;
            } else if constexpr (std::is_same_v<T, int32_t>) {
                type = 6; // int32_t
                size = 4;
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                type = 7; // uint64_t
                size = 8;
            } else if constexpr (std::is_same_v<T, int64_t>) {
                type = 8; // int64_t
                size = 8;
            } else if constexpr (std::is_same_v<T, float>) {
                type = 9; // float
                size = 4;
            } else if constexpr (std::is_same_v<T, double>) {
                type = 10; // double
                size = 8;
            } else if constexpr (std::is_same_v<T, std::vector<std::byte>>) {
                type = 0; // Default case, unknown type which is a sequence of bytes
                size = attr.size();
            } else {
                type = 0; // Default case, unknown type which is a sequence of bytes
                size = 0;
            }
            return std::tuple<uint8_t, size_t>{type, size};
        };

        // fill the data layout with the minimum fields of the format

        dataLayout.format = Format;
        dataLayout.recordByteSize = LasPointCloudPoint_Format<Format>::minimumRecordByteSize;
        
        size_t nbAttributes = LasPointCloudPoint_Format<Format>::nbAttributes;
        dataLayout.minimumNumberOfAttributes = nbAttributes;

        for (size_t i = 0; i < nbAttributes; ++i) {
            dataLayout.attributeNames.push_back(std::string{LasPointCloudPoint_Format<Format>::attributeNames[i]});
            dataLayout.fieldType.push_back(LasPointCloudPoint_Format<Format>::getLasDataType(i));
            dataLayout.fieldByteSize.push_back(LasPointCloudPoint_Format<Format>::fieldByteSize[i]);
            dataLayout.fieldOffset.push_back(LasPointCloudPoint_Format<Format>::getFieldOffset(i));
            dataLayout.usePriorDataOffset.push_back(LasPointCloudPoint_Format<Format>::usePriorDataOffset[i]);
            dataLayout.isBitfield.push_back(LasPointCloudPoint_Format<Format>::isBitfield[i]);
            dataLayout.bitfieldSize.push_back(LasPointCloudPoint_Format<Format>::bitfieldSize[i]);
            dataLayout.bitfieldOffset.push_back(LasPointCloudPoint_Format<Format>::bitfieldOffset[i]);
        }
        size_t offset = LasPointCloudPoint_Format<Format>::minimumRecordByteSize;
        // now get all the custom attributes
        for (const auto& attrName : pointcloudPointAccessInterface.attributeList()) {
            // find the attribute in the list
            auto it = std::find(dataLayout.attributeNames.begin(), dataLayout.attributeNames.end(), attrName);
            if (it == dataLayout.attributeNames.end()) { // if not found, it is a new custom attribute
                // try to get it
                auto attrOpt = pointcloudPointAccessInterface.getAttributeByName(attrName.c_str());
                if (attrOpt.has_value()) {
                    auto attrValue = attrOpt.value();
                    auto [type, size] = std::visit(visitorCustomAttributes, attrValue);
                    dataLayout.attributeNames.push_back(attrName);
                    dataLayout.fieldType.push_back(type);
                    dataLayout.fieldByteSize.push_back(size);
                    dataLayout.fieldOffset.push_back(offset);
                    dataLayout.usePriorDataOffset.push_back(false);
                    dataLayout.isBitfield.push_back(false);
                    dataLayout.bitfieldSize.push_back(0);
                    dataLayout.bitfieldOffset.push_back(0);
                    
                    dataLayout.recordByteSize += size;
                    offset += size;
                    nbAttributes++;
                }
            }
        }
        return dataLayout;
    } else { // default case
        return getLasDataLayoutFromPointcloudPoint_helper<Format + 1>(pointcloudPointAccessInterface, format);
    }
}

template <class D>
template <size_t N>
uint8_t LasPointCloudPoint_Base<D>::getLasDataType(size_t id) {
    if constexpr (N > D::nbAttributes - 1) {
        return getLasDataType<D::nbAttributes - 1>(D::nbAttributes - 1);
    } else if (N == id) {
        return getLasDataType<returnType<N>>();
    } else {
        return getLasDataType<N+1>(id);
    }
}

template <class D>
template <typename T>
uint8_t LasPointCloudPoint_Base<D>::getLasDataType() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return 1; // uint8_t
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return 2; // int8_t
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return 3; // uint16_t
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return 4; // int16_t
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return 5; // uint32_t
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return 6; // int32_t
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return 7; // uint64_t
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return 8; // int64_t
    } else if constexpr (std::is_same_v<T, float>) {
        return 9; // float
    } else if constexpr (std::is_same_v<T, double>) {
        return 10; // double
    } else {
        return 0; // Default case, unknown type which is a sequence of bytes
    }
}

template <class D>
template <size_t N>
size_t LasPointCloudPoint_Base<D>::getFieldOffset(size_t id) {
    if constexpr(N > D::nbAttributes - 1) {
        return getFieldOffset<D::nbAttributes - 1>(D::nbAttributes - 1);
    } else if (N == id) {
        return offset<N>;
    } else {
        return getFieldOffset<N+1>(id);
    }
}

template <class D>
template <size_t N>
std::optional<PointCloudGenericAttribute> LasPointCloudPoint_Base<D>::getAttributeById_helper(size_t id) const {
    if constexpr (N >= D::nbAttributes) {
        return std::nullopt;
    } else {
        if (N == id) {
            // depending on the type, we can have different behavior
            if constexpr (std::is_same_v<returnType<N>, std::string>) {
                const auto begin = getRecordDataBuffer() + offset<N>;
                const auto end = std::find(begin, begin + size<N>, '\0');
                return std::string{begin, end}; // return the string
            } else if constexpr (is_vector_v<returnType<N>>) { // if the type is a vector
                // value type of the vector
                using value_type = typename std::remove_cv_t<returnType<N>>::value_type;
                // make sure that count*sizeof(value_type) = size
                static_assert(sizeof(value_type) * count<N> == size<N>, "The size of the vector is not correct");
                return vectorFromBytes<value_type>(getRecordDataBuffer() + offset<N>, count<N>);
            } else { // just a basic type
                // test the size
                static_assert(sizeof(returnType<N>) == size<N>, "The size of the attribute is not correct");
                // test if the type is a bitfield
                if constexpr (isBitfield<N>) {
                    static_assert(sizeof(typeof(bitfieldSize<N>)) >= sizeof(returnType<N>),
                    "The size of the bitfield is too large");
                    auto data = fromBytes<returnType<N>>(getRecordDataBuffer() + offset<N>);
                    // shift and mask
                    data = data >> bitfieldOffset<N> & ((1 << bitfieldSize<N>) - 1);
                    return data;
                } else {
                    return fromBytes<returnType<N>>(getRecordDataBuffer() + offset<N>);;
                }
            }
        } else {
            return getAttributeById_helper<N+1>(id);
        }
    }
}

}
}
