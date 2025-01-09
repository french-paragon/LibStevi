#include <cstring>
#include <array>
#include <iostream>
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
 * The data will be in an invalid state until gotoNext is called.
 * @param reader the istream
 * @param recordByteSize the size of the point record
 * @param format the record format number
 * @param dataBuffer the data buffer
 * @return std::unique_ptr<LasPointCloudPoint>
 */
static std::unique_ptr<LasPointCloudPoint> createLasPointCloudPoint(std::unique_ptr<std::istream> reader, size_t recordByteSize, size_t recordFormatNumber, char* dataBuffer = nullptr);

template <size_t N = size_t{0}>
static std::unique_ptr<LasPointCloudPoint> createLasPointCloudPoint_helper(std::unique_ptr<std::istream> reader, size_t recordByteSize, size_t recordFormatNumber, char* dataBuffer);

template <class D>
class LasPointCloudPoint_Base : public LasPointCloudPoint {
public:

    LasPointCloudPoint_Base(std::unique_ptr<std::istream> reader, size_t recordByteSize, char* dataBuffer) :
        reader{std::move(reader)},
        LasPointCloudPoint(recordByteSize, dataBuffer) { }

    LasPointCloudPoint_Base(std::unique_ptr<std::istream> reader, size_t recordByteSize) :
        reader{std::move(reader)},
        LasPointCloudPoint(recordByteSize) { }

protected:
    const std::unique_ptr<std::istream> reader;

private:
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

public:
    PtGeometry<PointCloudGenericAttribute> getPointPosition() const override;

    std::optional<PtColor<PointCloudGenericAttribute>> getPointColor() const override;

    std::optional<PointCloudGenericAttribute> getAttributeById(int id) const override {
        return getAttributeById_helper(id);
    }

    std::optional<PointCloudGenericAttribute> getAttributeByName(const char* attributeName) const override;

    std::vector<std::string> attributeList() const override;

    bool gotoNext() override;
private:
    /**
     * @brief helper functions to get the attribute by id.
     * 
     * If N > nbAttributes, return nullopt.
     * If N == id, return the attribute.
     * If N != id, recursively call the function with N+1.
     */
    template<size_t N = size_t{0}>
    std::optional<PointCloudGenericAttribute> getAttributeById_helper(int id) const {
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
                        static_assert(sizeof(typeof(bitfieldSize<N>)) >= sizeof(returnType<N>), "The size of the bitfield is too large");
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
        uint8_t, int8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint16_t>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "ScanDirectionFlag"sv, "EdgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
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
        uint8_t, int8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint16_t, double>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "ScanDirectionFlag"sv, "EdgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
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
        uint8_t, int8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "ScanDirectionFlag"sv, "EdgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
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
        uint8_t, int8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint16_t, double, uint16_t, uint16_t, uint16_t>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "ScanDirectionFlag"sv, "EdgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
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
        uint8_t, int8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint16_t, double, uint8_t, uint64_t, uint32_t, float,
        float, float, float>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "ScanDirectionFlag"sv, "EdgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
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
        uint8_t, int8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint16_t, double, uint16_t, uint16_t, uint16_t, uint8_t,
        uint64_t, uint32_t, float, float, float, float>;

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "ScanDirectionFlag"sv, "EdgeOfFlightLineFlag"sv, "classification"sv, "syntheticFlag"sv,
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

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
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

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
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

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
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

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
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

    constexpr static auto attributeNames = std::array{"x"sv, "y"sv, "z"sv, "intensity"sv, "ReturnNumber"sv,
        "NumberOfReturns"sv, "syntheticFlag"sv, "keyPointFlag"sv, "withheldFlag"sv, "overlapFlag"sv, "scannerChannel"sv,
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
std::optional<PointCloudGenericAttribute> LasPointCloudHeader::getAttributeById(int id) const {
    // TODO: add support for VLRs and EVLRs
    if (id < 0) {
        return std::nullopt;
    } else if (id < publicHeaderBlock.publicHeaderAttributes.size()) {
            return publicHeaderBlock.getPublicHeaderAttributeById(id);
    } else {
        return std::nullopt;
    }
}

std::optional<PointCloudGenericAttribute> LasPointCloudHeader::getAttributeByName(const char *attributeName) const {
    // TODO: add support for VLRs and EVLRs
    auto it = std::find(publicHeaderBlock.publicHeaderAttributes.begin(), publicHeaderBlock.publicHeaderAttributes.end(), attributeName);
    if (it != publicHeaderBlock.publicHeaderAttributes.end()) {
        auto id = std::distance(publicHeaderBlock.publicHeaderAttributes.begin(), it);
        return publicHeaderBlock.getPublicHeaderAttributeById(id);
    } else {
        return std::nullopt;
    }
}
std::vector<std::string> LasPointCloudHeader::attributeList() const {
    // TODO: add support for VLRs and EVLRs
    return publicHeaderBlock.publicHeaderAttributes;
}
std::unique_ptr<LasPointCloudHeader> LasPointCloudHeader::readHeader(std::istream &reader)
{
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

std::optional<LasPublicHeaderBlock> LasPublicHeaderBlock::readPublicHeader(std::istream &reader) {
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
    reader.read(reinterpret_cast<char*>(&header.startOfFirstExtendedVariableLengthRecord), startOfFirstExtendedVariableLengthRecord_size);
    reader.read(reinterpret_cast<char*>(&header.numberOfExtendedVariableLengthRecords), numberOfExtendedVariableLengthRecords_size);
    reader.read(reinterpret_cast<char*>(&header.numberOfPointRecords), numberOfPointRecords_size);
    reader.read(reinterpret_cast<char*>(header.numberOfPointsByReturn.data()), numberOfPointsByReturn_size);

    if (reader.fail()) return std::nullopt;

    return header;

}

void LasPublicHeaderBlock::writePublicHeader(std::ostream &writer, const LasPublicHeaderBlock &header) {
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
    writer.write(reinterpret_cast<const char*>(&header.startOfFirstExtendedVariableLengthRecord), startOfFirstExtendedVariableLengthRecord_size);
    writer.write(reinterpret_cast<const char*>(&header.numberOfExtendedVariableLengthRecords), numberOfExtendedVariableLengthRecords_size);
    writer.write(reinterpret_cast<const char*>(&header.numberOfPointRecords), numberOfPointRecords_size);
    writer.write(reinterpret_cast<const char*>(header.numberOfPointsByReturn.data()), numberOfPointsByReturn_size);
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
        return std::string{projectID_GUID_Data4.begin(), projectID_GUID_Data4.end()};
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
    return PtGeometry<PointCloudGenericAttribute>{
        fromBytes<returnType<D::x_id>>(getRecordDataBuffer() + offset<D::x_id>),
        fromBytes<returnType<D::y_id>>(getRecordDataBuffer() + offset<D::y_id>),
        fromBytes<returnType<D::z_id>>(getRecordDataBuffer() + offset<D::z_id>)};
}

template <class D>
inline std::optional<PtColor<PointCloudGenericAttribute>> LasPointCloudPoint_Base<D>::getPointColor() const {
    if constexpr (containsColor) {
        return PtColor<PointCloudGenericAttribute>{
            fromBytes<returnType<D::r_id>>(getRecordDataBuffer() + offset<D::r_id>),
            fromBytes<returnType<D::g_id>>(getRecordDataBuffer() + offset<D::g_id>),
            fromBytes<returnType<D::b_id>>(getRecordDataBuffer() + offset<D::b_id>),
            std::numeric_limits<returnType<D::r_id>>::max() // no alpha channel
            };
    } else {
        return std::nullopt;
    }
}

template <class D>
std::optional<PointCloudGenericAttribute> LasPointCloudPoint_Base<D>::getAttributeByName(const char *attributeName) const {
    constexpr auto begin = D::attributeNames.begin();
    constexpr auto end = begin + D::nbAttributes;
    auto it = std::find(begin, end, attributeName);
    if (it != end) {
        return getAttributeById(std::distance(begin, it));
    }
    return std::nullopt; // Attribute not found
}

template <class D>
std::vector<std::string> LasPointCloudPoint_Base<D>::attributeList() const {
    return std::vector<std::string>(D::attributeNames.begin(), D::attributeNames.begin() + D::nbAttributes);
}

template <class Derived>
bool LasPointCloudPoint_Base<Derived>::gotoNext() {
    reader->read(dataBuffer, recordByteSize);
    if (!reader->good()) return false;
    return true;
}

LasPointCloudPoint::LasPointCloudPoint(size_t recordByteSize) :
    LasPointCloudPoint(recordByteSize, nullptr)
{
    dataBufferContainer.resize(recordByteSize);
    dataBuffer = dataBufferContainer.data();
}

LasPointCloudPoint::LasPointCloudPoint(size_t recordByteSize, char *dataBuffer) :
    recordByteSize{recordByteSize}, dataBuffer{dataBuffer} { }

std::optional<FullPointCloudAccessInterface> openPointCloudLas(const std::filesystem::path &lasFilePath) {
    // open the file
    auto reader = std::make_unique<ifstreamCustomBuffer<lasFileReaderBufferSize>>();

    constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
    reader->precision(maxPrecision); // set the precision for the reader

    reader->open(lasFilePath, std::ios_base::binary);

    if (!reader->is_open()) return std::nullopt;

    // read the header
    auto header = LasPointCloudHeader::readHeader(*reader);
    // test if header ptr is not null
    if (header == nullptr) {
        return std::nullopt;
    }

    // create a point cloud
    auto pointCloud = createLasPointCloudPoint(std::move(reader), header->publicHeaderBlock.pointDataRecordLength, header->publicHeaderBlock.pointDataRecordFormat);

    if (pointCloud == nullptr) {
        return std::nullopt;
    }

    // return the point cloud
    if (pointCloud->gotoNext()) {
            FullPointCloudAccessInterface fullPointInterface;
            fullPointInterface.headerAccess = std::move(header);
            fullPointInterface.pointAccess = std::move(pointCloud);
            return fullPointInterface;
    }

    return std::nullopt;
}

template <size_t N>
std::unique_ptr<LasPointCloudPoint> createLasPointCloudPoint_helper(std::unique_ptr<std::istream> reader, size_t recordByteSize, size_t recordFormatNumber, char *dataBuffer) {
    if constexpr (N > 10) {
        return nullptr;
    } else if (N == recordFormatNumber) {
        if (dataBuffer == nullptr) {
            return std::make_unique<LasPointCloudPoint_Format<N>>(std::move(reader), recordByteSize);
        } else  {
            return std::make_unique<LasPointCloudPoint_Format<N>>(std::move(reader), recordByteSize, dataBuffer);
        }
    } else {
        return createLasPointCloudPoint_helper<N+1>(std::move(reader), recordByteSize, recordFormatNumber, dataBuffer);
    }
}

std::unique_ptr<LasPointCloudPoint> createLasPointCloudPoint(std::unique_ptr<std::istream> reader, size_t recordByteSize, size_t recordFormatNumber, char *dataBuffer) {
    return createLasPointCloudPoint_helper(std::move(reader), recordByteSize, recordFormatNumber, dataBuffer);
}

}
}
