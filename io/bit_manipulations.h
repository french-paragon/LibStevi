#ifndef STEREOVISION_IO_BITMANIPULATION_H
#define STEREOVISION_IO_BITMANIPULATION_H

#include <type_traits>
#include <vector>
#include <cstring>

namespace StereoVision {
namespace IO {

// similar to std::bit_cast in C++20
template<class To, class From>
To bit_cast(const From& from) {
    // the types must have the same size
    static_assert(sizeof(To) == sizeof(From), "The size of To and From must be the same");
    // the types must be trivially copyable to avoid undefined behavior
    static_assert(std::is_trivially_copyable<To>::value, "To must be trivially copyable");
    static_assert(std::is_trivially_copyable<From>::value, "From must be trivially copyable");

    using To_ = std::remove_cv_t<To>;
    To_ to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}

// return an object of type To from a pointer to a sequence of bytes of type From.
template<typename To, typename From>
To fromBytes(const From* bytes) {
    // types that can alias any other type
    static_assert(std::is_same_v<std::remove_cv_t<From>, char> or std::is_same_v<std::remove_cv_t<From>, unsigned char>
                  || std::is_same_v<std::remove_cv_t<From>, std::byte>, "From must be a pointer that can alias any other type");
    static_assert(std::is_trivially_copyable<To>::value, "To must be trivially copyable");

    using To_ = std::remove_cv_t<To>;
    To_ to;
    std::memcpy(&to, bytes, sizeof(To));
    return to;
}

// return an vector of To containg N elements from a pointer to a sequence of bytes of type From representing to vector data.
template<typename To, typename From>
std::vector<To> VectorfromBytes(const From* bytes, size_t N) {
    // types that can alias any other type
    static_assert(std::is_same_v<std::remove_cv_t<From>, char> or std::is_same_v<std::remove_cv_t<From>, unsigned char>
                  || std::is_same_v<std::remove_cv_t<From>, std::byte>, "From must be a pointer that can alias any other type");
    static_assert(std::is_trivially_copyable<To>::value, "To must be trivially copyable");
    std::vector<To> to(N);
    for (auto i = 0; i < N; i++)
    {
        std::memcpy(&to[i], bytes + i * sizeof(To), sizeof(To));
    }
    return to;
}
} // namespace IO
} // namespace StereoVision

#endif //STEREOVISION_IO_BITMANIPULATION_H