#ifndef STEREOVISION_IO_FSTREAMCUSTOMBUFFER_H
#define STEREOVISION_IO_FSTREAMCUSTOMBUFFER_H

#include <fstream>
// ifstream, ofstream and fstream with custom buffer
namespace StereoVision {
namespace IO {

/// @brief  a fstream with a custom buffer
template<size_t bufferSize>
class fstreamCustomBuffer : public std::fstream
{
public:
    inline fstreamCustomBuffer() : std::fstream() {
        this->rdbuf()->pubsetbuf(buffer, bufferSize);
    }
private:
    char buffer[bufferSize];
};

/// @brief an ifstream with a custom buffer
template<size_t bufferSize>
class ifstreamCustomBuffer : public std::ifstream
{
public:
    inline ifstreamCustomBuffer() : std::ifstream() {
        this->rdbuf()->pubsetbuf(buffer, bufferSize);
    }
private:
    char buffer[bufferSize];
};

/// @brief an ifstream with a custom buffer
template<size_t bufferSize>
class ofstreamCustomBuffer : public std::ofstream
{
public:
    inline ofstreamCustomBuffer() : std::ofstream() {
        this->rdbuf()->pubsetbuf(buffer, bufferSize);
    }
private:
    char buffer[bufferSize];
};

}
}

#endif //STEREOVISION_IO_FSTREAMCUSTOMBUFFER_H