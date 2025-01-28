#ifndef READ_EXR_H
#define READ_EXR_H


#include "../utils/types_manipulations.h"

#include <MultidimArrays/MultidimArrays.h>

#include <string>
#include <fstream>

#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfIO.h>
#include <OpenEXR/ImfFrameBuffer.h>
// #include <OpenEXR/ImathBox.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfAttribute.h>
#include <OpenEXR/ImfStandardAttributes.h>

namespace StereoVision {
namespace IO {

inline bool isExrFile(const char * fileName) {
    std::ifstream f (fileName, std::ios_base::binary);
    char b[4];
    f.read (b, sizeof (b));
    return !!f && b[0] == 0x76 && b[1] == 0x2f && b[2] == 0x31 && b[3] == 0x01;
}

template<typename T>
Multidim::Array<T,2> readExrChannel(std::string file, std::string channel) {

    const char * fileName = file.c_str();

    if (!isExrFile(fileName)) {
        return Multidim::Array<T,2>();
    }

    Imf::InputFile exr_file (fileName);

    const Imf::ChannelList &channels = exr_file.header().channels();
    const Imf::Channel *channelPtr = channels.findChannel(channel.c_str());


    if (channelPtr == nullptr) {
        return Multidim::Array<T,2>();
    }

    Imath::Box2i dw = exr_file.header().dataWindow();
    int width  = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;

    Imf::Array2D<float> pixels;
    pixels.resizeErase(height, width);

    Multidim::Array<float,2> r(height, width);

    Imf::FrameBuffer frameBuffer;

    frameBuffer.insert (channel.c_str(),                                  // name
                        Imf::Slice (Imf::FLOAT,                         // type
                                    static_cast<char*>(static_cast<void*>(&r.atUnchecked(0,0))),
                                    sizeof (float) * 1,    // xStride
                                    sizeof (float) * static_cast<unsigned long>(width),// yStride
                                    1, 1,                          // x/y sampling
                                    FLT_MAX));                     // fillValue

    exr_file.setFrameBuffer (frameBuffer);
    exr_file.readPixels (dw.min.y, dw.max.y);

    return r.cast<T>();
}

template<typename T>
Multidim::Array<T,3> readExrLayer(std::string file, std::string layer) {

    const char * fileName = file.c_str();

    if (!isExrFile(fileName)) {
        return Multidim::Array<T,3>();
    }

    Imf::InputFile exr_file (fileName);

    const Imf::ChannelList &channels = exr_file.header().channels();

    Imf::ChannelList::ConstIterator layerBegin, layerEnd;
    channels.channelsInLayer (layer, layerBegin, layerEnd);

    Imath::Box2i dw = exr_file.header().dataWindow();
    int width  = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;

    int nChannel = 0;
    for (Imf::ChannelList::ConstIterator j = layerBegin; j != layerEnd; ++j) {
        nChannel++;
    }

    if (nChannel <= 0) {
        return Multidim::Array<T,3>();
    }

    Multidim::Array<T,3> r({height, width, nChannel}, {width, 1, height*width});

    Imf::FrameBuffer frameBuffer;

    int n_chan = 0;
    for (Imf::ChannelList::ConstIterator j = layerBegin; j != layerEnd; ++j, n_chan++) {

        frameBuffer.insert (j.name(),                                  // name
                            Imf::Slice (Imf::FLOAT,                         // type
                                        static_cast<char*>(static_cast<void*>(&r.atUnchecked(0, 0, n_chan))),
                                        sizeof (float) * 1,    // xStride
                                        sizeof (float) * static_cast<unsigned long>(width),// yStride
                                        1, 1,                          // x/y sampling
                                        FLT_MAX));

    }

    exr_file.setFrameBuffer (frameBuffer);
    exr_file.readPixels (dw.min.y, dw.max.y);

    return r;

}

}
}
#endif // READ_EXR_H
