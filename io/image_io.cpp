#include "image_io.h"

#ifdef STEVI_IO_USE_OPENEXR
#include "read_exr.h"
#endif

#include "read_flo.h"

namespace StereoVision {
namespace IO {

#define EXR_LAYER_EXT ".exrlayer"
#define EXR_CHANNEL_EXT ".exrchannel"

#define EXR_INNER_SEPARATOR '/'

#ifdef STEVI_IO_USE_OPENEXR
template<typename ImgType>
Multidim::Array<ImgType, 3> readExrLayerAsImage(std::string const& path) {

    std::string exr_layer_ext = EXR_LAYER_EXT;

    if (path.size() < exr_layer_ext.size() or 0 != path.compare(path.size()-exr_layer_ext.size(), exr_layer_ext.size(), exr_layer_ext)) {
        return Multidim::Array<ImgType, 3>();
    }

    int pos = path.find_last_of(EXR_INNER_SEPARATOR);

    if (pos <= 0 or pos >= path.size()) {
        return Multidim::Array<ImgType, 3>();
    }

    int nCharLayer = path.size()-1-pos-exr_layer_ext.size();

    if (nCharLayer <= 0) {
        return Multidim::Array<ImgType, 3>();
    }

    std::string file = path.substr(0,pos);
    std::string layer = path.substr(pos+1,nCharLayer);

    return readExrLayer<ImgType>(file, layer);

}

template<typename ImgType>
Multidim::Array<ImgType, 3> readExrChannelAsImage(std::string const& path) {

    std::string exr_channel_ext = EXR_CHANNEL_EXT;

    if (path.size() < exr_channel_ext.size() or 0 != path.compare(path.size()-exr_channel_ext.size(), exr_channel_ext.size(), exr_channel_ext)) {
        return Multidim::Array<ImgType, 3>();
    }

    int pos = path.find_last_of(EXR_INNER_SEPARATOR);

    if (pos <= 0 or pos >= path.size()) {
        return Multidim::Array<ImgType, 3>();
    }

    int nCharLayer = path.size()-1-pos-exr_channel_ext.size();

    if (nCharLayer <= 0) {
        return Multidim::Array<ImgType, 3>();
    }

    std::string file = path.substr(0,pos);
    std::string channel = path.substr(pos+1,nCharLayer);

    Multidim::Array<ImgType, 2> channelArray = readExrChannel<ImgType>(file, channel);

    std::array<int,3> shape = {channelArray.shape()[0], channelArray.shape()[1],1};
    std::array<int,3> strides = {channelArray.strides()[0], channelArray.strides()[1],shape[0]*shape[1]};
    constexpr bool manage = true;
    return Multidim::Array<ImgType, 3>(channelArray.takePointer(), shape,strides, manage);
}
#endif

template<typename ImgType>
Multidim::Array<ImgType, 3> readImage(std::string const& fileName) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	std::string stevimg_ext = ".stevimg";
	if (fileName.size() >= stevimg_ext.size() && 0 == fileName.compare(fileName.size()-stevimg_ext.size(), stevimg_ext.size(), stevimg_ext)) {
		return readStevimg<ImgType, 3>(fileName);
	}

	std::string middleburry_flo_ext = ".flo";
    if (fileName.size() >= middleburry_flo_ext.size() && 0 == fileName.compare(fileName.size()-middleburry_flo_ext.size(), middleburry_flo_ext.size(), middleburry_flo_ext)) {
		return readFloImg<ImgType>(fileName);
	}

    #ifdef STEVI_IO_USE_OPENEXR
    std::string exr_layer_ext = EXR_LAYER_EXT;
    std::string exr_channel_ext = EXR_CHANNEL_EXT;

    if (fileName.size() >= exr_layer_ext.size() && 0 == fileName.compare(fileName.size()-exr_layer_ext.size(), exr_layer_ext.size(), exr_layer_ext)) {
        return readExrLayerAsImage<ImgType>(fileName);
    }

    if (fileName.size() >= exr_channel_ext.size() && 0 == fileName.compare(fileName.size()-exr_channel_ext.size(), exr_channel_ext.size(), exr_channel_ext)) {
        return readExrChannelAsImage<ImgType>(fileName);
    }
    #endif

    try {
        cimg_library::CImg<ImgType> image(fileName.c_str());

        if (image.is_empty()) { // could not read image
            return Multidim::Array<ImgType, 3>();
        }

        typename Multidim::Array<ImgType, 3>::ShapeBlock shape = {image.height(), image.width(), image.spectrum()};

        Multidim::Array<ImgType, 3> r(shape);

        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int c = 0; c < shape[2]; c++) {
                    r.template at<Nc>(i,j,c) = image(j,i,c); //invert the height and width coordinates to match the convention followed by libstevi
                }
            }
        }

        return r;

    } catch (cimg_library::CImgException const& e) {
        return Multidim::Array<ImgType, 3>();
    }

    return Multidim::Array<ImgType, 3>();
}

template<typename ImgType, typename InType>
bool writeImage(std::string const& fileName, Multidim::Array<InType, 3> const& image) {

    constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

    if (image.empty()) {
        return false;
    }

    std::string stevimg_ext = ".stevimg";
    if (fileName.size() >= stevimg_ext.size() && 0 == fileName.compare(fileName.size()-stevimg_ext.size(), stevimg_ext.size(), stevimg_ext)) {
        return writeStevimg<ImgType, InType, 3>(fileName, image);
    }

    typename Multidim::Array<InType, 3>::ShapeBlock shape = image.shape();

    cimg_library::CImg<ImgType> img(shape[1], shape[0], 1, shape[2]);


    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            for (int c = 0; c < shape[2]; c++) {
                img(j,i,c) = static_cast<ImgType>(image.template value<Nc>(i,j,c)); //invert the height and width coordinates to match the convention followed by CImg
            }
        }
    }

    try {
        img.save(fileName.c_str());
    } catch (cimg_library::CImgException const& e) {
        return false;
    }

    return true;
}

template<typename ImgType, typename InType>
bool writeImage(std::string const& fileName, Multidim::Array<InType, 2> const& image) {

    constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

    if (image.empty()) {
        return false;
    }

    std::string stevimg_ext = ".stevimg";
    if (fileName.size() >= stevimg_ext.size() && 0 == fileName.compare(fileName.size()-stevimg_ext.size(), stevimg_ext.size(), stevimg_ext)) {
        return writeStevimg<ImgType, InType, 2>(fileName, image);
    }

    typename Multidim::Array<InType, 2>::ShapeBlock shape = image.shape();

    cimg_library::CImg<ImgType> img(shape[1], shape[0], 1, 1);


    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {

            if (std::is_same_v<InType, bool>) {

            }

            img(j,i) = static_cast<ImgType>(image.template value<Nc>(i,j)); //invert the height and width coordinates to match the convention followed by CImg

            if (std::is_same_v<InType, bool>) {
                img(j,i) = (image.template value<Nc>(i,j)) ? TypesManipulations::defaultWhiteLevel<ImgType>() : TypesManipulations::defaultBlackLevel<ImgType>();
            }
        }
    }

    try {
        img.save(fileName.c_str());
    } catch (cimg_library::CImgException const& e) {
        return false;
    }

    return true;
}

//Implement the readImage function instances

template Multidim::Array<int8_t, 3> readImage(std::string const& fileName);
template Multidim::Array<uint8_t, 3> readImage(std::string const& fileName);

template Multidim::Array<int16_t, 3> readImage(std::string const& fileName);
template Multidim::Array<uint16_t, 3> readImage(std::string const& fileName);

template Multidim::Array<int32_t, 3> readImage(std::string const& fileName);
template Multidim::Array<uint32_t, 3> readImage(std::string const& fileName);

template Multidim::Array<int64_t, 3> readImage(std::string const& fileName);
template Multidim::Array<uint64_t, 3> readImage(std::string const& fileName);

template Multidim::Array<float, 3> readImage(std::string const& fileName);
template Multidim::Array<double, 3> readImage(std::string const& fileName);

//implement the writeImage function instance

#define implementWrite4Type(Type) \
    template bool writeImage<Type, bool>(std::string const& fileName, Multidim::Array<bool, 3> const& image); \
    template bool writeImage<Type, int8_t>(std::string const& fileName, Multidim::Array<int8_t, 3> const& image); \
    template bool writeImage<Type, uint8_t>(std::string const& fileName, Multidim::Array<uint8_t, 3> const& image);\
    template bool writeImage<Type, int16_t>(std::string const& fileName, Multidim::Array<int16_t, 3> const& image); \
    template bool writeImage<Type, uint16_t>(std::string const& fileName, Multidim::Array<uint16_t, 3> const& image);\
    template bool writeImage<Type, int32_t>(std::string const& fileName, Multidim::Array<int32_t, 3> const& image); \
    template bool writeImage<Type, uint32_t>(std::string const& fileName, Multidim::Array<uint32_t, 3> const& image);\
    template bool writeImage<Type, int64_t>(std::string const& fileName, Multidim::Array<int64_t, 3> const& image); \
    template bool writeImage<Type, uint64_t>(std::string const& fileName, Multidim::Array<uint64_t, 3> const& image);\
    template bool writeImage<Type, float>(std::string const& fileName, Multidim::Array<float, 3> const& image); \
    template bool writeImage<Type, double>(std::string const& fileName, Multidim::Array<double, 3> const& image); \
    \
    template bool writeImage<Type, bool>(std::string const& fileName, Multidim::Array<bool, 2> const& image); \
    template bool writeImage<Type, int8_t>(std::string const& fileName, Multidim::Array<int8_t, 2> const& image); \
    template bool writeImage<Type, uint8_t>(std::string const& fileName, Multidim::Array<uint8_t, 2> const& image);\
    template bool writeImage<Type, int16_t>(std::string const& fileName, Multidim::Array<int16_t, 2> const& image); \
    template bool writeImage<Type, uint16_t>(std::string const& fileName, Multidim::Array<uint16_t, 2> const& image);\
    template bool writeImage<Type, int32_t>(std::string const& fileName, Multidim::Array<int32_t, 2> const& image); \
    template bool writeImage<Type, uint32_t>(std::string const& fileName, Multidim::Array<uint32_t, 2> const& image);\
    template bool writeImage<Type, int64_t>(std::string const& fileName, Multidim::Array<int64_t, 2> const& image); \
    template bool writeImage<Type, uint64_t>(std::string const& fileName, Multidim::Array<uint64_t, 2> const& image);\
    template bool writeImage<Type, float>(std::string const& fileName, Multidim::Array<float, 2> const& image); \
    template bool writeImage<Type, double>(std::string const& fileName, Multidim::Array<double, 2> const& image);


implementWrite4Type(char)

implementWrite4Type(int8_t)
implementWrite4Type(uint8_t)

implementWrite4Type(int16_t)
implementWrite4Type(uint16_t)

implementWrite4Type(int32_t)
implementWrite4Type(uint32_t)

implementWrite4Type(int64_t)
implementWrite4Type(uint64_t)

implementWrite4Type(float)
implementWrite4Type(double)

} //namespace IO
} //namespace StereoVision
