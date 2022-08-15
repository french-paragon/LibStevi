#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#define cimg_display 0 //no display from Cimg
#include <CImg.h>

#include <MultidimArrays/MultidimArrays.h>

namespace StereoVision {
namespace IO {

template<typename ImgType>
Multidim::Array<ImgType, 3> readImage(std::string const& fileName) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

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

	typename Multidim::Array<InType, 2>::ShapeBlock shape = image.shape();

	cimg_library::CImg<ImgType> img(shape[1], shape[0], 1, 1);


	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			img(j,i) = static_cast<ImgType>(image.template value<Nc>(i,j)); //invert the height and width coordinates to match the convention followed by CImg
		}
	}

	try {
		img.save(fileName.c_str());
	} catch (cimg_library::CImgException const& e) {
		return false;
	}

	return true;
}

} //namespace IO
} //namespace StereoVision

#endif // IMAGE_IO_H
