#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#define cimg_display 0 //no display from Cimg
#include <CImg.h>

#include "../utils/types_manipulations.h"

#include <MultidimArrays/MultidimArrays.h>

#include <iostream>
#include <fstream>
#include <cstdio>

namespace StereoVision {
namespace IO {

template<typename ImgType, int nDim>
bool stevImgFileMatchTypeAndDim(std::string const& fileName) {

	std::ifstream infile;
	infile.open(fileName, std::ios_base::in);

	if (infile.is_open()) {

		std::string line;
		getline( infile, line );

		std::stringstream strs;
		strs.str(line);

		std::string type;
		strs >> type;

		if (type != TypesManipulations::dtypeDescr<ImgType>()) {
			return false;
		}

		int nDimInFile;
		strs >> nDimInFile;

		if (nDimInFile > nDim) {
			return false;
		}

	}

	return true;
}

template<typename ImgType, typename InType, int nDim>
bool writeStevimg(std::string const& fileName, Multidim::Array<InType, nDim> const& image) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	if (!std::is_same_v<ImgType, InType>) {
		Multidim::Array<ImgType, nDim> converted(image.shape());

		typename Multidim::Array<ImgType, nDim>::IndexBlock idx;
		idx.setZero();

		for (int i = 0; i < image.flatLenght(); i++) {
			converted.template at<Nc>(idx) = static_cast<ImgType>(image.template value<Nc>(idx));
			idx.moveToNextIndex(image.shape());
		}

		return writeStevimg<ImgType, ImgType, nDim>(fileName, converted);
	}

	if (!image.isDense()) {
		Multidim::Array<InType, nDim> dense = image;
		return writeStevimg<ImgType, InType, nDim>(fileName, dense);
	}

	std::FILE* outfile = std::fopen(fileName.c_str(), "w");

	if (outfile) {

		std::stringstream strs;

		//first write the data type and number of dimensions
		strs << TypesManipulations::dtypeDescr<ImgType>() << ' ' << nDim;
		//write the shape
		for (int i = 0; i < nDim; i++) {
			strs << ' ' << image.shape()[i];
		}
		//write the strides
		for (int i = 0; i < nDim; i++) {
			strs << ' ' << image.strides()[i];
		}
		strs << std::endl;

		std::string str = strs.str();

		bool ok = true;
		ok = ok and std::fwrite(str.c_str(), sizeof (char), str.length(), outfile) == str.length();
		ok = ok and std::fwrite(&const_cast<Multidim::Array<InType, nDim>*>(&image)->atUnchecked(0), sizeof (ImgType), image.flatLenght(), outfile) == image.flatLenght();

		ok = ok and std::fclose(outfile) == 0;

		return ok;

	} else {
		errno = 0; //reset the error
		return false;
	}

}

template<typename ImgType, int nDim>
Multidim::Array<ImgType, nDim> readStevimg(std::string const& fileName) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	std::ifstream infile;
	infile.open(fileName, std::ios_base::in);

	if (infile.is_open()) {

		std::string line;
		getline( infile, line );

		std::stringstream strs;
		strs.str(line);

		std::string type;
		strs >> type;

		if (type != TypesManipulations::dtypeDescr<ImgType>()) {
			return Multidim::Array<ImgType, nDim>();
		}

		int nDimInFile;
		strs >> nDimInFile;

		if (nDimInFile > nDim) {
			return Multidim::Array<ImgType, nDim>();
		}

		typename Multidim::Array<ImgType, nDim>::ShapeBlock shape;
		typename Multidim::Array<ImgType, nDim>::ShapeBlock stride;

		for (int i = 0; i < nDim; i++) {
			if (i < nDimInFile) {
				strs >> shape[i];
			} else {
				shape[i] = 1;
			}
		}

		for (int i = 0; i < nDim; i++) {
			if (i < nDimInFile) {
				strs >> stride[i];
			} else {
				stride[i] = 1;
			}
		}

		Multidim::Array<ImgType, nDim> img(shape, stride);
		infile.read(reinterpret_cast<char*>(&img.atUnchecked(0)), sizeof (ImgType) * img.flatLenght());

		return img;

	} else {
		return Multidim::Array<ImgType, nDim>();
	}

}

template<typename ImgType>
Multidim::Array<ImgType, 3> readImage(std::string const& fileName) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	std::string stevimg_ext = ".stevimg";
	if (fileName.size() >= stevimg_ext.size() && 0 == fileName.compare(fileName.size()-stevimg_ext.size(), stevimg_ext.size(), stevimg_ext)) {
		return readStevimg<ImgType, 3>(fileName);
	}

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

} //namespace IO
} //namespace StereoVision

#endif // IMAGE_IO_H
