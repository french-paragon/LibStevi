#ifndef READ_FLO_H
#define READ_FLO_H

#include <MultidimArrays/MultidimArrays.h>

#include <string>
#include <fstream>

namespace StereoVision {
namespace IO {


template<typename T>
Multidim::Array<T,3> readFloImg(std::string file) {

	std::ifstream inFile(file, std::ios::binary);

	if (!inFile.is_open()) {
		return Multidim::Array<T,3>();
	}

	const char magicCheck[4] = {'P','I','E','H'};
	char magic[4];
	inFile.read(magic,4);

	for (int i = 0; i < 4; i++) {
		if (magic[i] != magicCheck[i]) {
			return Multidim::Array<T,3>();
		}
	}

	int32_t w;
	int32_t h;

	inFile >> w;
	inFile >> h;

	if (w <= 0 or h <= 0) {
		return Multidim::Array<T,3>();
	}

	int nBytes = 2*w*h*sizeof(float);

	Multidim::Array<float,3> data({h,w,2},{2*w,2,1});

	inFile.read(reinterpret_cast<char*>(&data.at(0,0,0)), nBytes);

	return data.cast<T>();

}

} // namespace IO
} // namespace StereoVision

#endif // READ_FLO_H
