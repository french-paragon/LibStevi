/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022  Paragon<french.paragon@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "io/image_io.h"
#include "imageProcessing/meanShiftClustering.h"
#include "imageProcessing/colorConversions.h"

#include <QTextStream>
#include <QString>
#include <QDir>
#include <QFileInfo>

#include <string>

int main(int argc, char** argv) {

	QTextStream out(stdout);

	if (argc < 2) { //no input image
		out << "No input image provided" << Qt::endl;
		return 1;
	}

	Multidim::Array<float, 3> img = StereoVision::IO::readImage<float>(std::string(argv[1]));

	if (img.empty()) {
		out << "impossible to read image: " << argv[1] << Qt::endl;
		return 1;
	} else {
		out << "Read image: " << argv[1] << Qt::endl;
	}

	QFileInfo info((QString(argv[1])));

	QDir outDir;

	if (argc >= 3) {
		outDir = QDir(argv[2]);
	} else {
		outDir = info.absoluteDir();
	}
	Multidim::Array<float, 3> intensityNormalized = StereoVision::ImageProcessing::normalizedIntensityRGBImage<float, float>(img, 128);

	bool ok = StereoVision::IO::writeImage<uint8_t>((outDir.filePath(info.baseName() + "_int_norm") + ".bmp" ).toStdString(), intensityNormalized);

	if (ok) {
		out << "\t" << "Intensity normalized file succesfully written to disk" << Qt::endl;
	} else {
		out << "\t" << "Failed to write intensity normalized file to disk" << Qt::endl;
		return 1;
	}

	std::function<float(std::vector<float> const&,std::vector<float> const&)> kernel = StereoVision::ImageProcessing::RadiusKernel<float>(5);

	Multidim::Array<float, 3> flat_approx = StereoVision::ImageProcessing::meanShiftClustering<float, 3, float>
			(intensityNormalized, kernel, 2, std::nullopt, 10);

	ok = StereoVision::IO::writeImage<uint8_t>((outDir.filePath(info.baseName() + "_flat_approx") + ".bmp" ).toStdString(), flat_approx);

	if (ok) {
		out << "\t" << "mean shift clustered file succesfully written to disk" << Qt::endl;
	} else {
		out << "\t" << "Failed to write mean shift clustered file to disk" << Qt::endl;
		return 1;
	}

	return 0;
}
