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
#include "imageProcessing/checkBoardDetection.h"

#include <QTextStream>
#include <QString>
#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <string>

int main(int argc, char** argv) {

	QTextStream out(stdout);

	if (argc < 3) { //no input image
		out << "No input image and or outfile provided" << Qt::endl;
		return 1;
	}

	Multidim::Array<float, 3> img = StereoVision::IO::readImage<float>(std::string(argv[1]));

	if (img.empty()) {
		out << "impossible to read image: " << argv[1] << Qt::endl;
		return 1;
	} else {
		out << "Read image: " << argv[1] << Qt::endl;
	}

	float maxGrey = 1;
	Multidim::Array<float, 2> greyscale(img.shape()[0], img.shape()[1]);

	for (int i = 0; i < img.shape()[0]; i++) {
		for (int j = 0; j < img.shape()[1]; j++) {

			greyscale.atUnchecked(i,j) = 0;

			for (int c = 0; c < img.shape()[2]; c++) {
				greyscale.atUnchecked(i,j) += img.valueUnchecked(i,j,c);
			}

			greyscale.atUnchecked(i,j) /= img.shape()[2];

			if (greyscale.valueUnchecked(i,j) > maxGrey) {
				maxGrey = greyscale.valueUnchecked(i,j);
			}

		}
	}

	for (int i = 0; i < img.shape()[0]; i++) {
		for (int j = 0; j < img.shape()[1]; j++) {
			greyscale.atUnchecked(i,j) /= maxGrey;
		}
	}

	auto candidates = StereoVision::checkBoardCornersCandidates(greyscale, 1, 2, 6.);
	auto filtereds = StereoVision::checkBoardFilterCandidates(greyscale, candidates, 0.2, 0.6);

	out << "Found " << candidates.size() << " candidates" << Qt::endl;
	out << "Filtered " << filtereds.size() << " candidates" << Qt::endl;

	auto selected = StereoVision::isolateCheckBoard(filtereds, 0.3, 0.25);

	out << "Selected " << selected.nPointsFound() << " candidates" << Qt::endl;

	auto refined = StereoVision::refineCheckBoardCorners(greyscale, selected);

	out << "Refined " << refined.size() << " points" << Qt::endl;

	QFileInfo info(argv[2]);
	QString rawName = info.absoluteDir().absoluteFilePath( QString("raw_") + info.fileName());
	QFile rawFile(rawName);


	if (!rawFile.open(QIODevice::WriteOnly)) {
		out << "impossible to open file: " << rawName << Qt::endl;
		return 1;
	}

	QTextStream fraw(&rawFile);

	for (auto & candidate : candidates) {
		fraw << candidate.pix_coord_x << ',' << candidate.pix_coord_y << ','
			 << candidate.lambda_min << ',' << candidate.lambda_max << ','
			 << candidate.main_dir << Qt::endl;
	}

	rawFile.close();

	QFile outFile(argv[2]);

	if (!outFile.open(QIODevice::WriteOnly)) {
		out << "impossible to open file: " << argv[2] << Qt::endl;
		return 1;
	}
	QTextStream fout(&outFile);

	for (auto& point : refined) {

		int i = point.grid_coord_y;
		int j = point.grid_coord_x;

		if (!selected.hasPointInCoord(i,j)) {
			continue;
		}

		auto candidate = selected.pointInCoord(i,j).value();

		fout << point.pix_coord_x << ',' << point.pix_coord_y << ','
			 << candidate.lambda_min << ',' << candidate.lambda_max << ','
			 << candidate.main_dir << Qt::endl;
	}

	outFile.close();

	return 0;

}
