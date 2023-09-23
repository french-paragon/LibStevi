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
#include "imageProcessing/intrinsicImageDecomposition.h"

#ifdef WITH_GUI

#include <QApplication>

#include <qImageDisplayWidget/imagewindow.h>
#include "gui/arraydisplayadapter.h"

#endif

#include <QTextStream>
#include <QString>
#include <QSet>
#include <QDir>
#include <QFileInfo>

#include <string>

using namespace StereoVision;
using namespace StereoVision::ImageProcessing;

int main(int argc, char** argv) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

    #ifdef WITH_GUI
    QApplication app(argc, argv);
    #endif

    QVector<QString> arguments;
    QSet<QString> options;

    for (int i = 1; i < argc; i++) {
        QString input(argv[i]);

        if (input.startsWith("-")) {
            options.insert(input);
        } else {
            arguments.push_back(input);
        }
    }

	QTextStream out(stdout);

    if (arguments.size() < 1) { //no input image
		out << "No input image provided" << Qt::endl;
		return 1;
	}

    Multidim::Array<float, 3> img = StereoVision::IO::readImage<float>(arguments[0].toStdString());

	if (img.empty()) {
        out << "impossible to read image: " << arguments[0] << Qt::endl;
		return 1;
    } else {
        out << "Read image: " << arguments[0] << Qt::endl;
        out << "Image shape: " << img.shape()[0] << "x" << img.shape()[1] << "x" <<  img.shape()[2] << Qt::endl;
	}

    #ifdef WITH_GUI
    QVector<QString> channelsNames = {"R", "G", "B"};

    StereoVision::Gui::ArrayDisplayAdapter<float> imgAdapter(&img, 0, 255);
    imgAdapter.configureOriginalChannelDisplay(channelsNames);
    QImageDisplay::ImageWindow imgWindow;
    imgWindow.setImage(&imgAdapter);
    imgWindow.setWindowTitle("Base image");
    #endif

    QFileInfo info((QString(arguments[0])));

	QDir outDir;

    if (arguments.size() >= 2) {
        outDir = QDir(arguments[1]);
	} else {
		outDir = info.absoluteDir();
	}

	float maxVal = 0;

	for (int i = 0; i < img.shape()[0]; i++) {
		for (int j = 0; j < img.shape()[1]; j++) {
			for (int c = 0; c < img.shape()[2]; c++) {
				float val = img.valueUnchecked(i,j,c);

				if (val > maxVal) {
					maxVal = val;
				}

				if (val < 1.0) {
					img.atUnchecked(i,j,c) = 1.0;
				}
			}
		}
	}

	out << "Image max val = " << maxVal << Qt::endl;

	/*Multidim::Array<float, 3> intensityNormalized = StereoVision::ImageProcessing::normalizedIntensityRGBImage<float, float>(img, 128);

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
	}*/

	IntrinsicImageDecomposition<float, 3> decomposition = autoRetinexWithNonLocalTextureConstraint<float, float>(img);

	out << "Decomposition computed!" << Qt::endl;
	out << "Out shape: " << decomposition.reflectance.shape()[0] << ' ' << decomposition.reflectance.shape()[1] << Qt::endl;

	float minS = decomposition.shading.at<Nc>(0,0,0);
	float maxS = decomposition.shading.at<Nc>(0,0,0);

	float minR = decomposition.reflectance.at<Nc>(0,0,0);
	float maxR = decomposition.reflectance.at<Nc>(0,0,0);

	float maxE = 0;

	Multidim::Array<float,3> reconstructed(img.shape());
	Multidim::Array<float,3> error(img.shape());

	for (int i = 0; i < img.shape()[0]; i++) {
		for (int j = 0; j < img.shape()[1]; j++) {

			float valS = decomposition.shading.value<Nc>(i,j,0);

			if (valS < minS) {
				minS = valS;
			}

			if (valS > maxS) {
				maxS = valS;
			}

			for (int c = 0; c < img.shape()[2]; c++) {
				float valR = decomposition.reflectance.value<Nc>(i,j,c);

				if (valR < minR) {
					minR = valR;
				}

				if (valR > maxR) {
					maxR = valR;
				}

				reconstructed.at<Nc>(i,j,c) = valS*valR;
				error.at<Nc>(i,j,c) = std::abs(reconstructed.at<Nc>(i,j,c) - img.at<Nc>(i,j,c));

				if (error.at<Nc>(i,j,c) > maxE) {
					maxE = error.at<Nc>(i,j,c);
				}
			}
		}
	}

	out << "min and max shading: min=" << minS << " max=" << maxS << Qt::endl;
	out << "min and max reflectance: min=" << minR << " max=" << maxR << Qt::endl;
	out << "max reconstruction error:" << maxE << Qt::endl;

    bool outputFiles = true;
    bool ok = true;

    #ifdef WITH_GUI
    outputFiles = options.contains("-o") or options.contains("--outputfiles");
    #endif

    if (outputFiles) {
        ok = StereoVision::IO::writeImage<float>((outDir.filePath(info.baseName() + "_reflectance") + ".stevimg" ).toStdString(), decomposition.reflectance);

        if (ok) {
            out << "\t" << "Reflectance file succesfully written to disk" << Qt::endl;
        } else {
            out << "\t" << "Failed to write reflectance file to disk" << Qt::endl;
            return 1;
        }
    }

    #ifdef WITH_GUI
	StereoVision::Gui::ArrayDisplayAdapter<float> reflectanceAdapter(&decomposition.reflectance, 0, maxS);
    reflectanceAdapter.configureOriginalChannelDisplay(channelsNames);
    QImageDisplay::ImageWindow reflectanceWindow;
    reflectanceWindow.setImage(&reflectanceAdapter);
    reflectanceWindow.setWindowTitle("Reflectance image");
    #endif

	#ifdef WITH_GUI
	StereoVision::Gui::ArrayDisplayAdapter<float> shadingAdapter(&decomposition.shading, 0, maxS);
	shadingAdapter.configureOriginalChannelDisplay(channelsNames);
	QImageDisplay::ImageWindow shadingWindow;
	shadingWindow.setImage(&shadingAdapter);
	shadingWindow.setWindowTitle("shading image");
	#endif

	#ifdef WITH_GUI
	StereoVision::Gui::ArrayDisplayAdapter<float> errorAdapter(&error, 0, maxE);
	errorAdapter.configureOriginalChannelDisplay(channelsNames);
	QImageDisplay::ImageWindow errorWindow;
	errorWindow.setImage(&errorAdapter);
	errorWindow.setWindowTitle("reconstruction error");
	#endif

    if (outputFiles) {
        ok = StereoVision::IO::writeImage<float>((outDir.filePath(info.baseName() + "_shading") + ".stevimg" ).toStdString(), decomposition.shading);

        if (ok) {
            out << "\t" << "Shading file succesfully written to disk" << Qt::endl;
        } else {
            out << "\t" << "Failed to write shading file to disk" << Qt::endl;
            return 1;
        }
	}

    #ifdef WITH_GUI
    imgWindow.show();
	reflectanceWindow.show();
	shadingWindow.show();
	errorWindow.show();
    return app.exec();
    #endif

	return 0;
}
