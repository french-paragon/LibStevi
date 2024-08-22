/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2023  Paragon<french.paragon@gmail.com>

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
#include "imageProcessing/shapeFromShading.h"

#ifdef WITH_GUI

#include <QApplication>

#include <qImageDisplayWidget/imagewindow.h>
#define LIBSTEREOVISION_BUILDING //get the QImageWidgets headers from the right place at build time
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

	if (arguments.size() < 2) { //no input image decomposition
		out << "No input image decomposition provided" << Qt::endl;
		return 1;
	}

	if (arguments.size() < 3) { //no input image decomposition
		out << "No input mask provided" << Qt::endl;
		return 1;
	}

    Multidim::Array<float, 3> shading = StereoVision::IO::readStevimg<float, 3>(arguments[0].toStdString());
    Multidim::Array<float, 3> reflectance = StereoVision::IO::readStevimg<float, 3>(arguments[1].toStdString());

    if (shading.empty()) {
        out << "impossible to read shading image: " << arguments[0] << Qt::endl;
        return 1;
    }

    if (reflectance.empty()) {
        out << "impossible to read reflectance image: " << arguments[1] << Qt::endl;
        return 1;
    }

    std::array<int,3> shape_s = shading.shape();
    std::array<int,3> shape_r = reflectance.shape();

    if (shape_s[0] != shape_r[0] or shape_s[1] != shape_r[1]) {
        out << "Mismatched shapes: " << shape_s[0] << 'x' << shape_s[1] << " vs " << shape_r[0] << 'x' << shape_r[1] << Qt::endl;
        return 1;
    }

    Multidim::Array<float, 3> img(shape_s);

    bool allSFinite = true;
    bool allRFinite = true;

    float maxS = 0;
    float maxR = 0;

    for (int i = 0; i < shape_s[0]; i++) {
        for (int j = 0; j < shape_s[1]; j++) {
            for (int c = 0; c < shape_s[2]; c++) {
                float s = shading.atUnchecked(i,j,c);
                float r = reflectance.atUnchecked(i,j,c);

                if (s > maxS) {
                    maxS = s;
                }

                if (r > maxR) {
                    maxR = r;
                }

                allSFinite = allSFinite and std::isfinite(s);
                allRFinite = allRFinite and std::isfinite(r);

				if (std::isfinite(s) and std::isfinite(r)) {
					img.atUnchecked(i,j,c) = s*r;
				} else {
					img.atUnchecked(i,j,c) = 0;
				}
            }
        }
    }

    if (!allSFinite) {
        std::cout << "Not all shading value finite!" << std::endl;
    }

    if (!allRFinite) {
        std::cout << "Not all reflectance value finite!" << std::endl;
    }


    #ifdef WITH_GUI
    QVector<QString> channelsNames = {"R", "G", "B"};
    #endif

    #ifdef WITH_GUI
	StereoVision::Gui::ArrayDisplayAdapter<float> reflectanceAdapter(&reflectance, 0, 17);
    reflectanceAdapter.configureOriginalChannelDisplay(channelsNames);
    QImageDisplay::ImageWindow reflectanceWindow;
    reflectanceWindow.setImage(&reflectanceAdapter);
    reflectanceWindow.setWindowTitle("Reflectance image");
    #endif

    #ifdef WITH_GUI
	StereoVision::Gui::ArrayDisplayAdapter<float> shadingAdapter(&shading, 0, 17);
    shadingAdapter.configureOriginalChannelDisplay(channelsNames);
    QImageDisplay::ImageWindow shadingWindow;
    shadingWindow.setImage(&shadingAdapter);
    shadingWindow.setWindowTitle("shading image");
    #endif

    Eigen::Matrix<float, 3, 1> lightDirection;
    lightDirection << 0.0, 1, 1;

	float lambdaNorm = 9.0;
    float lambdaDiff = 4.0;
    float lambdaDir = 2.0;
    float propEdges = 0.05;

    int nIter = 50;
    float incrTol = 1e-5;
	/*Multidim::Array<float, 3> normals = initialNormalMapEstimate(shading.sliceView(2,0), lightDirection);*/
	Multidim::Array<float, 3> normals = normalMapFromIntrinsicDecomposition(shading.sliceView(2,0),
																			img,
                                                                            lightDirection,
                                                                            lambdaNorm,
                                                                            lambdaDiff,
                                                                            lambdaDir,
                                                                            propEdges,
                                                                            nIter,
																			incrTol);

	Multidim::Array<uint8_t, 3> areaOfInterestImg = StereoVision::IO::readImage<uint8_t>(arguments[2].toStdString());

	Multidim::Array<bool, 2> areaOfInterest(areaOfInterestImg.shape()[0], areaOfInterestImg.shape()[1]);

	for (int i = 0; i < areaOfInterestImg.shape()[0]; i++) {
		for (int j = 0; j < areaOfInterestImg.shape()[1]; j++) {

			areaOfInterest.atUnchecked(i,j) = areaOfInterestImg.valueUnchecked(i,j,0) >= 128;

		}
	}

	#ifdef WITH_GUI
	StereoVision::Gui::ArrayDisplayAdapter<uint8_t> areaOfInterestAdapter(&areaOfInterestImg, 0, 255);
	QImageDisplay::ImageWindow areaOfInterestWindow;
	areaOfInterestWindow.setImage(&areaOfInterestAdapter);
	areaOfInterestWindow.setWindowTitle("Area of interest");
	#endif

	Multidim::Array<float, 3> rectifiedNormals = rectifyNormalMap(normals, areaOfInterest);

	out << "Normal map processing finished -> normals shape = " << normals.shape()[0] << " x " << normals.shape()[1] << Qt::endl;

	float maxDiff = 0.5;

	Multidim::Array<float, 2> rawHeightMap = heightFromNormalMap(rectifiedNormals, maxDiff);

	out << "Raw height map processing finished -> height shape = " << rawHeightMap.shape()[0] << " x " << rawHeightMap.shape()[1] << Qt::endl;

	Multidim::Array<float, 2> flatHeightMap = flattenHeightMapInAreaOfInterest(rawHeightMap, areaOfInterest);

	out << "Flat height map processing finished -> height shape = " << flatHeightMap.shape()[0] << " x " << flatHeightMap.shape()[1] << Qt::endl;

    #ifdef WITH_GUI
    QVector<QString> normalChannelsNames = {"X", "Y", "Z"};

	Multidim::Array<float, 3> normalsDisplay(rectifiedNormals.shape());

	for (int i = 0; i < rectifiedNormals.shape()[0]; i++) {
		for (int j = 0; j < rectifiedNormals.shape()[1]; j++) {

			float x = rectifiedNormals.atUnchecked(i,j,0);
			float y = rectifiedNormals.atUnchecked(i,j,1);
			float z = rectifiedNormals.atUnchecked(i,j,2);

            normalsDisplay.atUnchecked(i,j,0) = (x+1)/2;
            normalsDisplay.atUnchecked(i,j,1) = (y+1)/2;
            normalsDisplay.atUnchecked(i,j,2) = z;
        }
    }

	StereoVision::Gui::ArrayDisplayAdapter<float> normalAdapter(&rectifiedNormals, {-1, -1, 0}, {1,1,1});
    normalAdapter.configureOriginalChannelDisplay(normalChannelsNames);
    QImageDisplay::ImageWindow normalWindow;
    normalWindow.setImage(&normalAdapter);
    normalWindow.setWindowTitle("normal map");

	float minHeight = std::numeric_limits<float>::infinity();
	float maxHeight = -std::numeric_limits<float>::infinity();

	for (int i = 0; i < flatHeightMap.shape()[0]; i++) {
		for (int j = 0; j < flatHeightMap.shape()[1]; j++) {

			float val = flatHeightMap.valueUnchecked(i,j);

			if (val < minHeight) {
				minHeight = val;
			}

			if (val > maxHeight) {
				maxHeight = val;
			}

		}
	}

	StereoVision::Gui::GrayscaleArrayDisplayAdapter<float> heightAdapter(&flatHeightMap, minHeight, maxHeight);
	heightAdapter.configureOriginalChannelDisplay("height");
	QImageDisplay::ImageWindow heightWindow;
	heightWindow.setImage(&heightAdapter);
	heightWindow.setWindowTitle("height map");
    #endif

    #ifdef WITH_GUI
	areaOfInterestWindow.show();
    reflectanceWindow.show();
    shadingWindow.show();
    normalWindow.show();
	heightWindow.show();
    return app.exec();
    #endif
    return 0;
}
