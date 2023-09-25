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

    #ifdef WITH_GUI
    QVector<QString> channelsNames = {"R", "G", "B"};
    #endif

    #ifdef WITH_GUI
    StereoVision::Gui::ArrayDisplayAdapter<float> reflectanceAdapter(&reflectance, 0, 1);
    reflectanceAdapter.configureOriginalChannelDisplay(channelsNames);
    QImageDisplay::ImageWindow reflectanceWindow;
    reflectanceWindow.setImage(&reflectanceAdapter);
    reflectanceWindow.setWindowTitle("Reflectance image");
    #endif

    #ifdef WITH_GUI
    StereoVision::Gui::ArrayDisplayAdapter<float> shadingAdapter(&shading, 0, 1);
    shadingAdapter.configureOriginalChannelDisplay(channelsNames);
    QImageDisplay::ImageWindow shadingWindow;
    shadingWindow.setImage(&shadingAdapter);
    shadingWindow.setWindowTitle("shading image");
    #endif

    Eigen::Matrix<float, 3, 1> lightDirection;
    lightDirection << 0, 1, 1;

	float lambdaNorm = 9.0;
    float lambdaDiff = 0.25;
    float lambdaDir = 0.25;
    float propEdges = 0.05;

    int nIter = 60;
    float incrTol = 1e-5;

	Multidim::Array<float, 3> img(shape_s);

	for (int i = 0; i < shape_s[0]; i++) {
		for (int j = 0; j < shape_s[1]; j++) {
			for (int c = 0; c < shape_s[2]; c++) {
				img.atUnchecked(i,j,c) = shading.atUnchecked(i,j,c)*reflectance.atUnchecked(i,j,c);
			}
		}
	}

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

    out << "Processing finished -> normals shape = " << normals.shape()[0] << " x " << normals.shape()[1] << Qt::endl;

    #ifdef WITH_GUI
    QVector<QString> normalChannelsNames = {"X", "Y", "Z"};

    Multidim::Array<float, 3> normalsDisplay(normals.shape());

    for (int i = 0; i < normals.shape()[0]; i++) {
        for (int j = 0; j < normals.shape()[1]; j++) {

            float x = normals.atUnchecked(i,j,0);
            float y = normals.atUnchecked(i,j,1);
            float z = normals.atUnchecked(i,j,2);

            normalsDisplay.atUnchecked(i,j,0) = (x+1)/2;
            normalsDisplay.atUnchecked(i,j,1) = (y+1)/2;
            normalsDisplay.atUnchecked(i,j,2) = z;
        }
    }

    StereoVision::Gui::ArrayDisplayAdapter<float> normalAdapter(&normalsDisplay, 0, 1);
    normalAdapter.configureOriginalChannelDisplay(normalChannelsNames);
    QImageDisplay::ImageWindow normalWindow;
    normalWindow.setImage(&normalAdapter);
    normalWindow.setWindowTitle("normal map");
    #endif

    #ifdef WITH_GUI
    reflectanceWindow.show();
    shadingWindow.show();
    normalWindow.show();
    return app.exec();
    #endif
    return 0;
}
