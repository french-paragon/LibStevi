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

#include <QApplication>
#include <QTextStream>

#include <qImageDisplayWidget/imagewindow.h>
#include "gui/arraydisplayadapter.h"

int main(int argc, char** argv) {

    QApplication app(argc, argv);

    QVector<QString> arguments;
    QMap<QString, QString> options;

    for (int i = 1; i < argc; i++) {
        QString input(argv[i]);

        if (input.startsWith("-")) {
            QStringList split = input.split("=");
            if (split.size() == 1) {
                options.insert(split[0], "");
            } else {
                options.insert(split[0], split[1]);
            }

        } else {
            arguments.push_back(input);
        }
    }

    QTextStream out(stdout);

    if (arguments.size() < 1) { //no input image
        out << "No input image provided" << Qt::endl;
        return 1;
    }

    std::string inFileName = arguments[0].toStdString();

    Multidim::Array<float, 3> img = StereoVision::IO::readImage<float>(inFileName);

    if (img.shape()[2] != 1) {
        out << "Input image has more than a single channel" << Qt::endl;
        return 1;
    }

    Multidim::Array<float, 2> grayImg = img.sliceView(2,0);

    float noVal;
    if (options.contains("--noval")) {
        noVal = options.value("--noval").toFloat();
    }

    float blackLevel = std::numeric_limits<float>::infinity();
    float whiteLevel = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < grayImg.shape()[0]; i++) {
        for (int j = 0; j < grayImg.shape()[1]; j++) {

            float val = grayImg.valueUnchecked(i,j);

            if (options.contains("--noval")) {
                if (val == noVal) {
                    continue;
                }
            }

            if (val < blackLevel) {
                blackLevel = val;
            }

            if (val > whiteLevel) {
                whiteLevel = val;
            }
        }
    }

    StereoVision::Gui::GrayscaleArrayDisplayAdapter<float> gimgAdapter(&grayImg, blackLevel, whiteLevel);
    gimgAdapter.configureOriginalChannelDisplay("Values");

    QImageDisplay::ImageWindow imgWindow;
    imgWindow.setImage(&gimgAdapter);

    if (options.contains("--title")) {
        imgWindow.setWindowTitle(options.value("--title"));
    } else {
        imgWindow.setWindowTitle("Data image viewer");
    }

    imgWindow.show();
    return app.exec();

}
