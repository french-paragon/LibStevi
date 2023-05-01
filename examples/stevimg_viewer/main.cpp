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
#include "utils/types_manipulations.h"

#include <QApplication>

#include <qImageDisplayWidget/imagewindow.h>
#include "gui/arraydisplayadapter.h"


#include <QTextStream>
#include <QString>
#include <QMap>
#include <QVector>

template<typename T, int nDim>
int displayImage(std::string const& imageName, QMap<QString, QString> const& options, QApplication & application, QTextStream & outStream) {

    static_assert (nDim == 3 or nDim == 2, "Can only process colored of grayscale images");

    T blackLevel = StereoVision::TypesManipulations::defaultBlackLevel<T>();
    T whiteLevel = StereoVision::TypesManipulations::defaultWhiteLevel<T>();

    if (options.contains("--blacklevel")) {
        bool ok;
        if (std::is_floating_point_v<T>) {
            double val = options["--blacklevel"].toDouble(&ok);
            if (ok) {
                blackLevel = val;
            }
        } else {
            long val = options["--blacklevel"].toLong(&ok);
            if (ok) {
                blackLevel = val;
            }
        }
    }

    if (options.contains("--whitelevel")) {
        bool ok;
        if (std::is_floating_point_v<T>) {
            double val = options["--whitelevel"].toDouble(&ok);
            if (ok) {
                blackLevel = val;
            }
        } else {
            long val = options["--whitelevel"].toLong(&ok);
            if (ok) {
                blackLevel = val;
            }
        }
    }

    Multidim::Array<T, nDim> img = StereoVision::IO::readStevimg<T, nDim>(imageName);

    if (img.empty()) {
        outStream << "impossible to read image: " << QString(imageName.c_str()) << Qt::endl;
        return 1;
    } else {
        outStream << "Read image: " << QString(imageName.c_str()) << Qt::endl;
        outStream << "Image shape: " << img.shape()[0] << "x" << img.shape()[1] << "x" <<  img.shape()[2] << Qt::endl;
    }

    Multidim::Array<T, 3>* coloredImg = nullptr;
    Multidim::Array<T, 2>* grayImg = nullptr;

    if (nDim == 3) {
        coloredImg = reinterpret_cast<Multidim::Array<T, 3>*>(&img);
    } else if (nDim == 2) {
        grayImg = reinterpret_cast<Multidim::Array<T, 2>*>(&img);
    }

    StereoVision::Gui::ArrayDisplayAdapter<T> cimgAdapter(coloredImg, blackLevel, whiteLevel);
    StereoVision::Gui::GrayscaleArrayDisplayAdapter<T> gimgAdapter(grayImg, blackLevel, whiteLevel);

    if (options.contains("--channels")) {
        QVector<QString> channelsNames = options.value("--channels").split(",").toVector();
        cimgAdapter.configureOriginalChannelDisplay(channelsNames);
    }

    QImageDisplay::ImageWindow imgWindow;

    if (nDim == 3) {
        imgWindow.setImage(&cimgAdapter);
    } else if (nDim == 2) {
        imgWindow.setImage(&gimgAdapter);
    }

    if (options.contains("--title")) {
        imgWindow.setWindowTitle(options.value("--title"));
    } else {
        imgWindow.setWindowTitle("Stevimg viewer");
    }

    imgWindow.show();
    return application.exec();

}

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

    //8 bit integer
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<int8_t, 2>(inFileName)) {
        return displayImage<int8_t,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<int8_t, 3>(inFileName)) {
        return displayImage<int8_t,3>(inFileName, options, app, out);
    }

    if (StereoVision::IO::stevImgFileMatchTypeAndDim<uint8_t, 2>(inFileName)) {
        return displayImage<uint8_t,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<uint8_t, 3>(inFileName)) {
        return displayImage<uint8_t,3>(inFileName, options, app, out);
    }

    //16 bit integer
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<int16_t, 2>(inFileName)) {
        return displayImage<int16_t,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<int16_t, 3>(inFileName)) {
        return displayImage<int16_t,3>(inFileName, options, app, out);
    }

    if (StereoVision::IO::stevImgFileMatchTypeAndDim<uint16_t, 2>(inFileName)) {
        return displayImage<uint16_t,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<uint16_t, 3>(inFileName)) {
        return displayImage<uint16_t,3>(inFileName, options, app, out);
    }

    //32 bit integer
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<int32_t, 2>(inFileName)) {
        return displayImage<int32_t,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<int32_t, 3>(inFileName)) {
        return displayImage<int32_t,3>(inFileName, options, app, out);
    }

    if (StereoVision::IO::stevImgFileMatchTypeAndDim<uint32_t, 2>(inFileName)) {
        return displayImage<uint32_t,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<uint32_t, 3>(inFileName)) {
        return displayImage<uint32_t,3>(inFileName, options, app, out);
    }

    //64 bit integer
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<int64_t, 2>(inFileName)) {
        return displayImage<int64_t,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<int64_t, 3>(inFileName)) {
        return displayImage<int64_t,3>(inFileName, options, app, out);
    }

    if (StereoVision::IO::stevImgFileMatchTypeAndDim<uint64_t, 2>(inFileName)) {
        return displayImage<uint64_t,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<uint64_t, 3>(inFileName)) {
        return displayImage<uint64_t,3>(inFileName, options, app, out);
    }

    //floating points
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<float, 2>(inFileName)) {
        return displayImage<float,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<float, 3>(inFileName)) {
        return displayImage<float,3>(inFileName, options, app, out);
    }

    //double precision floating points
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<double, 2>(inFileName)) {
        return displayImage<double,2>(inFileName, options, app, out);
    }
    if (StereoVision::IO::stevImgFileMatchTypeAndDim<double, 3>(inFileName)) {
        return displayImage<double,3>(inFileName, options, app, out);
    }

    out << "Input image does not match expected type or shape for a displayable image!" << Qt::endl;
    out << "The image must have at most 3 dimensions and be a numerical array!" << Qt::endl;
    return 1;
}
