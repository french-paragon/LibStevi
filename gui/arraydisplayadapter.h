#ifndef LIBSTEVI_ARRAYDISPLAYADAPTER_H
#define LIBSTEVI_ARRAYDISPLAYADAPTER_H

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

#include <qImageDisplayWidget/imageadapter.h>

#include <QPoint>
#include <QSize>
#include <QRect>
#include <QColor>
#include <QImage>

#include <MultidimArrays/MultidimArrays.h>

#include <optional>

#include "../utils/types_manipulations.h"

namespace StereoVision {

namespace Gui {

template<typename Array_T, Multidim::ArrayDataAccessConstness viewConstness = Multidim::NonConstView>
class ArrayDisplayAdapter : public QImageDisplay::ImageAdapter
{
public:

    ArrayDisplayAdapter(Multidim::Array<Array_T, 3, viewConstness> const* array,
                        Array_T blackLevel = TypesManipulations::defaultBlackLevel<Array_T>(),
                        Array_T whiteLevel = TypesManipulations::defaultWhiteLevel<Array_T>(),
                        int xAxis = 1,
                        int yAxis = 0,
                        int channelAxis = 2,
                        std::array<int, 3> colorChannel = {0,1,2},
                        QObject* parent = nullptr) :
        QImageDisplay::ImageAdapter(parent),
        _array(array),
        _x_axis(xAxis),
        _y_axis(yAxis),
        _channel_axis(channelAxis),
        _color_channels(colorChannel),
        _black_level(blackLevel),
        _white_level(whiteLevel)
    {

    }

    QSize getImageSize() const override {
        if (_array == nullptr) {
            return QSize();
        }
        return QSize(_array->shape()[_x_axis], _array->shape()[_y_axis]);
    }

    QColor getColorAtPoint(int x, int y) const override{

        if (_array == nullptr) {
            return QColor();
        }

        std::array<int, 3> idx;
        idx[_x_axis] = x;
        idx[_y_axis] = y;

        QColor ret;

        idx[_channel_axis] = _color_channels[0];
        ret.setRed(valueToColor(_array->valueOrAlt(idx, 0)));

        idx[_channel_axis] = _color_channels[1];
        ret.setGreen(valueToColor(_array->valueOrAlt(idx, 0)));

        idx[_channel_axis] = _color_channels[2];
        ret.setBlue(valueToColor(_array->valueOrAlt(idx, 0)));

        return ret;
    }

    QVector<ChannelInfo> getOriginalChannelsInfos(QPoint const& pos) const override {

        if (!_displayOriginalChannels) {
            return QImageDisplay::ImageAdapter::getOriginalChannelsInfos(pos);
        }

        int nChannels = std::min(_channelsName.size(), _array->shape()[_channel_axis]);
        QVector<ChannelInfo> ret(nChannels);

        QString formatStr = "%1 ";

        std::array<int, 3> idx;
        idx[_x_axis] = pos.x();
        idx[_y_axis] = pos.y();

        for (int i = 0; i < nChannels; i++) {
            idx[_channel_axis] = i;
            ret[i].channelName = _channelsName[i];

            double tmp = _array->valueOrAlt(idx, 0);

            if (std::is_floating_point_v<Array_T>) {
                ret[i].channelValue = QString(formatStr).arg(tmp, 0, 'g', 3);
            } else {
                ret[i].channelValue = QString(formatStr).arg(_array->valueOrAlt(idx, 0));
            }
        }

        return ret;

    }

    inline void configureOriginalChannelDisplay( QVector<QString> const& channels) {
        _displayOriginalChannels = true;
        _channelsName = channels;
    }

    inline void clearOriginalChannelDisplay() {
        _displayOriginalChannels = false;
        _channelsName.clear();
    }

protected:

    using ComputeType = TypesManipulations::accumulation_extended_t<Array_T>;

    inline uint8_t valueToColor(Array_T const& value) const {

        if (value < _black_level) {
            return 0;
        }

        if (value >= _white_level) {
            return 255;
        }

        ComputeType transformed = (255*(static_cast<ComputeType>(value) - static_cast<ComputeType>(_black_level)))
                /(_white_level - _black_level);

        return static_cast<uint8_t>(transformed);
    }

    Multidim::Array<Array_T, 3, viewConstness> const* _array;

    std::array<int, 3> _color_channels;

    int _x_axis;
    int _y_axis;
    int _channel_axis;

    Array_T _black_level;
    Array_T _white_level;

    bool _displayOriginalChannels;
    QVector<QString> _channelsName;

};


template<typename Array_T, Multidim::ArrayDataAccessConstness viewConstness = Multidim::NonConstView>
class GrayscaleArrayDisplayAdapter : public QImageDisplay::ImageAdapter
{
public:

    GrayscaleArrayDisplayAdapter(Multidim::Array<Array_T, 2, viewConstness> const* array,
                        Array_T blackLevel = TypesManipulations::defaultBlackLevel<Array_T>(),
                        Array_T whiteLevel = TypesManipulations::defaultWhiteLevel<Array_T>(),
                        int xAxis = 1,
                        int yAxis = 0,
                        QObject* parent = nullptr) :
        QImageDisplay::ImageAdapter(parent),
        _array(array),
        _x_axis(xAxis),
        _y_axis(yAxis),
        _black_level(blackLevel),
        _white_level(whiteLevel),
        _colorMap(std::nullopt)
    {

    }

    QSize getImageSize() const override {
        if (_array == nullptr) {
            return QSize();
        }
        return QSize(_array->shape()[_x_axis], _array->shape()[_y_axis]);
    }

    QColor getColorAtPoint(int x, int y) const override{

        if (_array == nullptr) {
            return QColor();
        }

        std::array<int, 2> idx;
        idx[_x_axis] = x;
        idx[_y_axis] = y;

        if (_colorMap.has_value()) {
            return valueToColorWithColorMap(_array->valueOrAlt(idx, 0));
        }

        QColor ret;

        ret.setRed(valueToColor(_array->valueOrAlt(idx, 0)));
        ret.setGreen(valueToColor(_array->valueOrAlt(idx, 0)));
        ret.setBlue(valueToColor(_array->valueOrAlt(idx, 0)));

        return ret;
    }

    QVector<ChannelInfo> getOriginalChannelsInfos(QPoint const& pos) const override {

        if (!_displayOriginalChannel) {
            return QImageDisplay::ImageAdapter::getOriginalChannelsInfos(pos);
        }

        QVector<ChannelInfo> ret(1);

        QString formatStr = "%1 ";

        std::array<int, 2> idx;
        idx[_x_axis] = pos.x();
        idx[_y_axis] = pos.y();

        ret[0].channelName = _channelName;

        double tmp = _array->valueOrAlt(idx, 0);

        if (std::is_floating_point_v<Array_T>) {
            ret[0].channelValue = QString(formatStr).arg(tmp, 0, 'g', 3);
        } else {
            ret[0].channelValue = QString(formatStr).arg(_array->valueOrAlt(idx, 0));
        }

        return ret;

    }

    inline void configureOriginalChannelDisplay( QString const& channel) {
        _displayOriginalChannel = true;
        _channelName = channel;
    }

    inline void clearOriginalChannelDisplay() {
        _displayOriginalChannel = false;
        _channelName.clear();
    }

    inline void setColorMap(std::function<QColor(float)> const& colorMap) {
        _colorMap = colorMap;
    }

    inline void clearColorMap(std::function<QColor(float)> const& colorMap) {
        _colorMap = std::nullopt;
    }

protected:

    using ComputeType = TypesManipulations::accumulation_extended_t<Array_T>;

    inline uint8_t valueToColor(Array_T const& value) const {

        if (value < _black_level) {
            return 0;
        }

        if (value >= _white_level) {
            return 255;
        }

        ComputeType transformed = (255*(static_cast<ComputeType>(value) - static_cast<ComputeType>(_black_level)))
                /(_white_level - _black_level);

        return static_cast<uint8_t>(transformed);
    }

    inline QColor valueToColorWithColorMap(Array_T const& value) const {

        const std::function<QColor(float)>& cm = _colorMap.value();

        if (value < _black_level) {
            return cm(0);
        }

        if (value >= _white_level) {
            return cm(1);
        }

        return cm(static_cast<float>(value - _black_level)/static_cast<float>(_white_level - _black_level));
    }

    Multidim::Array<Array_T, 2, viewConstness> const* _array;

    int _x_axis;
    int _y_axis;

    Array_T _black_level;
    Array_T _white_level;

    bool _displayOriginalChannel;
    QString _channelName;

    std::optional<std::function<QColor(float)>> _colorMap;
};

} //namespace Gui

} //namespace StereoVision

#endif // LIBSTEVI_ARRAYDISPLAYADAPTER_H
