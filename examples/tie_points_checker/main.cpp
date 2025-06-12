/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024  Paragon<french.paragon@gmail.com>

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
#include <QFileInfo>
#include <QFile>
#include <QDir>

#include <QMap>
#include <QVector>
#include <QSet>

#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QAction>

#include <qImageDisplayWidget/imagewidget.h>
#include <qImageDisplayWidget/overlay.h>
#include "gui/arraydisplayadapter.h"

#include <optional>
#include <random>
#include <algorithm>

struct ImagePointIdx {
    qint64 imIdx;
    qint64 pointIdx;
};

bool operator< (ImagePointIdx const& id1, ImagePointIdx const& id2) {
    if (id1.imIdx == id2.imIdx) {
        return id1.pointIdx < id2.pointIdx;
    }

    return id1.imIdx < id2.imIdx;
}

class PointDisplayOverlay : public QImageDisplay::Overlay {

public:

    enum PointStatus {
        Unknown,
        Good,
        Wrong
    };

    explicit PointDisplayOverlay(QWidget *parent = nullptr) :
        QImageDisplay::Overlay(parent),
        _imgPointCoord(std::nullopt)
    {

    }

    virtual void paintItemImpl(QPainter* painter) const {
        if (_imgPointCoord.has_value()) {
            QColor color;
            switch (_pointStatus) {
            case Unknown:
                color = QColor(255,150,90);
                break;
            case Good:
                color = QColor(0,255,90);
                break;
            case Wrong:
                color = QColor(255,0,90);
                break;
            }

            drawPoint(painter, _imgPointCoord.value(), QColor(0,0,0), 7);
            drawPoint(painter, _imgPointCoord.value(), color, 5);
        }
    }

    void setImagePointCoord(QPointF const& imgPt, PointStatus status) {
        _imgPointCoord = imgPt;
        _pointStatus = status;
        Q_EMIT repaintingRequested(QRect());
    }

    void clearPointCoord() {
        _imgPointCoord = std::nullopt;
        _pointStatus = Unknown;
        Q_EMIT repaintingRequested(QRect());
    }

    void setPointStatus(PointStatus status) {
        _pointStatus = status;
        Q_EMIT repaintingRequested(QRect());
    }

protected:

    PointStatus _pointStatus;
    std::optional<QPointF> _imgPointCoord;

};

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
        out << "No input configuration file provided" << Qt::endl;
        return 1;
    }

    //GUI variables

    uint8_t blackLevel = 0;
    uint8_t whiteLevel = 255;

    QMainWindow mw;
    QWidget mainArea;

    QVBoxLayout mwLayout(&mainArea);

    QHBoxLayout imgLayout;

    QImageDisplay::ImageWidget im1Widget;
    im1Widget.setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
    QImageDisplay::ImageWidget im2Widget;
    im2Widget.setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));

    Multidim::Array<uint8_t, 3> im1;
    Multidim::Array<uint8_t, 3> im2;

    StereoVision::Gui::ArrayDisplayAdapter<uint8_t> imageAdapter1(&im1);
    StereoVision::Gui::ArrayDisplayAdapter<uint8_t> imageAdapter2(&im2);

    im1Widget.setImage(&imageAdapter1);
    im2Widget.setImage(&imageAdapter2);

    PointDisplayOverlay pointOverlayIm1;
    PointDisplayOverlay pointOverlayIm2;

    im1Widget.addOverlay(&pointOverlayIm1);
    im2Widget.addOverlay(&pointOverlayIm2);

    imgLayout.addWidget(&im1Widget);
    imgLayout.addWidget(&im2Widget);

    mwLayout.addLayout(&imgLayout);

    QHBoxLayout toolsLayout;

    QPushButton saveButton("save (ctrl+s)");
    QPushButton validButton("valid (ctrl+v)");
    QPushButton wrongButton("wrong (ctrl+x)");

    QPushButton previousButton("previous (ctrl+d)");
    QPushButton nextButton("next (ctrl+f)");

    toolsLayout.addWidget(&saveButton);
    toolsLayout.addWidget(&validButton);
    toolsLayout.addWidget(&wrongButton);
    toolsLayout.addStretch();
    toolsLayout.addWidget(&previousButton);
    toolsLayout.addWidget(&nextButton);

    mwLayout.addLayout(&toolsLayout);

    mw.setCentralWidget(&mainArea);

    QAction saveAction;
    saveAction.setShortcut(QKeySequence(Qt::CTRL + Qt::Key_S));
    QAction validAction;
    validAction.setShortcut(QKeySequence(Qt::CTRL + Qt::Key_V));
    QAction wrongAction;
    wrongAction.setShortcut(QKeySequence(Qt::CTRL + Qt::Key_X));
    QAction previousAction;
    previousAction.setShortcut(QKeySequence(Qt::CTRL + Qt::Key_D));
    QAction nextAction;
    nextAction.setShortcut(QKeySequence(Qt::CTRL + Qt::Key_F));

    mw.addAction(&saveAction);
    mw.addAction(&validAction);
    mw.addAction(&wrongAction);
    mw.addAction(&previousAction);
    mw.addAction(&nextAction);

    //Data variables
    int currentPtIdx = -1;
    QString inputDataPath = arguments[0];

    QFileInfo inputInfos(inputDataPath);

    if (!inputInfos.exists()) {
        out << "Non existant input data file" << Qt::endl;
        return 1;
    }

    QDir inDir = inputInfos.dir();

    QString name = inputInfos.baseName();

    QString goodOutPath = inDir.filePath(name + "_good.txt");
    QString badOutPath = inDir.filePath(name + "_bad.txt");

    QMap<qint64, QVector<qint64>> pointsImgs;
    QVector<qint64> pointsIdxs;
    QMap<qint64, qint64> pointsRevIdxs;
    QMap<qint64,QString> imagesFiles;
    QMap<ImagePointIdx, QPointF> pointsCoord;
    QSet<qint64> validPoints;
    QSet<qint64> wrongPoints;

    //signals and slots

    QObject::connect(&saveButton, &QPushButton::clicked, &saveAction, &QAction::trigger);
    QObject::connect(&validButton, &QPushButton::clicked, &validAction, &QAction::trigger);
    QObject::connect(&wrongButton, &QPushButton::clicked, &wrongAction, &QAction::trigger);

    QObject::connect(&previousButton, &QPushButton::clicked, &previousAction, &QAction::trigger);
    QObject::connect(&nextButton, &QPushButton::clicked, &nextAction, &QAction::trigger);

    QObject::connect(&saveAction, &QAction::triggered, [&] () {

        QFile goodFile(goodOutPath);

        if (goodFile.exists()) {
            goodFile.remove();
        }

        QFile badFile(badOutPath);

        if (badFile.exists()) {
            badFile.remove();
        }

        if (validPoints.size() > 0) {
            goodFile.open(QFile::WriteOnly);
            QTextStream fStream(&goodFile);
            for (qint64 id : validPoints) {
                fStream << id << "\n";
            }
        }

        if (wrongPoints.size() > 0) {
            badFile.open(QFile::WriteOnly);
            QTextStream fStream(&badFile);
            for (qint64 id : wrongPoints) {
                fStream << id << "\n";
            }
        }

        goodFile.close();

    });
    QObject::connect(&validAction, &QAction::triggered, [&] () {

        if (currentPtIdx >= 0 and currentPtIdx < pointsIdxs.size()) {
            qint64 ptId = pointsIdxs[currentPtIdx];
            wrongPoints.remove(ptId);
            validPoints.insert(ptId);

            pointOverlayIm1.setPointStatus(PointDisplayOverlay::Good);
            pointOverlayIm2.setPointStatus(PointDisplayOverlay::Good);
        }

    });
    QObject::connect(&wrongAction, &QAction::triggered, &wrongAction, [&] () {

        if (currentPtIdx >= 0 and currentPtIdx < pointsIdxs.size()) {
            qint64 ptId = pointsIdxs[currentPtIdx];
            validPoints.remove(ptId);
            wrongPoints.insert(ptId);

            pointOverlayIm1.setPointStatus(PointDisplayOverlay::Wrong);
            pointOverlayIm2.setPointStatus(PointDisplayOverlay::Wrong);
        }

    });

    auto move = [&] (int delta) {

        currentPtIdx = (pointsIdxs.size() + currentPtIdx+delta)%pointsIdxs.size();
        qint64 pointIdx = pointsIdxs[currentPtIdx];

        if (pointsImgs[currentPtIdx].size() != 2) {
            out << "Point pointIdx has no image pair" << Qt::endl;
            return;
        }

        qint64 im1Id = pointsImgs[pointIdx][0];
        qint64 im2Id = pointsImgs[pointIdx][1];

        QPointF ptCoord1 = pointsCoord[{im1Id, pointIdx}];
        QPointF ptCoord2 = pointsCoord[{im2Id, pointIdx}];

        PointDisplayOverlay::PointStatus status = PointDisplayOverlay::Unknown;

        if (validPoints.contains(pointIdx)) {
            status = PointDisplayOverlay::Good;
        } else if (wrongPoints.contains(pointIdx)) {
            status = PointDisplayOverlay::Wrong;
        }

        QString im1File = imagesFiles[im1Id];
        QString im2File = imagesFiles[im2Id];

        im1 = StereoVision::IO::readImage<uint8_t>(inDir.filePath(im1File).toStdString());
        im2 = StereoVision::IO::readImage<uint8_t>(inDir.filePath(im2File).toStdString());

        imageAdapter1.imageDataUpdated();
        imageAdapter2.imageDataUpdated();

        im1Widget.setTranslation(QPoint(0,0));
        im1Widget.setZoom(100);

        im2Widget.setTranslation(QPoint(0,0));
        im2Widget.setZoom(100);

        pointOverlayIm1.setImagePointCoord(ptCoord1, status);
        pointOverlayIm2.setImagePointCoord(ptCoord2, status);
    };

    QObject::connect(&previousAction, &QAction::triggered, [&] () {
        if (currentPtIdx == -1) {
            currentPtIdx = 0;
        }
        move(-1);
    });
    QObject::connect(&nextAction, &QAction::triggered, [&] () {
        move(+1);
    });

    //actually load the data

    QFile inFile(inputDataPath);

    bool ok = inFile.open(QFile::ReadOnly);

    if (!ok) {
        out << "Failed to open input file " << inputDataPath << Qt::endl;
    }

    QTextStream inFileStream(&inFile);

    while (!inFileStream.atEnd()) {
        QString lineIm = QString::fromLocal8Bit(inFile.readLine());
        QString lineMatches = QString::fromLocal8Bit(inFile.readLine());

        QStringList imData = lineIm.split(QChar(' '));
        qint64 im_id = imData[0].toLong();
        imagesFiles[im_id] = imData.last().simplified();

        QStringList matchesData = lineMatches.split(QChar(' '));

        for (int i = 0; i < matchesData.size(); i += 3) {
            qint64 pointIdx = matchesData[i+2].toLong();

            if (!pointsRevIdxs.contains(pointIdx)) {
                qint64 newPointPos = pointsIdxs.size();
                pointsIdxs.push_back(pointIdx);
                pointsRevIdxs[pointIdx] = newPointPos;
            }

            qreal xPos = matchesData[i+0].toDouble();
            qreal yPos = matchesData[i+1].toDouble();

            QPointF point(xPos, yPos);

            pointsCoord[{im_id, pointIdx}] = point;
            pointsImgs[pointIdx].push_back(im_id);
        }
    }

    std::default_random_engine rng;
    rng.seed(4269);//deterministic seed

    std::shuffle(pointsIdxs.begin(), pointsIdxs.end(), rng);

    QFile goodFile(goodOutPath);

    QFile badFile(badOutPath);

    if (goodFile.exists()) {
        bool ok = goodFile.open(QFile::ReadOnly);

        if (ok) {
            while (!goodFile.atEnd()) {
                QString line = goodFile.readLine();
                qint64 id = line.toLong();
                validPoints.insert(id);
            }
        }
    }

    if (badFile.exists()) {
        bool ok = badFile.open(QFile::ReadOnly);

        if (ok) {
            while (!badFile.atEnd()) {
                QString line = badFile.readLine();
                qint64 id = line.toLong();
                validPoints.remove(id);
                wrongPoints.insert(id);
            }
        }
    }

    mw.resize(800,600);
    mw.show();
    return app.exec();

}
