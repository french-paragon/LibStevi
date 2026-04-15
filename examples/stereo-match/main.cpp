/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2026  Paragon<french.paragon@gmail.com>

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

#include <iostream>

#include <tclap/CmdLine.h>

#include "correlation/cross_correlations.h"
#include "correlation/hierarchical.h"
#include "correlation/patchmatch.h"

#include "io/image_io.h"

#include "correlation/cost_based_refinement.h"
#include "correlation/image_based_refinement.h"
#include "correlation/on_demand_features_volume.h"

#ifdef WITH_GUI
#include <QApplication>
#include "gui/arraydisplayadapter.h"
#include "qImageDisplayWidget/imagewindow.h"
#endif

using namespace StereoVision;
using namespace StereoVision::Correlation;

using T_FV = float;
using T_CV = float;

int main(int argc, char** argv) {

    std::string img1Path;
    std::string img2Path;

    std::string gtUPath;
    std::string gtVPath;

    int searchWindowRadius;

    int left_search_searchDelta;
    int right_search_searchDelta;
    int top_search_searchDelta;
    int bottom_search_searchDelta;

    int nIter = 10;
    int nRandomSearch = 4;

    bool refineDisp = false;
    bool interactive = false;

    try {
        TCLAP::CmdLine cmd("Perform dense stereo matching between two images", '=', "0.0");

        TCLAP::UnlabeledValueArg<std::string> img1PathArg("img1Path", "Path to the source image", true, "", "local path to image file");
        TCLAP::UnlabeledValueArg<std::string> img2PathArg("img2Path", "Path to the target image", true, "", "local path to image file");

        TCLAP::ValueArg<uint> radiusArg("r", "radius", "radius of the search window", false, 3, "positive int");

        TCLAP::ValueArg<int> left_search_deltaArg("", "left-search-delta", "position of the search window", false, 0, "int");
        TCLAP::ValueArg<int> right_search_deltaArg("", "right-search-delta", "position of the search window", false, 0, "int");
        TCLAP::ValueArg<int> top_search_deltaArg("", "top-search-delta", "position of the search window", false, 0, "int");
        TCLAP::ValueArg<int> bottom_search_deltaArg("", "bottom-search-delta", "position of the search window", false, 0, "int");

        TCLAP::ValueArg<uint> nIterArg("", "n-iter", "number of patchmatch iterations", false, 10, "positive int");
        TCLAP::ValueArg<uint> nRandomSearchArg("", "n-random-search", "number of random search per patchmatch iterations", false, 4, "positive int");

        TCLAP::ValueArg<std::string> gtDispUArg("", "gt-u", "Path to the target gt u disparity", false, "", "local path to disparity map file");
        TCLAP::ValueArg<std::string> gtDispVArg("", "gt-v", "Path to the target gt v disparity", false, "", "local path to disparity map file");

        TCLAP::SwitchArg refineSwitch("","refine","Refine the estimated disparity", false);
        #ifdef WITH_GUI
        TCLAP::SwitchArg interactiveSwitch("i","interactive","Show results in interactive windows", false);
        #endif

        cmd.add(img1PathArg);
        cmd.add(img2PathArg);

        cmd.add(radiusArg);

        cmd.add(left_search_deltaArg);
        cmd.add(right_search_deltaArg);
        cmd.add(top_search_deltaArg);
        cmd.add(bottom_search_deltaArg);

        cmd.add(nIterArg);
        cmd.add(nRandomSearchArg);

        cmd.add(gtDispUArg);
        cmd.add(gtDispVArg);

        cmd.add(refineSwitch);
        #ifdef WITH_GUI
        cmd.add(interactiveSwitch);
        #endif

        cmd.parse(argc, argv);

        img1Path = img1PathArg.getValue();
        img2Path = img2PathArg.getValue();

        gtUPath = gtDispUArg.getValue();
        gtVPath = gtDispVArg.getValue();

        searchWindowRadius = radiusArg.getValue();

        left_search_searchDelta = left_search_deltaArg.getValue();
        right_search_searchDelta = right_search_deltaArg.getValue();
        top_search_searchDelta = top_search_deltaArg.getValue();
        bottom_search_searchDelta = bottom_search_deltaArg.getValue();

        refineDisp = refineSwitch.getValue();
        #ifdef WITH_GUI
        interactive = interactiveSwitch.getValue();
        #endif

    } catch (TCLAP::ArgException &e) {
        std::cerr << "Argument error: " << e.error().c_str() << " for arg " << e.argId().c_str() << std::endl;
    }

    Multidim::Array<T_FV, 3> img_source = StereoVision::IO::readImage<T_FV>(img1Path);
    Multidim::Array<T_FV, 3> img_target = StereoVision::IO::readImage<T_FV>(img2Path);

    if (img_source.empty() or img_target.empty()) {
        std::cerr << "Could not load input images" << std::endl;
        return 1;
    }

    std::cout << "Source image: size " << img_source.shape()[0] << "x" << img_source.shape()[1] << "x" <<  img_source.shape()[2] << "\n";
    std::cout << "Target image: size " << img_target.shape()[0] << "x" << img_target.shape()[1] << "x" <<  img_target.shape()[2] << "\n" << std::endl;

    if (img_source.shape()[2] != img_target.shape()[2]) {
        std::cerr << "Inconsistent number of channels, aborting!" << std::endl;
        return 1;
    }

    int sSide = 2*searchWindowRadius+1;
    int nChannels = img_source.shape()[2];
    int sSize = sSide*sSide;

    std::vector<std::array<int,3>> featuresWindow;
    featuresWindow.reserve(sSize);

    for (int i = -searchWindowRadius; i <= searchWindowRadius; i++) {
        for (int j = -searchWindowRadius; j <= searchWindowRadius; j++) {
            for (int c = 0; c < nChannels; c++) {
                featuresWindow.push_back({i,j,c});
            }
        }
    }

    constexpr matchingFunctions matchFunc = matchingFunctions::ZNCC;
    constexpr bool ZeroMean = MatchingFunctionTraits<matchFunc>::ZeroMean;
    constexpr bool Normalized = MatchingFunctionTraits<matchFunc>::Normalized;
    constexpr Multidim::ArrayDataAccessConstness constness = Multidim::ArrayDataAccessConstness::NonConstView;
    using OnDemandFeaturesT =
        OnDemandDecoratedFeaturesVolume<ZNFeaturesVolumeDecorator<ZeroMean, Normalized>,T_FV, 3, constness, 2>;

    static_assert(OnDemandFeaturesT::nOutDim == 2);

    searchOffset<2> searchRegion(top_search_searchDelta, bottom_search_searchDelta,
                                  left_search_searchDelta, right_search_searchDelta);

    OnDemandFeaturesT features_source(featuresWindow, img_source);
    OnDemandFeaturesT features_target(featuresWindow, img_target);

    Multidim::Array<disp_t, 3> out = cachelessPatchMatch<matchFunc, 2>(features_source,
                                                                       features_target,
                                                                       searchRegion,
                                                                       nIter,
                                                                       nRandomSearch,
                                                                       std::nullopt);

    std::cout << "Disparity computed" << std::endl;

    constexpr StereoVision::Correlation::dispDirection direction = StereoVision::Correlation::dispDirection::RightToLeft;
    constexpr StereoVision::Contiguity::bidimensionalContiguity queen = StereoVision::Contiguity::Queen;

    Multidim::Array<float,3> refinedDisp;

    constexpr StereoVision::Correlation::InterpolationKernel costInterpolationKernel =
        StereoVision::Correlation::InterpolationKernel::Equiangular;

    if (refineDisp) {
        /*refinedDisp = StereoVision::Correlation::refineBarycentric2dDisp<matchFunc, queen, direction>
                (features_target, features_source, out, searchRegion);*/
        using CostVolT = CachelessOnDemandImageFlowVolume<matchFunc, T_CV, OnDemandFeaturesT, OnDemandFeaturesT>;
        using SearchSpaceT = typename CostVolT::SearchSpaceType;

        SearchSpaceT searchSpace (SearchSpaceBase::SearchDim(searchRegion.template lowerOffset<0>(), searchRegion.template upperOffset<0>()),
                                 SearchSpaceBase::SearchDim(searchRegion.template lowerOffset<1>(), searchRegion.template upperOffset<1>()),
                                 SearchSpaceBase::FeatureDim());

        CachelessOnDemandImageFlowVolume<matchFunc, T_CV, OnDemandFeaturesT, OnDemandFeaturesT> onDemandCV(features_source, features_target, searchSpace);
        refinedDisp = StereoVision::Correlation::refineDisp2dCostInterpolation<costInterpolationKernel>(onDemandCV.truncatedCostVolume(out), out);
    }

    auto getDisp = [refineDisp, &refinedDisp, &out] (int i, int j, int d) -> float {
        if (refineDisp) {
            return refinedDisp.valueUnchecked(i,j,d);
        } else {
            return out.valueUnchecked(i,j,d);
        }
    };

    Multidim::Array<float, 3> gt_disp_u;
    float minDispU;
    float maxDispU;
    if (!gtUPath.empty()) {

        gt_disp_u = StereoVision::IO::readImage<float>(gtUPath);

        std::cout << "Ground truth U disp: size " << gt_disp_u.shape()[0] << "x" << gt_disp_u.shape()[1] << "x" <<  gt_disp_u.shape()[2] << "\n";

        if (out.shape()[0] != gt_disp_u.shape()[0] or out.shape()[1] != out.shape()[1]) {
            std::cerr << "Inconsistent number of channels, aborting!" << std::endl;
            return 1;
        }

        float threshold = 2;

        int bad = 0;
        int good = 0;

        for (int i = 0; i < out.shape()[0]; i++) {
            for (int j = 0; j < out.shape()[1]; j++) {
                float gt = gt_disp_u.valueUnchecked(i,j,0);

                minDispU = std::min(gt, minDispU);
                maxDispU = std::max(gt, maxDispU);

                float actual = getDisp(i,j,0);
                float error = gt - actual;
                if (std::abs(error) <= threshold) {
                    good++;
                } else {
                    bad++;
                }
            }
        }

        float propGood = 100*float(good)/(good+bad);

        std::cout << "\tProp good = " << propGood << "%" << std::endl;
    }

    Multidim::Array<float, 3> gt_disp_v;
    float minDispV;
    float maxDispV;
    if (!gtVPath.empty()) {

        gt_disp_v = StereoVision::IO::readImage<float>(gtVPath);

        std::cout << "Ground truth V disp: size " << gt_disp_v.shape()[0] << "x" << gt_disp_v.shape()[1] << "x" <<  gt_disp_v.shape()[2] << "\n";

        if (out.shape()[0] != gt_disp_v.shape()[0] or out.shape()[1] != gt_disp_v.shape()[1]) {
            std::cerr << "Inconsistent number of channels, aborting!" << std::endl;
            return 1;
        }

        float threshold = 2;

        int bad = 0;
        int good = 0;

        float mae = 0;

        for (int i = 0; i < out.shape()[0]; i++) {
            for (int j = 0; j < out.shape()[1]; j++) {
                float gt = gt_disp_v.valueUnchecked(i,j,0);

                minDispV = std::min(gt, minDispV);
                maxDispV = std::max(gt, maxDispV);

                float actual = getDisp(i,j,1);
                float error = gt - actual;
                if (std::abs(error) <= threshold) {
                    mae += std::abs(error);
                    good++;
                } else {
                    bad++;
                }
            }
        }

        float propGood = 100*float(good)/(good+bad);
        mae /= good;

        std::cout << "\tProp good = " << propGood << "%, MAE inliers = " << mae << std::endl;
    }

#ifdef  WITH_GUI
    if (interactive) {
        QApplication app(argc, argv);

        std::function<QColor(float)> gradient = [] (float prop) -> QColor {
            std::array<std::array<float,3>,5> colors =
                {std::array<float,3>{102,0,172},
                 std::array<float,3>{33,87,166},
                 std::array<float,3>{78,156,103},
                 std::array<float,3>{252,255,146},
                 std::array<float,3>{220,26,0}};

            float range = prop * colors.size();

            int colId1 = std::floor(range);
            int colId2 = std::ceil(range);

            if (colId1 < 0) {
                colId1 = 0;
            }

            if (colId2 < 0) {
                colId2 = 0;
            }

            if (colId1 >= colors.size()) {
                colId1 = colors.size()-1;
            }

            if (colId2 >= colors.size()) {
                colId2 = colors.size()-1;
            }

            float w = range - std::floor(range);

            uint8_t red = (1-w)*colors[colId1][0] + w*colors[colId2][0];
            uint8_t green = (1-w)*colors[colId1][1] + w*colors[colId2][1];
            uint8_t blue = (1-w)*colors[colId1][2] + w*colors[colId2][2];

            return QColor(red, green, blue);
        };

        T_FV whiteLevelSource = 255;
        T_FV whiteLevelTarget = 255;

        std::array<int, 3> colorChannelsSource = {0,1,2};
        std::array<int, 3> colorChannelsTarget = {0,1,2};

        if (QString::fromStdString(img1Path).endsWith(".exrlayer")) {
            whiteLevelSource = 1;
        }

        if (img_source.shape()[2] < 3) {
            colorChannelsSource = {0,0,0};
        }

        if (QString::fromStdString(img2Path).endsWith(".exrlayer")) {
            whiteLevelTarget = 1;
        }

        if (img_target.shape()[2] < 3) {
            colorChannelsTarget = {0,0,0};
        }

        QImageDisplay::ImageWindow* sourceImgWindow = new QImageDisplay::ImageWindow();
        StereoVision::Gui::ArrayDisplayAdapter<T_FV>* sourceImgAdapter = new StereoVision::Gui::ArrayDisplayAdapter<T_FV>(&img_source,0,whiteLevelSource,1,0,2,
                                                                                                                          colorChannelsSource,sourceImgWindow);
        sourceImgAdapter->configureOriginalChannelDisplay(QVector<QString>{"Red", "Green", "Blue"});
        sourceImgWindow->setWindowTitle("Source Image");
        sourceImgWindow->setImage(sourceImgAdapter);
        sourceImgWindow->show();

        QImageDisplay::ImageWindow* targetImgWindow = new QImageDisplay::ImageWindow();
        StereoVision::Gui::ArrayDisplayAdapter<T_FV>* targetImgAdapter = new StereoVision::Gui::ArrayDisplayAdapter<T_FV>(&img_target,0,whiteLevelTarget,1,0,2,
                                                                                                                         colorChannelsTarget,targetImgWindow);
        targetImgAdapter->configureOriginalChannelDisplay(QVector<QString>{"Red", "Green", "Blue"});
        targetImgWindow->setWindowTitle("Target Image");
        targetImgWindow->setImage(targetImgAdapter);
        targetImgWindow->show();

        QImageDisplay::ImageWindow* gtDispUWindow = nullptr;
        Multidim::Array<float, 2, Multidim::ArrayDataAccessConstness::ConstView> gtU;
        if (!gt_disp_u.empty()) {

            gtDispUWindow = new QImageDisplay::ImageWindow();
            gtU = gt_disp_u.sliceView(2,0);

            StereoVision::Gui::GrayscaleArrayDisplayAdapter<float, Multidim::ArrayDataAccessConstness::ConstView>* gtDispAdapter =
                new StereoVision::Gui::GrayscaleArrayDisplayAdapter<float, Multidim::ArrayDataAccessConstness::ConstView>(&gtU,minDispU-1,maxDispU+1,1,0,gtDispUWindow);
            gtDispAdapter->configureOriginalChannelDisplay("gt disp u");
            gtDispAdapter->setColorMap(gradient);
            gtDispUWindow->setWindowTitle("Gt disparity U");
            gtDispUWindow->setImage(gtDispAdapter);
            gtDispUWindow->show();
        }

        QImageDisplay::ImageWindow* gtDispVWindow = nullptr;
        Multidim::Array<float, 2, Multidim::ArrayDataAccessConstness::ConstView> gtV;
        if (!gt_disp_v.empty()) {

            gtDispVWindow = new QImageDisplay::ImageWindow();
            gtV = gt_disp_v.sliceView(2,0);

            StereoVision::Gui::GrayscaleArrayDisplayAdapter<float, Multidim::ArrayDataAccessConstness::ConstView>* gtDispAdapter =
                new StereoVision::Gui::GrayscaleArrayDisplayAdapter<float, Multidim::ArrayDataAccessConstness::ConstView>(&gtV,minDispV-1,maxDispV+1,1,0,gtDispVWindow);
            gtDispAdapter->configureOriginalChannelDisplay("gt disp v");
            gtDispAdapter->setColorMap(gradient);
            gtDispVWindow->setWindowTitle("Gt disparity V");
            gtDispVWindow->setImage(gtDispAdapter);
            gtDispVWindow->show();
        }

        QImageDisplay::ImageWindow* dispUWindow = nullptr;
        Multidim::Array<float, 2, Multidim::ArrayDataAccessConstness::ConstView> dispU;
        Multidim::Array<disp_t, 2, Multidim::ArrayDataAccessConstness::ConstView> dispURaw;
        if (!out.empty() and top_search_searchDelta != bottom_search_searchDelta) {

            dispUWindow = new QImageDisplay::ImageWindow();

            if (!refinedDisp.empty()) {
                dispU = refinedDisp.sliceView(2,0);
                StereoVision::Gui::GrayscaleArrayDisplayAdapter<float, Multidim::ArrayDataAccessConstness::ConstView>* dispAdapter =
                    new StereoVision::Gui::GrayscaleArrayDisplayAdapter<float, Multidim::ArrayDataAccessConstness::ConstView>(&dispU,minDispU-1,maxDispU+1,1,0,dispUWindow);
                dispAdapter->configureOriginalChannelDisplay("disp u");
                dispAdapter->setColorMap(gradient);
                dispUWindow->setImage(dispAdapter);
            } else {
                dispURaw = out.sliceView(2,0);
                StereoVision::Gui::GrayscaleArrayDisplayAdapter<disp_t, Multidim::ArrayDataAccessConstness::ConstView>* dispAdapter =
                    new StereoVision::Gui::GrayscaleArrayDisplayAdapter<disp_t, Multidim::ArrayDataAccessConstness::ConstView>(&dispURaw,minDispU-1,maxDispU+1,1,0,dispUWindow);
                dispAdapter->configureOriginalChannelDisplay("disp u");
                dispAdapter->setColorMap(gradient);
                dispUWindow->setImage(dispAdapter);
            }
            dispUWindow->setWindowTitle("Disparity U");
            dispUWindow->show();
        }

        QImageDisplay::ImageWindow* dispVWindow = nullptr;
        Multidim::Array<float, 2, Multidim::ArrayDataAccessConstness::ConstView> dispV;
        Multidim::Array<disp_t, 2, Multidim::ArrayDataAccessConstness::ConstView> dispVRaw;
        if (!out.empty() and left_search_searchDelta != right_search_searchDelta) {

            dispVWindow = new QImageDisplay::ImageWindow();

            if (!refinedDisp.empty()) {
                dispV = refinedDisp.sliceView(2,1);
                StereoVision::Gui::GrayscaleArrayDisplayAdapter<float, Multidim::ArrayDataAccessConstness::ConstView>* dispAdapter =
                    new StereoVision::Gui::GrayscaleArrayDisplayAdapter<float, Multidim::ArrayDataAccessConstness::ConstView>(&dispV,minDispV-1,maxDispV+1,1,0,dispVWindow);
                dispAdapter->configureOriginalChannelDisplay("disp v");
                dispAdapter->setColorMap(gradient);
                dispVWindow->setImage(dispAdapter);
            } else {
                dispVRaw = out.sliceView(2,1);
                StereoVision::Gui::GrayscaleArrayDisplayAdapter<disp_t, Multidim::ArrayDataAccessConstness::ConstView>* dispAdapter =
                    new StereoVision::Gui::GrayscaleArrayDisplayAdapter<disp_t, Multidim::ArrayDataAccessConstness::ConstView>(&dispVRaw,minDispV-1,maxDispV+1,1,0,dispVWindow);
                dispAdapter->configureOriginalChannelDisplay("disp v");
                dispAdapter->setColorMap(gradient);
                dispVWindow->setImage(dispAdapter);
            }
            dispVWindow->setWindowTitle("Disparity V");
            dispVWindow->show();
        }

        app.exec();

#define DelWindow(ptr) if (ptr != nullptr) {delete ptr; ptr = nullptr;}
        DelWindow(sourceImgWindow);
        DelWindow(targetImgWindow);
        DelWindow(gtDispUWindow);
        DelWindow(gtDispVWindow);
        DelWindow(dispUWindow);
        DelWindow(dispVWindow);
    }
#endif

    return 0;

}
