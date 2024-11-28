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

#include <QFile>
#include <QTextStream>

#include <vector>
#include <optional>

#include "io/image_io.h"

#include "interpolation/downsampling.h"
#include "imageProcessing/convolutions.h"
#include "imageProcessing/standardConvolutionFilters.h"

#include "correlation/matching_costs.h"
#include "correlation/correlation_base.h"
#include "correlation/unfold.h"
#include "correlation/cost_based_refinement.h"
#include "correlation/image_based_refinement.h"

#ifdef WITH_GUI
#include <QApplication>
#include "gui/arraydisplayadapter.h"
#include "qImageDisplayWidget/imagewindow.h"
#endif

struct Metrics {
    double mae;
    double rmse;
    double snr;
};

struct Results {
    Metrics raw;
    Metrics parabola;
    Metrics equiangular;
    Metrics symmetric;
    Metrics image;
    Metrics imagePreNorm;
    Metrics predictive;
    Metrics predictivePreNorm;
    Metrics splines;

    double minDisp;
    double maxDisp;
};

template<StereoVision::Correlation::dispDirection direction>
Multidim::Array<bool,2> computeInliersMask(Multidim::Array<StereoVision::Correlation::disp_t,2> const& rawDisp,
                                           Multidim::Array<float,2> const& refinedParabola,
                                           Multidim::Array<float,2> const& refinedEquiangular,
                                           Multidim::Array<float,2> const& gt) {

    using count_t = uint8_t;

    Multidim::Array<count_t,2> input(rawDisp.shape());

    for (int i = 0; i < input.shape()[0]; i++) {
        for (int j = 0; j < input.shape()[1]; j++) {

            auto rawD = rawDisp.valueUnchecked(i,j);
            auto gtD = gt.valueUnchecked(i,j);

            if (std::abs(rawD - gtD) >= 1 or
                    !std::isfinite(gtD)) {
                input.atUnchecked(i,j) = 0;
                continue;
            }

            if (!std::isfinite(refinedParabola.valueUnchecked(i,j)) or
                    !std::isfinite(refinedEquiangular.valueUnchecked(i,j))) {
                input.atUnchecked(i,j) = 0;
                continue;
            }

            input.atUnchecked(i,j) = 1;
        }
    }

    constexpr int nRadius = 2;
    constexpr int threshold = 25;

    auto padding = StereoVision::ImageProcessing::Convolution::PaddingInfos(nRadius);
    auto axisDefinition = StereoVision::ImageProcessing::Convolution::MovingWindowAxis(padding);
    auto filter = StereoVision::ImageProcessing::Convolution::constantFilter<count_t>(1,nRadius,axisDefinition,axisDefinition);
    filter.setPaddingConstant(0);

    Multidim::Array<count_t,2> count = filter.convolve(input);

    Multidim::Array<bool,2> mask(rawDisp.shape());

    for (int i = 0; i < count.shape()[0]; i++) {
        for (int j = 0; j < count.shape()[1]; j++) {
            bool countOk = (count.valueUnchecked(i,j) >= threshold);
            bool inputOk = input.valueUnchecked(i,j) > 0;
            mask.atUnchecked(i,j) = countOk and inputOk;
        }
    }

    return mask;

}

template<typename T, StereoVision::Correlation::dispDirection direction>
Metrics compareWithGroundTruth(Multidim::Array<T,2> const& disp,
                               Multidim::Array<bool,2> const& inliers,
                               Multidim::Array<T,2> const& gt) {

    constexpr int nSubpixelBins = 40;

    double mae = 0;
    double rmse = 0;
    double snr = 0;

    std::vector<double> subPix;
    std::vector<double> err;
    std::vector<double> ae;
    Multidim::Array<double,1> subpixelErrorHistogram(nSubpixelBins+1);
    Multidim::Array<int,1> subpixelErrorCount(nSubpixelBins+1);

    for (int i = 0; i <= nSubpixelBins; i++) {
        subpixelErrorHistogram.atUnchecked(i) = 0;
        subpixelErrorCount.atUnchecked(i) = 0;
    }

    subPix.reserve(inliers.flatLenght());
    err.reserve(inliers.flatLenght());
    ae.reserve(inliers.flatLenght());

    for (int i = 0; i < disp.shape()[0]; i++) {
        for (int j = 0; j < disp.shape()[1]; j++) {

            if (!inliers.valueUnchecked(i,j)) {
                continue; //do not include outliers in a test about subpixels refinement
            }

            T disp_true = gt.valueUnchecked(i,j);

            double error = disp.valueUnchecked(i,j) - disp_true;
            double subpixelPart = disp_true - std::floor(disp_true);

            int bin = std::round(subpixelPart*nSubpixelBins);
            if (bin > nSubpixelBins) {
                bin = nSubpixelBins;
            }
            subpixelErrorHistogram.atUnchecked(bin) += error;
            subpixelErrorCount.atUnchecked(bin) += 1;

            subPix.push_back(subpixelPart);
            err.push_back(error);
            ae.push_back(std::abs(error));
        }
    }

    std::sort(ae.begin(), ae.end()); //sort to sum smaller floating points numbers first for numerical accuracy

    for (double & err : ae) {
        mae += err;
        rmse += err*err;
    }

    mae /= ae.size();
    rmse /= ae.size();
    rmse = std::sqrt(rmse);

    //average the error in the histogram
    for (int i = 0; i <= nSubpixelBins; i++) {
        subpixelErrorHistogram.atUnchecked(i) /= subpixelErrorCount.valueUnchecked(i);
    }

    auto floatingPointAmpl = [] (double v1, double v2) {return std::abs(v1) < std::abs(v2);};

    std::vector<double> sortedErr = err;
    std::sort(sortedErr.begin(), sortedErr.end(), floatingPointAmpl);

    double me = 0;
    for (double err : sortedErr) {
        me += err;
    }
    me /= sortedErr.size();

    std::vector<double> eErrors(err.size());
    std::vector<double> noise(err.size());

    for (int i = 0; i < err.size(); i++) {
        double pos = subPix[i]*nSubpixelBins;
        double expected = StereoVision::Interpolation::interpolateValue<1,double,StereoVision::Interpolation::pyramidFunction<double,1>,1>(subpixelErrorHistogram,{pos});
        double eError = expected - me;
        eErrors[i] = eError;
        noise[i] = err[i] - eError;
    }

    std::sort(eErrors.begin(), eErrors.end(), floatingPointAmpl);
    std::sort(noise.begin(), noise.end(), floatingPointAmpl);

    double sumExpError = 0;
    double sumNoise = 0;

    for (int i = 0; i < noise.size(); i++) {
        sumExpError += eErrors[i]*eErrors[i];
        sumNoise += noise[i]*noise[i];
    }

    snr = std::log10(sumExpError) - std::log10(sumNoise);
    snr *= 10; //convert to decibels

    return {mae, rmse, snr};
}

int global_argc;
char** global_argv;

template<StereoVision::Correlation::matchingFunctions matchFunc>
std::optional<Results> getResultsWMatchFunc(QString leftImg,
                                            QString rightImg,
                                            QString leftDisp,
                                            QString rightDisp,
                                            int radius,
                                            int downscaleFactor,
                                            bool interactive) {

    using MatchFuncTraits = StereoVision::Correlation::MatchingFunctionTraits<matchFunc>;

    constexpr StereoVision::Correlation::dispDirection direction = StereoVision::Correlation::dispDirection::RightToLeft;

    using T_FV = float;
    using T_CV = float;

    Multidim::Array<T_FV, 3> imgLeft = StereoVision::IO::readImage<T_FV>(leftImg.toStdString());
    Multidim::Array<T_FV, 3> imgRight = StereoVision::IO::readImage<T_FV>(rightImg.toStdString());

    if (imgRight.empty() or imgLeft.empty()) {
        return std::nullopt;
    }

    Multidim::Array<T_FV, 3> dispLeft = StereoVision::IO::readImage<T_FV>(leftDisp.toStdString());
    Multidim::Array<T_FV, 3> dispRight = StereoVision::IO::readImage<T_FV>(rightDisp.toStdString());

    if (dispRight.empty() or dispLeft.empty()) {
        return std::nullopt;
    }

    if (downscaleFactor > 1) {
        StereoVision::Interpolation::DownSampleWindows downSampleWindow(downscaleFactor);
        imgLeft = StereoVision::Interpolation::averagePoolingDownsample(imgLeft, downSampleWindow);
        imgRight = StereoVision::Interpolation::averagePoolingDownsample(imgRight, downSampleWindow);
        dispLeft = StereoVision::Interpolation::averagePoolingDownsample(dispLeft, downSampleWindow);
        dispRight = StereoVision::Interpolation::averagePoolingDownsample(dispRight, downSampleWindow);

        auto dispShape = dispLeft.shape();

        #pragma omp parallel for
        for (int i = 0; i < dispShape[0]; i++) {
            for (int j = 0; j < dispShape[1]; j++) {
                dispLeft.atUnchecked(i,j,0) /= downscaleFactor;
                dispRight.atUnchecked(i,j,0) /= downscaleFactor;
            }
        }
    }

    for (int d = 0; d < 2; d++) {
        if (imgRight.shape()[d] != imgLeft.shape()[d] or
                imgRight.shape()[d] != dispRight.shape()[d] or
                imgRight.shape()[d] != dispLeft.shape()[d]) {
            return std::nullopt;
        }
    }

	Multidim::Array<T_FV, 3> right_features = StereoVision::Correlation::unfold<T_FV,T_FV>(radius,radius,imgRight);
	Multidim::Array<T_FV, 3> left_features = StereoVision::Correlation::unfold<T_FV,T_FV>(radius,radius,imgLeft);

    StereoVision::Correlation::disp_t disp_width;

    double minDisp = imgRight.shape()[1];
    double maxDisp = 0;

    for (int i = 0; i < imgRight.shape()[0]; i++) {
        for (int j = 0; j < imgRight.shape()[1]; j++) {

            double candR = dispRight.valueUnchecked(i,j,0);

            if (std::isfinite(candR) and candR <= imgRight.shape()[1] and candR >= 0) { //remove obviously wrong values

                if (candR > maxDisp) {
                    maxDisp = candR;
                }

                if (candR < minDisp) {
                    minDisp = candR;
                }
            }

            double candL = dispLeft.valueUnchecked(i,j,0);

            if (std::isfinite(candL) and candL <= imgRight.shape()[1] and candL >= 0) { //remove obviously wrong values

                if (candL > maxDisp) {
                    maxDisp = candL;
                }

                if (candL < minDisp) {
                    minDisp = candL;
                }
            }

        }
    }

    disp_width = std::ceil(maxDisp)+3; //put a little bit of additional margin

    auto cost_volume =
            StereoVision::Correlation::featureVolume2CostVolume
            <matchFunc,
            T_FV,
            T_FV,
            decltype(disp_width),
            direction,
            T_CV>
            (left_features, right_features, disp_width);

    auto rawDisp = StereoVision::Correlation::extractSelectedIndex<MatchFuncTraits::extractionStrategy>(cost_volume);
    auto truncatedCostVolume = StereoVision::Correlation::truncatedCostVolume(cost_volume, rawDisp, 0, 0, 1);

    constexpr StereoVision::Correlation::InterpolationKernel parabolaKernel = StereoVision::Correlation::InterpolationKernel::Parabola;
    constexpr StereoVision::Correlation::InterpolationKernel equiangularKernel = StereoVision::Correlation::InterpolationKernel::Equiangular;

    auto refinedParabola = StereoVision::Correlation::refineDispCostInterpolation<parabolaKernel>(truncatedCostVolume, rawDisp);
    auto refinedEquiangular = StereoVision::Correlation::refineDispCostInterpolation<equiangularKernel>(truncatedCostVolume, rawDisp);

	Multidim::Array<T_FV, 3>  right_features_processed;
	Multidim::Array<T_FV, 3>  left_features_processed;

	Multidim::Array<T_FV, 3>  right_features_zeromean;
	Multidim::Array<T_FV, 3>  left_features_zeromean;

	if (StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::ZeroMean) {
		Multidim::Array<T_FV, 2> mean_right = StereoVision::Correlation::channelsMean<T_FV, T_FV>(right_features);
		Multidim::Array<T_FV, 2> mean_left = StereoVision::Correlation::channelsMean<T_FV, T_FV>(left_features);
		right_features_zeromean = StereoVision::Correlation::zeromeanFeatureVolume<T_FV,T_FV,T_FV>(right_features, mean_right);
		left_features_zeromean = StereoVision::Correlation::zeromeanFeatureVolume<T_FV,T_FV,T_FV>(left_features, mean_left);
	}

	right_features_processed = StereoVision::Correlation::getFeatureVolumeForMatchFunc<matchFunc>(right_features);
	left_features_processed = StereoVision::Correlation::getFeatureVolumeForMatchFunc<matchFunc>(left_features);

    bool preNormalize = false;
    auto refinedSymmetric = StereoVision::Correlation::refineCostSymmetricDisp<matchFunc, direction>
            (left_features_processed, right_features_processed, rawDisp, cost_volume);

	Multidim::Array<float,2> refinedImage;

	if (StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::ZeroMean) {
		refinedImage = StereoVision::Correlation::refineBarycentricDisp<matchFunc, direction>
					(left_features_zeromean, right_features_zeromean, rawDisp);
	} else {
		refinedImage = StereoVision::Correlation::refineBarycentricDisp<matchFunc, direction>
				(left_features, right_features, rawDisp);
	}


	Multidim::Array<float,2> refinedPredictive;

	if (StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::ZeroMean) {
		refinedPredictive = StereoVision::Correlation::refineBarycentricSymmetricDisp<matchFunc, 1, direction>
					(left_features_zeromean, right_features_zeromean, rawDisp, disp_width);
	} else {
		refinedPredictive = StereoVision::Correlation::refineBarycentricSymmetricDisp<matchFunc, 1, direction>
					(left_features, right_features, rawDisp, disp_width);
	}

	Multidim::Array<float,2> refinedImagePreNormalized;
	Multidim::Array<float,2> refinedPredictivePreNormalized;

	if (StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::Normalized) {
		refinedImagePreNormalized = StereoVision::Correlation::refineBarycentricDisp<matchFunc, direction>
            (left_features_processed, right_features_processed, rawDisp);

		refinedImagePreNormalized = StereoVision::Correlation::refineBarycentricSymmetricDisp<matchFunc, 1, direction>
            (left_features_processed, right_features_processed, rawDisp, disp_width);
	}

    constexpr int kernelRadius = 2;
    constexpr int nPixelsCut = 10;
    constexpr int bicubicNumerator = 1;
    constexpr int bicubicDenominator = 2;
    constexpr bool withAdditionalRefine = true;
    auto refinedBicubicSplines =
            StereoVision::Correlation::refineArbitraryInterpolationDisp
            <matchFunc,
            StereoVision::Interpolation::bicubicKernel<float,1,bicubicNumerator,bicubicDenominator>,
            kernelRadius,
            direction,
            withAdditionalRefine>
			(left_features,
			 right_features,
             rawDisp,
             nPixelsCut);

    Results ret;

    Multidim::Array<T_FV,2> gtLeft = dispLeft.sliceView(2,0);
    Multidim::Array<T_FV,2> gtRight = dispRight.sliceView(2,0);

    StereoVision::Correlation::condImgRef<T_FV,T_FV,direction,2> condGt(gtLeft, gtRight);

    Multidim::Array<T_FV,2> const& gtDisp = condGt.source();

    Multidim::Array<bool,2> inliers = computeInliersMask<direction>(rawDisp, refinedParabola, refinedEquiangular, gtDisp);

    ret.raw = compareWithGroundTruth<T_FV, direction>(rawDisp.template cast<float>(), inliers, gtDisp);
    ret.parabola = compareWithGroundTruth<T_FV, direction>(refinedParabola, inliers, gtDisp);
    ret.equiangular = compareWithGroundTruth<T_FV, direction>(refinedEquiangular, inliers, gtDisp);
    ret.symmetric = compareWithGroundTruth<T_FV, direction>(refinedSymmetric, inliers, gtDisp);
	ret.image = compareWithGroundTruth<T_FV, direction>(refinedImage, inliers, gtDisp);
	ret.predictive = compareWithGroundTruth<T_FV, direction>(refinedPredictive, inliers, gtDisp);
    ret.splines = compareWithGroundTruth<T_FV, direction>(refinedBicubicSplines, inliers, gtDisp);

	if (StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::Normalized) {
		ret.imagePreNorm = compareWithGroundTruth<T_FV, direction>(refinedImagePreNormalized, inliers, gtDisp);
		ret.predictivePreNorm = compareWithGroundTruth<T_FV, direction>(refinedPredictivePreNormalized, inliers, gtDisp);
	} else {
		ret.imagePreNorm = Metrics{std::nan(""),std::nan(""),std::nan("")};
		ret.predictivePreNorm = Metrics{std::nan(""),std::nan(""),std::nan("")};
	}

    ret.minDisp = minDisp;
    ret.maxDisp = maxDisp;

#ifdef WITH_GUI

    if (interactive) {
        QApplication app(global_argc, global_argv);

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

        T_FV whiteLevelLeft = 255;
        T_FV whiteLevelRight = 255;

        std::array<int, 3> colorChannelsLeft = {0,1,2};
        std::array<int, 3> colorChannelsRight = {0,1,2};

        if (leftImg.endsWith(".exrlayer")) {
            whiteLevelLeft = 1;
        }

        if (imgLeft.shape()[2] < 3) {
            colorChannelsLeft = {0,0,0};
        }

        if (rightImg.endsWith(".exrlayer")) {
            whiteLevelRight = 1;
        }

        if (imgRight.shape()[2] < 3) {
            colorChannelsRight = {0,0,0};
        }

        QImageDisplay::ImageWindow leftImgWindow;
        StereoVision::Gui::ArrayDisplayAdapter<T_FV>* leftImgAdapter = new StereoVision::Gui::ArrayDisplayAdapter<T_FV>(&imgLeft,0,whiteLevelLeft,1,0,2,colorChannelsLeft,&leftImgWindow);
        leftImgAdapter->configureOriginalChannelDisplay(QVector<QString>{"Red", "Green", "Blue"});
        leftImgWindow.setWindowTitle("Left Image");
        leftImgWindow.setImage(leftImgAdapter);
        leftImgWindow.show();

        QImageDisplay::ImageWindow rightImgWindow;
        StereoVision::Gui::ArrayDisplayAdapter<T_FV>* rightImgAdapter = new StereoVision::Gui::ArrayDisplayAdapter<T_FV>(&imgRight,0,whiteLevelRight,1,0,2,colorChannelsRight,&rightImgWindow);
        rightImgAdapter->configureOriginalChannelDisplay(QVector<QString>{"Red", "Green", "Blue"});
        rightImgWindow.setWindowTitle("Right Image");
        rightImgWindow.setImage(leftImgAdapter);
        rightImgWindow.show();

        QImageDisplay::ImageWindow gtDispWindow;
        StereoVision::Gui::GrayscaleArrayDisplayAdapter<T_FV>* gtDispAdapter =
                new StereoVision::Gui::GrayscaleArrayDisplayAdapter<T_FV>(&gtDisp,0,maxDisp,1,0,&gtDispWindow);
        gtDispAdapter->configureOriginalChannelDisplay("gt disp");
        gtDispAdapter->setColorMap(gradient);
        gtDispWindow.setWindowTitle("Gt disparity");
        gtDispWindow.setImage(gtDispAdapter);
        gtDispWindow.show();

        QImageDisplay::ImageWindow rawDispWindow;
        StereoVision::Gui::GrayscaleArrayDisplayAdapter<StereoVision::Correlation::disp_t>* rawDispAdapter =
                new StereoVision::Gui::GrayscaleArrayDisplayAdapter<StereoVision::Correlation::disp_t>(&rawDisp,0,maxDisp,1,0,&rawDispWindow);
        rawDispAdapter->configureOriginalChannelDisplay("raw disp");
        rawDispAdapter->setColorMap(gradient);
        rawDispWindow.setWindowTitle("Raw disparity");
        rawDispWindow.setImage(rawDispAdapter);
        rawDispWindow.show();

        app.exec();
    }
#endif

    return ret;

}

std::optional<Results> getResults( QString leftImg,
                                   QString rightImg,
                                   QString leftDisp,
                                   QString rightDisp,
                                   StereoVision::Correlation::matchingFunctions matchFunc,
                                   int radius,
                                   int downscaleFactor,
                                   bool interactive) {

    switch(matchFunc) {
    case StereoVision::Correlation::matchingFunctions::NCC:
        return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::NCC>(leftImg, rightImg, leftDisp, rightDisp, radius, downscaleFactor, interactive);
    case StereoVision::Correlation::matchingFunctions::ZNCC:
        return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::ZNCC>(leftImg, rightImg, leftDisp, rightDisp, radius, downscaleFactor, interactive);
    case StereoVision::Correlation::matchingFunctions::SSD:
        return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::SSD>(leftImg, rightImg, leftDisp, rightDisp, radius, downscaleFactor, interactive);
    case StereoVision::Correlation::matchingFunctions::ZSSD:
        return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::ZSSD>(leftImg, rightImg, leftDisp, rightDisp, radius, downscaleFactor, interactive);
    case StereoVision::Correlation::matchingFunctions::SAD:
        return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::SAD>(leftImg, rightImg, leftDisp, rightDisp, radius, downscaleFactor, interactive);
    case StereoVision::Correlation::matchingFunctions::ZSAD:
        return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::ZSAD>(leftImg, rightImg, leftDisp, rightDisp, radius, downscaleFactor, interactive);
    case StereoVision::Correlation::matchingFunctions::MEDAD:
        return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::MEDAD>(leftImg, rightImg, leftDisp, rightDisp, radius, downscaleFactor, interactive);
    case StereoVision::Correlation::matchingFunctions::ZMEDAD:
        return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::ZMEDAD>(leftImg, rightImg, leftDisp, rightDisp, radius, downscaleFactor, interactive);
    default:
        break;
    }

    return std::nullopt;

}

QString matchingFunctionName(StereoVision::Correlation::matchingFunctions matchFunc) {
    switch(matchFunc) {
    case StereoVision::Correlation::matchingFunctions::CC:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::CC>::Name);
    case StereoVision::Correlation::matchingFunctions::NCC:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::NCC>::Name);
    case StereoVision::Correlation::matchingFunctions::ZNCC:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::ZNCC>::Name);
    case StereoVision::Correlation::matchingFunctions::SSD:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::SSD>::Name);
    case StereoVision::Correlation::matchingFunctions::ZSSD:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::ZSSD>::Name);
    case StereoVision::Correlation::matchingFunctions::SAD:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::SAD>::Name);
    case StereoVision::Correlation::matchingFunctions::ZSAD:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::ZSAD>::Name);
    case StereoVision::Correlation::matchingFunctions::MEDAD:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::MEDAD>::Name);
    case StereoVision::Correlation::matchingFunctions::ZMEDAD:
        return QString::fromStdString(StereoVision::Correlation::MatchingFunctionTraits<StereoVision::Correlation::matchingFunctions::ZMEDAD>::Name);
    default:
        break;
    }

    return "Unknown";
}

int main(int argc, char** argv) {

    global_argc = argc;
    global_argv = argv;

    enum InFileDefinition {
        NameCol = 0,
        LeftImgCol = 1,
        RightImgCol = 2,
        LeftDispCol = 3,
        RightDispCol = 4,
        BaselineCol = 5,
        FLenCol = 6,
        ScaleCol = 7,
        NCols = 8,
    };

    std::vector<StereoVision::Correlation::matchingFunctions> matchinFunctions = {
        StereoVision::Correlation::matchingFunctions::NCC,
        StereoVision::Correlation::matchingFunctions::ZNCC,
        StereoVision::Correlation::matchingFunctions::SSD,
        StereoVision::Correlation::matchingFunctions::ZSSD,
        StereoVision::Correlation::matchingFunctions::SAD,
        StereoVision::Correlation::matchingFunctions::ZSAD,
        StereoVision::Correlation::matchingFunctions::MEDAD,
        StereoVision::Correlation::matchingFunctions::ZMEDAD
    };

    std::vector<int> radiuses = {2,3,4,5};

    QTextStream out(stdout);
    QTextStream err(stderr);

    bool interactive = false;

#ifdef WITH_GUI
    if (argc < 2) {
        err << "Wrong number of argument provided, expected to get filepath to input file list" << Qt::endl;
        return 1;
    }

    for (int i = 2; i < argc; i++) {
        if (QString(argv[i]) == "-i") {
            interactive = true;
        }
    }

#else
    if (argc != 2) {
        err << "Wrong number of argument provided, expected to get filepath to input file list" << Qt::endl;
        return 1;
    }
#endif

    QFile inList(argv[1]);

    bool status = inList.open(QFile::ReadOnly);

    if (!status) {
        err << "Could not read input configuration in:" << argv[1] << "! Aborting!" << Qt::endl;
        return 1;
    }

    out << "Image" << ',';
    out << "Cost function" << ',';
    out << "Correlation window" << ',';
    out << "Baseline[mm]" << ',';
    out << "fLen[pix]" << ',';
    out << "scale" << ',';

    out << "min disparity" << ',';
    out << "max disparity" << ',';

    out << "mae raw [px]" << ',';
    out << "mae parabola [px]" << ',';
    out << "mae equiangular [px]" << ',';
    out << "mae symmetric [px]" << ',';
    out << "mae image [px]" << ',';
    out << "mae image pre-norm [px]" << ',';
    out << "mae predictive [px]" << ',';
    out << "mae predictive pre-norm [px]" << ',';
    out << "mae bicubic [px]" << ',';

    out << "rmse raw [px]" << ',';
    out << "rmse parabola [px]" << ',';
    out << "rmse equiangular [px]" << ',';
    out << "rmse symmetric [px]" << ',';
    out << "rmse image [px]" << ',';
    out << "rmse image pre-norm [px]" << ',';
    out << "rmse predictive [px]" << ',';
    out << "rmse predictive pre-norm [px]" << ',';
    out << "rmse bicubic [px]" << ',';

    out << "snr raw [dB]" << ',';
    out << "snr parabola [dB]" << ',';
    out << "snr equiangular [dB]" << ',';
    out << "snr symmetric [dB]" << ',';
    out << "snr image [dB]" << ',';
    out << "snr image pre-norm [dB]" << ',';
    out << "snr predictive [dB]" << ',';
    out << "snr predictive pre-norm [dB]" << ',';
    out << "snr bicubic [dB]" << Qt::endl;


    QTextStream inStream(&inList);

    for (StereoVision::Correlation::matchingFunctions matchFunc : matchinFunctions) {
        for (int radius : radiuses) {

            inStream.seek(0); //restart at the beginning;

            for (QString line = inStream.readLine(); !line.isNull(); line = inStream.readLine()) {
                QStringList parameters = line.split(',');

                if (parameters.size() != InFileDefinition::NCols) {
                    continue;
                }

                QString name = parameters[NameCol];
                QString leftImg = parameters[LeftImgCol];
                QString rightImg = parameters[RightImgCol];
                QString leftDisp = parameters[LeftDispCol];
                QString rightDisp = parameters[RightDispCol];

                double baseline = parameters[BaselineCol].toDouble();
                double flen = parameters[FLenCol].toDouble();
                int scale = parameters[ScaleCol].toDouble();

                int correlationWindowWidth = 2*radius+1;

                out << name << ',';
                out << matchingFunctionName(matchFunc) << ',';
                out << correlationWindowWidth << "x" << correlationWindowWidth << ',';
                out << baseline << ',';
                out << flen << ',';
                out << scale << ',';

                auto ret = getResults(leftImg, rightImg, leftDisp, rightDisp, matchFunc, radius, scale, interactive);

                out << ret->minDisp << ',';
                out << ret->maxDisp << ',';

                out << ret->raw.mae << ',';
                out << ret->parabola.mae << ',';
                out << ret->equiangular.mae << ',';
                out << ret->symmetric.mae << ',';
                out << ret->image.mae << ',';
                out << ret->imagePreNorm.mae << ',';
                out << ret->predictive.mae << ',';
                out << ret->predictivePreNorm.mae << ',';
                out << ret->splines.mae << ',';

                out << ret->raw.rmse << ',';
                out << ret->parabola.rmse << ',';
                out << ret->equiangular.rmse << ',';
                out << ret->symmetric.rmse << ',';
                out << ret->image.rmse << ',';
                out << ret->imagePreNorm.rmse << ',';
                out << ret->predictive.rmse << ',';
                out << ret->predictivePreNorm.rmse << ',';
                out << ret->splines.rmse << ',';

                out << ret->raw.snr << ',';
                out << ret->parabola.snr << ',';
                out << ret->equiangular.snr << ',';
                out << ret->symmetric.snr << ',';
                out << ret->image.snr << ',';
                out << ret->imagePreNorm.snr << ',';
                out << ret->predictive.snr << ',';
                out << ret->predictivePreNorm.snr << ',';
                out << ret->splines.snr << Qt::endl;

            };

        }
    }
}
