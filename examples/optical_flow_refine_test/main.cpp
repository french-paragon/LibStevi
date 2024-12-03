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
	double md;
	double rmsd;
};

struct CostBasedRefined {
	Metrics parabola;
	Metrics equiangular;
};

struct ImageBasedRefined {
	Metrics rook;
	Metrics queen;
};

struct Results {
	Metrics raw;
	CostBasedRefined isotropic;
	CostBasedRefined anisotropic;
	Metrics paraboloid;
	ImageBasedRefined featuresSplit;
	ImageBasedRefined featuresSymmetric;
	Metrics splines;

	double minDispX;
	double maxDispX;
	double minDispY;
	double maxDispY;
};

Multidim::Array<bool,2> computeInliersMask(Multidim::Array<StereoVision::Correlation::disp_t,3> const& rawDisp,
										   Multidim::Array<float,3> const& gt) {

	using count_t = uint8_t;

	Multidim::Array<count_t,2> input(rawDisp.shape()[0], rawDisp.shape()[1]);

	for (int i = 0; i < input.shape()[0]; i++) {
		for (int j = 0; j < input.shape()[1]; j++) {

			auto rawDi = rawDisp.valueUnchecked(i,j,0);
			auto rawDj = rawDisp.valueUnchecked(i,j,1);
			auto gtDi = gt.valueUnchecked(i,j,1);
			auto gtDj = gt.valueUnchecked(i,j,0);


			if (std::abs(rawDi - gtDi) >= 1 or
					!std::isfinite(gtDi)) {
				input.atUnchecked(i,j) = 0;
				continue;
			}


			if (std::abs(rawDj - gtDj) >= 1 or
					!std::isfinite(gtDj)) {
				input.atUnchecked(i,j) = 0;
				continue;
			}

			input.atUnchecked(i,j) = 1;
		}
	}

	constexpr int nRadius = 2;
	constexpr int threshold = 6;

	auto padding = StereoVision::ImageProcessing::Convolution::PaddingInfos(nRadius);
	auto axisDefinition = StereoVision::ImageProcessing::Convolution::MovingWindowAxis(padding);
	auto filter = StereoVision::ImageProcessing::Convolution::constantFilter<count_t>(1,nRadius,axisDefinition,axisDefinition);
	filter.setPaddingConstant(0);

	Multidim::Array<count_t,2> count = filter.convolve(input);

	Multidim::Array<bool,2> mask(input.shape());

	for (int i = 0; i < count.shape()[0]; i++) {
		for (int j = 0; j < count.shape()[1]; j++) {
			bool countOk = (count.valueUnchecked(i,j) >= threshold);
			bool inputOk = input.valueUnchecked(i,j) > 0;
			mask.atUnchecked(i,j) = countOk and inputOk;
		}
	}

	return mask;

}

template<typename T>
Metrics compareWithGroundTruth(Multidim::Array<T,3> const& disp,
							   Multidim::Array<bool,2> const& inliers,
							   Multidim::Array<T,3> const& gt) {

	double md = 0;
	double rmsd = 0;

	std::vector<double> err;
	err.reserve(disp.shape()[0]*disp.shape()[1]);

	for (int i = 0; i < disp.shape()[0]; i++) {
		for (int j = 0; j < disp.shape()[1]; j++) {

			if (!inliers.valueUnchecked(i,j)) {
				continue; //do not include outliers in a test about subpixels refinement
			}

			T disp_true_x = gt.valueUnchecked(i,j,0);
			T disp_true_y = gt.valueUnchecked(i,j,1);

			double error_x = disp.valueUnchecked(i,j,1) - disp_true_x;
			double error_y = disp.valueUnchecked(i,j,0) - disp_true_y; //libstevi has vertical coord first

			double error = std::sqrt(error_x*error_x + error_y*error_y);

			err.push_back(error);
		}
	}

	std::sort(err.begin(), err.end()); //sort to sum smaller floating points numbers first for numerical accuracy

	for (double & error : err) {
		md += error;
		rmsd += error*error;
	}

	md /= err.size();
	rmsd /= err.size();
	rmsd = std::sqrt(rmsd);

	return {md, rmsd};
}

int global_argc;
char** global_argv;

template<StereoVision::Correlation::matchingFunctions matchFunc>
std::optional<Results> getResultsWMatchFunc(QString Img0Path,
											QString Img1Path,
											QString flowPath,
											int radius,
											int downscaleFactor,
											bool interactive) {

	using MatchFuncTraits = StereoVision::Correlation::MatchingFunctionTraits<matchFunc>;

	using T_FV = float;
	using T_CV = float;

	Multidim::Array<T_FV, 3> img0 = StereoVision::IO::readImage<T_FV>(Img0Path.toStdString());
	Multidim::Array<T_FV, 3> img1 = StereoVision::IO::readImage<T_FV>(Img1Path.toStdString());

	if (img0.empty() or img1.empty()) {
		return std::nullopt;
	}

	Multidim::Array<T_FV, 3> gtFlow = StereoVision::IO::readImage<T_FV>(flowPath.toStdString());

	if (gtFlow.empty()) {
		return std::nullopt;
	}

	if (downscaleFactor > 1) {
		StereoVision::Interpolation::DownSampleWindows downSampleWindow(downscaleFactor);
		img0 = StereoVision::Interpolation::averagePoolingDownsample(img0, downSampleWindow);
		img1 = StereoVision::Interpolation::averagePoolingDownsample(img1, downSampleWindow);
		gtFlow = StereoVision::Interpolation::averagePoolingDownsample(gtFlow, downSampleWindow);

		auto flowShape = gtFlow.shape();

		#pragma omp parallel for
		for (int i = 0; i < flowShape[0]; i++) {
			for (int j = 0; j < flowShape[1]; j++) {
				gtFlow.atUnchecked(i,j,0) /= downscaleFactor;
				gtFlow.atUnchecked(i,j,1) /= downscaleFactor;
			}
		}
	}

	for (int d = 0; d < 2; d++) {
		if (img0.shape()[d] != img1.shape()[d] or
				img0.shape()[d] != gtFlow.shape()[d]) {
			return std::nullopt;
		}
	}

	Multidim::Array<T_FV, 3> img0_features = StereoVision::Correlation::unfold<T_FV, T_FV>(radius,radius,img0);
	Multidim::Array<T_FV, 3> img1_features = StereoVision::Correlation::unfold<T_FV, T_FV>(radius,radius,img1);

	double minDispX = img0.shape()[1];
	double maxDispX = -img0.shape()[1];
	double minDispY = img0.shape()[0];
	double maxDispY = -img0.shape()[0];

	for (int i = 0; i < img0.shape()[0]; i++) {
		for (int j = 0; j < img0.shape()[1]; j++) {

			double candX = gtFlow.valueUnchecked(i,j,0);
			double candY = gtFlow.valueUnchecked(i,j,1);

			if (std::isfinite(candX) and candX <= img0.shape()[1] and candX >= -img0.shape()[1]) { //remove obviously wrong values

				if (candX > maxDispX) {
					maxDispX = candX;
				}

				if (candX < minDispX) {
					minDispX = candX;
				}
			}

			if (std::isfinite(candY) and candY <= img0.shape()[0] and candY >= -img0.shape()[0]) { //remove obviously wrong values

				if (candY > maxDispY) {
					maxDispY = candY;
				}

				if (candY < minDispY) {
					minDispY = candY;
				}
			}

		}
	}

	StereoVision::Correlation::searchOffset<2> searchRange(static_cast<int>(std::floor(minDispY)-1),
														   static_cast<int>(std::ceil(maxDispY)+1),
														   static_cast<int>(std::floor(minDispX)-1),
														   static_cast<int>(std::ceil(maxDispX)+1));

	constexpr StereoVision::Correlation::dispDirection direction = StereoVision::Correlation::dispDirection::LeftToRight;

	Multidim::Array<T_CV,4> cost_volume =
			StereoVision::Correlation::featureVolume2CostVolume
			<matchFunc,
			T_FV,
			T_FV,
			decltype(searchRange),
			direction,
			T_CV>
			(img0, img1, searchRange);

	Multidim::Array<StereoVision::Correlation::disp_t,3> selectedIdxs =
			StereoVision::Correlation::extractSelected2dIndex<MatchFuncTraits::extractionStrategy>(cost_volume);
	Multidim::Array<StereoVision::Correlation::disp_t,3> rawDisp =
			StereoVision::Correlation::selected2dIndexToDisp(
				selectedIdxs,
				searchRange
				);
	auto truncatedCostVolume = StereoVision::Correlation::truncatedBidirectionaCostVolume(cost_volume, rawDisp, 1,1);

	constexpr StereoVision::Correlation::InterpolationKernel parabolaKernel = StereoVision::Correlation::InterpolationKernel::Parabola;
	constexpr StereoVision::Correlation::InterpolationKernel equiangularKernel = StereoVision::Correlation::InterpolationKernel::Equiangular;

	constexpr StereoVision::Correlation::IsotropyHypothesis isotropic = StereoVision::Correlation::IsotropyHypothesis::Isotropic;
	constexpr StereoVision::Correlation::IsotropyHypothesis anisotropic = StereoVision::Correlation::IsotropyHypothesis::Anisotropic;

	constexpr StereoVision::Contiguity::bidimensionalContiguity rook = StereoVision::Contiguity::Rook;
	constexpr StereoVision::Contiguity::bidimensionalContiguity queen = StereoVision::Contiguity::Queen;

	Multidim::Array<T_FV, 3>  img0_features_zeromean;
	Multidim::Array<T_FV, 3>  img1_features_zeromean;

	if (StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::ZeroMean) {
		Multidim::Array<T_FV, 2> mean_img0 = StereoVision::Correlation::channelsMean<T_FV, T_FV>(img0_features);
		Multidim::Array<T_FV, 2> mean_img1 = StereoVision::Correlation::channelsMean<T_FV, T_FV>(img1_features);
		img0_features_zeromean = StereoVision::Correlation::zeromeanFeatureVolume<T_FV,T_FV,T_FV>(img0_features, mean_img0);
		img1_features_zeromean = StereoVision::Correlation::zeromeanFeatureVolume<T_FV,T_FV,T_FV>(img1_features, mean_img1);
	}

	Multidim::Array<T_FV, 3> const& img0_features_refine = (StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::ZeroMean) ? img0_features_zeromean : img0_features;
	Multidim::Array<T_FV, 3> const& img1_features_refine = (StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::ZeroMean) ? img1_features_zeromean : img1_features;;

	auto refinedParabolaIsotropic = StereoVision::Correlation::refineDisp2dCostInterpolation<parabolaKernel, isotropic>(truncatedCostVolume, rawDisp);
	auto refinedEquiangularIsotropic = StereoVision::Correlation::refineDisp2dCostInterpolation<equiangularKernel, isotropic>(truncatedCostVolume, rawDisp);

	auto refinedParabolaAnisotropic = StereoVision::Correlation::refineDisp2dCostInterpolation<parabolaKernel, anisotropic>(truncatedCostVolume, rawDisp);
	auto refinedEquiangularAnisotropic = StereoVision::Correlation::refineDisp2dCostInterpolation<equiangularKernel, anisotropic>(truncatedCostVolume, rawDisp);

	auto img0_features_processed = StereoVision::Correlation::getFeatureVolumeForMatchFunc<matchFunc>(img0_features);
	auto img1_features_processed = StereoVision::Correlation::getFeatureVolumeForMatchFunc<matchFunc>(img1_features);

	auto refinedParaboloid = StereoVision::Correlation::refineDisp2dCostPatchInterpolation<parabolaKernel>
			(truncatedCostVolume, rawDisp);

	auto refinedFeaturesSplitRook = StereoVision::Correlation::refineBarycentric2dDisp<matchFunc, rook, direction>
			(img0_features_refine, img1_features_refine, rawDisp, searchRange);

	auto refinedFeaturesSplitQueen = StereoVision::Correlation::refineBarycentric2dDisp<matchFunc, queen, direction>
			(img0_features_refine, img1_features_refine, rawDisp, searchRange);

	auto refinedFeaturesSymmetricRook = StereoVision::Correlation::refineBarycentricSymmetric2dDisp<matchFunc, rook, direction>
			(img0_features_refine, img1_features_refine, rawDisp, searchRange);

	auto refinedFeaturesSymmetricQueen = StereoVision::Correlation::refineBarycentricSymmetric2dDisp<matchFunc, queen, direction>
			(img0_features_refine, img1_features_refine, rawDisp, searchRange);

	constexpr int kernelRadius = 2;
	constexpr int nPixelsCut = 10;
	constexpr int bicubicNumerator = 1;
	constexpr int bicubicDenominator = 2;
	constexpr bool withAdditionalRefine = true;
	auto refinedBicubicSplines =
			StereoVision::Correlation::refineArbitraryInterpolation2dDisp
			<matchFunc,
			StereoVision::Interpolation::bicubicKernel<float,2,bicubicNumerator,bicubicDenominator>,
			kernelRadius,
			direction,
			withAdditionalRefine>
			(img0_features_processed,
			 img1_features_processed,
			 rawDisp,
			 nPixelsCut);

	Results ret;

	Multidim::Array<bool,2> inliers = computeInliersMask(rawDisp, gtFlow);

	ret.raw = compareWithGroundTruth<T_FV>(rawDisp.template cast<float>(), inliers, gtFlow);
	ret.isotropic.parabola = compareWithGroundTruth<T_FV>(refinedParabolaIsotropic, inliers, gtFlow);
	ret.isotropic.equiangular = compareWithGroundTruth<T_FV>(refinedEquiangularIsotropic, inliers, gtFlow);
	ret.anisotropic.parabola = compareWithGroundTruth<T_FV>(refinedParabolaAnisotropic, inliers, gtFlow);
	ret.anisotropic.equiangular = compareWithGroundTruth<T_FV>(refinedEquiangularAnisotropic, inliers, gtFlow);
	ret.paraboloid = compareWithGroundTruth<T_FV>(refinedParaboloid, inliers, gtFlow);
	ret.featuresSplit.rook = compareWithGroundTruth<T_FV>(refinedFeaturesSplitRook, inliers, gtFlow);
	ret.featuresSplit.queen = compareWithGroundTruth<T_FV>(refinedFeaturesSplitQueen, inliers, gtFlow);
	ret.featuresSymmetric.rook = compareWithGroundTruth<T_FV>(refinedFeaturesSymmetricRook, inliers, gtFlow);
	ret.featuresSymmetric.queen = compareWithGroundTruth<T_FV>(refinedFeaturesSymmetricQueen, inliers, gtFlow);
	ret.splines = compareWithGroundTruth<T_FV>(refinedBicubicSplines, inliers, gtFlow);

	ret.minDispX = minDispX;
	ret.maxDispX = maxDispX;
	ret.minDispY = minDispY;
	ret.maxDispY = maxDispY;

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
		StereoVision::Gui::ArrayDisplayAdapter<T_FV>* gtDispAdapter =
				new StereoVision::Gui::ArrayDisplayAdapter<T_FV>(&gtFlow,std::min(minDispX, minDispY),std::max(maxDispX, maxDispY),1,0,2,{1,1,0},&gtDispWindow);
		gtDispAdapter->configureOriginalChannelDisplay(QVector<QString>{"flowX", "flowY"});
		gtDispWindow.setWindowTitle("Gt flow");
		gtDispWindow.setImage(gtDispAdapter);
		gtDispWindow.show();

		QImageDisplay::ImageWindow rawDispWindow;
		StereoVision::Gui::ArrayDisplayAdapter<StereoVision::Correlation::disp_t>* rawDispAdapter =
				new StereoVision::Gui::ArrayDisplayAdapter<StereoVision::Correlation::disp_t>(&rawDisp,std::min(minDispX, minDispY),std::max(maxDispX, maxDispY),1,0,2,{0,0,1},&rawDispWindow);
		rawDispAdapter->configureOriginalChannelDisplay(QVector<QString>{"FlowY", "FlowX"});
		rawDispWindow.setWindowTitle("Raw flow");
		rawDispWindow.setImage(rawDispAdapter);
		rawDispWindow.show();

		app.exec();
	}
#endif

	return ret;

}

std::optional<Results> getResults( QString Img0,
								   QString Img1,
								   QString flowFile,
								   StereoVision::Correlation::matchingFunctions matchFunc,
								   int radius,
								   int downscaleFactor,
								   bool interactive) {

	switch(matchFunc) {
	case StereoVision::Correlation::matchingFunctions::NCC:
		return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::NCC>(Img0, Img1, flowFile, radius, downscaleFactor, interactive);
	case StereoVision::Correlation::matchingFunctions::ZNCC:
		return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::ZNCC>(Img0, Img1, flowFile, radius, downscaleFactor, interactive);
	case StereoVision::Correlation::matchingFunctions::SSD:
		return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::SSD>(Img0, Img1, flowFile, radius, downscaleFactor, interactive);
	case StereoVision::Correlation::matchingFunctions::ZSSD:
		return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::ZSSD>(Img0, Img1, flowFile, radius, downscaleFactor, interactive);
	case StereoVision::Correlation::matchingFunctions::SAD:
		return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::SAD>(Img0, Img1, flowFile, radius, downscaleFactor, interactive);
	case StereoVision::Correlation::matchingFunctions::ZSAD:
		return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::ZSAD>(Img0, Img1, flowFile, radius, downscaleFactor, interactive);
	case StereoVision::Correlation::matchingFunctions::MEDAD:
		return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::MEDAD>(Img0, Img1, flowFile, radius, downscaleFactor, interactive);
	case StereoVision::Correlation::matchingFunctions::ZMEDAD:
		return getResultsWMatchFunc<StereoVision::Correlation::matchingFunctions::ZMEDAD>(Img0, Img1, flowFile, radius, downscaleFactor, interactive);
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
		Img0Col = 1,
		Img1Col = 2,
		FlowFileCol = 3,
		ScaleCol = 4,
		NCols = 5,
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
	out << "scale" << ',';

	out << "min flow X" << ',';
	out << "max flow X" << ',';
	out << "min flow Y" << ',';
	out << "max floy Y" << ',';

	out << "md raw [px]" << ',';
	out << "md parabola isotropic [px]" << ',';
	out << "md equiangular isotropic [px]" << ',';
	out << "md parabola anisotropic [px]" << ',';
	out << "md parabola full patch [px]" << ',';
	out << "md equiangular anisotropic [px]" << ',';
	out << "md feature rook [px]" << ',';
	out << "md feature queen isotropic [px]" << ',';
	out << "md feature rook symmetric [px]" << ',';
	out << "md feature queen symmetric [px]" << ',';
	out << "md splines [px]" << ',';

	out << "rmsd raw [px]" << ',';
	out << "rmsd parabola isotropic [px]" << ',';
	out << "rmsd equiangular isotropic [px]" << ',';
	out << "rmsd parabola anisotropic [px]" << ',';
	out << "rmsd parabola full patch [px]" << ',';
	out << "rmsd equiangular anisotropic [px]" << ',';
	out << "rmsd feature rook [px]" << ',';
	out << "rmsd feature queen isotropic [px]" << ',';
	out << "rmsd feature rook symmetric [px]" << ',';
	out << "rmsd feature queen symmetric [px]" << ',';
	out << "rmsd splines [px]" << Qt::endl;


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
				QString Img0 = parameters[Img0Col];
				QString Img1 = parameters[Img1Col];
				QString flowFile = parameters[FlowFileCol];

				int scale = parameters[ScaleCol].toDouble();

				int correlationWindowWidth = 2*radius+1;

				out << name << ',';
				out << matchingFunctionName(matchFunc) << ',';
				out << correlationWindowWidth << "x" << correlationWindowWidth << ',';
				out << scale << ',';

				auto ret = getResults(Img0, Img1, flowFile, matchFunc, radius, scale, interactive);

				out << ret->minDispX << ',';
				out << ret->maxDispX << ',';
				out << ret->minDispY << ',';
				out << ret->maxDispY << ',';

				out << ret->raw.md << ',';
				out << ret->isotropic.parabola.md << ',';
				out << ret->isotropic.equiangular.md << ',';
				out << ret->anisotropic.parabola.md << ',';
				out << ret->paraboloid.md << ',';
				out << ret->anisotropic.equiangular.md << ',';
				out << ret->featuresSplit.rook.md << ',';
				out << ret->featuresSplit.queen.md << ',';
				out << ret->featuresSymmetric.rook.md << ',';
				out << ret->featuresSymmetric.queen.md << ',';
				out << ret->splines.md << ',';

				out << ret->raw.rmsd << ',';
				out << ret->isotropic.parabola.rmsd << ',';
				out << ret->isotropic.equiangular.rmsd << ',';
				out << ret->anisotropic.parabola.rmsd << ',';
				out << ret->paraboloid.rmsd << ',';
				out << ret->anisotropic.equiangular.rmsd << ',';
				out << ret->featuresSplit.rook.rmsd << ',';
				out << ret->featuresSplit.queen.rmsd << ',';
				out << ret->featuresSymmetric.rook.rmsd << ',';
				out << ret->featuresSymmetric.queen.rmsd << ',';
				out << ret->splines.rmsd << Qt::endl;

			};

		}
	}
}
