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
#include "correlation/disparity_plus_background_segmentation.h"

#include <QTextStream>
#include <QString>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDirIterator>
#include <QSet>

#include <string>
#include <chrono>
#include <optional>


using disp_t = StereoVision::Correlation::disp_t;

struct matchingResultsInfos {
	int64_t dense_match_time_ms;
	int64_t sparse_match_time_ms;
	int64_t matching_pixels;
	int64_t total_pixels;
};

template<StereoVision::Correlation::matchingFunctions matchFunc, class T_CV, class T_FV>
using DispEstimator = StereoVision::Correlation::DisparityEstimatorWithBackgroundRemoval<matchFunc, T_CV, T_FV>;


template<StereoVision::Correlation::matchingFunctions matchFunc, class T_CV, class T_FV>
std::optional<matchingResultsInfos> testMatching(QString fgRightImPath,
												 QString fgLeftImPath,
												 uint8_t searchRadius,
												 StereoVision::Correlation::searchOffset<1> const searchOffset,
												 DispEstimator<matchFunc, T_CV, T_FV> const& matcher) {

	matchingResultsInfos out;

	Multidim::Array<T_FV, 3> imgRight = StereoVision::IO::readImage<T_FV>(fgRightImPath.toStdString());
	Multidim::Array<T_FV, 3> imgLeft = StereoVision::IO::readImage<T_FV>(fgLeftImPath.toStdString());

	if (imgRight.empty() or imgLeft.empty()) {
		return std::nullopt;
	}

	Multidim::Array<T_FV, 3> fLeftFg = StereoVision::Correlation::unfold(searchRadius, searchRadius, imgLeft);
	Multidim::Array<T_FV, 3> fRightFg = StereoVision::Correlation::unfold(searchRadius, searchRadius, imgRight);

	//compute dense disparity map
	auto start = std::chrono::high_resolution_clock::now();

	auto source_bg_features = StereoVision::Correlation::getFeatureVolumeForMatchFunc<matchFunc, T_FV, Multidim::NonConstView, T_FV>(fRightFg);
	auto target_bg_features = StereoVision::Correlation::getFeatureVolumeForMatchFunc<matchFunc, T_FV, Multidim::NonConstView, T_FV>(fLeftFg);

	auto cost_volume =
			StereoVision::Correlation::featureVolume2CostVolume
			<matchFunc,
			T_FV,
			T_FV,
			StereoVision::Correlation::searchOffset<1>,
			StereoVision::Correlation::dispDirection::RightToLeft,
			T_CV>
			(target_bg_features, source_bg_features, searchOffset);

	auto fg_index =
			StereoVision::Correlation::extractSelectedIndex
			<StereoVision::Correlation::MatchingFunctionTraits<matchFunc>::extractionStrategy>
			(cost_volume);

	Multidim::Array<StereoVision::Correlation::disp_t, 2> fg_disp_dense = StereoVision::Correlation::selectedIndexToDisp(fg_index, searchOffset.lowerOffset<0>());

	auto stop = std::chrono::high_resolution_clock::now();
	//dense disparity map computed

	out.dense_match_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();

	//compute sparse disparity map
	start = std::chrono::high_resolution_clock::now();

	using OnDemandCVT = typename DispEstimator<matchFunc, T_CV, T_FV>::template OnDemandCVT<Multidim::NonConstView, Multidim::NonConstView>;
	using searchSpaceType = typename OnDemandCVT::SearchSpaceType;
	searchSpaceType searchSpace(StereoVision::Correlation::SearchSpaceBase::IgnoredDim(),
								StereoVision::Correlation::SearchSpaceBase::SearchDim(searchOffset.lowerOffset<0>(), searchOffset.upperOffset<0>()),
								StereoVision::Correlation::SearchSpaceBase::FeatureDim());

	 OnDemandCVT onDemandCv(fRightFg, fLeftFg, searchSpace);

	auto computed = matcher.computeDispAndForegroundMask(onDemandCv);

	stop = std::chrono::high_resolution_clock::now();
	//computed sparse disparity map

	Multidim::Array<disp_t, 2>const& fg_disp_sparse = computed.disp;

	out.sparse_match_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();

	//compute number of correctly matched pixels
	out.matching_pixels = 0;

	for (int i = 0; i < fg_disp_dense.shape()[0]; i++) {
		for (int j = 0; j < fg_disp_dense.shape()[1]; j++) {

			if (std::abs(fg_disp_sparse.valueUnchecked(i,j) - fg_disp_dense.valueUnchecked(i,j)) <= 1) {
				out.matching_pixels++;
			}

		}
	}

	StereoVision::IO::writeStevimg<float, disp_t, 2>("sparse_disp.stevimg", fg_disp_sparse);

	out.total_pixels = imgRight.shape()[0]*imgRight.shape()[1];

	return out;
}

QVector<QString> getNameTemplateList(QString dirPath, QString imgType) {

	QDirIterator it(dirPath, QDirIterator::NoIteratorFlags);

	QSet<QString> selectedBaseNames;

	while (it.hasNext()) {
		QFileInfo info(it.next());

		if (!info.isFile()) {
			continue;
		}

		if (info.completeSuffix() != imgType) {
			continue;
		}

		QString basename = info.baseName();

		QVector<QString> suffixes = {"_left", "_rgb", "_right"};

		for (QString suffix : suffixes) {
			if (basename.endsWith(suffix)) {
				selectedBaseNames.insert(basename.mid(0, basename.length() - suffix.length()));
			}
		}
	}

	return QVector<QString>(selectedBaseNames.begin(), selectedBaseNames.end());

}

int main(int argc, char** argv) {

	using T_F = float;
	using T_CV = float;

	constexpr StereoVision::Correlation::matchingFunctions matchFunc = StereoVision::Correlation::matchingFunctions::SAD;

	QTextStream out(stdout);

	if (argc < 3) { //no input image
		out << "No input image and background dir provided" << Qt::endl;
		return 1;
	}

	QDir bakgroundImageFolder(argv[1]);
	QDir foregroundImagesFolder(argv[2]);

	if (!bakgroundImageFolder.exists() or !foregroundImagesFolder.exists()) { //no input image
		out << "Invalid image or background dir provided" << Qt::endl;
		return 1;
	}

	QString imgType = "png";
	uint8_t searchRadius = 5;
	int search_width = 50;
	QVector<float> relThresholds = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	int disp_tol = 2;

	QVector<QString> bgImagesBaseNames = getNameTemplateList(bakgroundImageFolder.absolutePath(), imgType);

	if (bgImagesBaseNames.size() != 1) {
		out << "Invalid background folder provided" << Qt::endl;
		return 1;
	}


	Multidim::Array<T_F, 3> bgRight = StereoVision::IO::readImage<T_F>((bakgroundImageFolder.filePath(bgImagesBaseNames[0] + "_right." + imgType)).toStdString());
	Multidim::Array<T_F, 3> bgLeft = StereoVision::IO::readImage<T_F>((bakgroundImageFolder.filePath(bgImagesBaseNames[0] + "_left." + imgType)).toStdString());

	if (bgRight.empty() or bgLeft.empty()) {
		out << "No valid background images found" << Qt::endl;
		return 1;
	}

	Multidim::Array<T_F, 3> fLeftBg = StereoVision::Correlation::unfold(searchRadius, searchRadius, bgLeft);
	Multidim::Array<T_F, 3> fRightBg = StereoVision::Correlation::unfold(searchRadius, searchRadius, bgRight);

	DispEstimator<matchFunc, T_CV, T_F> matcher(relThresholds[0], disp_tol);

	StereoVision::Correlation::searchOffset<1> search_offset(0, search_width);
	bool computed = matcher.computeBackgroundDisp(fRightBg, fLeftBg, search_offset);

	if (!computed) {
		out << "Error while computing background disp" << Qt::endl;
		return 1;
	}

	QVector<QString> fgImagesBaseNames = getNameTemplateList(foregroundImagesFolder.absolutePath(), imgType);

	out << "Cost ratio threshold" << "," << "Time dense computation [ms]" << "," << "Time sparse computation [ms]" << "," << "Proportion of correctly matched pixels" << Qt::endl;

	for (float relThreshold : relThresholds) {

		matcher.setRelThreshold(relThreshold);

		int count = 0;
		double meanDenseTime = 0;
		double meanSparseTime = 0;
		double meanPropMatched = 0;

		for (QString basename : fgImagesBaseNames) {
			QString rightImg = (foregroundImagesFolder.filePath(basename + "_right." + imgType));
			QString leftImg = (foregroundImagesFolder.filePath(basename + "_left." + imgType));

			auto infoStruct = testMatching(rightImg,
										   leftImg,
										   searchRadius,
										   search_offset,
										   matcher);

			if (!infoStruct.has_value()) {
				continue;
			}

			count++;
			meanDenseTime += infoStruct.value().dense_match_time_ms;
			meanSparseTime += infoStruct.value().sparse_match_time_ms;
			meanPropMatched += double(infoStruct.value().matching_pixels) / double(infoStruct.value().total_pixels);

			/*out << "Image " << basename << ":\n";
			out << "\t" << "Time matching dense [ms]: " << infoStruct.value().dense_match_time_ms << "\n";
			out << "\t" << "Time matching sparse [ms]: " << infoStruct.value().sparse_match_time_ms << "\n";
			out << "\t" << "# matched pixels: " << infoStruct.value().matching_pixels << "\n";
			out << "\t" << "# pixels: " << infoStruct.value().total_pixels << "\n";
			out << Qt::endl;*/
		}

		meanDenseTime /= count;
		meanSparseTime /= count;
		meanPropMatched /= count;

		out << relThreshold << "," << meanDenseTime << "," << meanSparseTime << "," << meanPropMatched << Qt::endl;
	}

	return 0;
}
