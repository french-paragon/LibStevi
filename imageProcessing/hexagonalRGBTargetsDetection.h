#ifndef LIBSTEVI_HEXAGONALRGBTARGETSDETECTION_H
#define LIBSTEVI_HEXAGONALRGBTARGETSDETECTION_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022  Paragon<french.paragon@gmail.com>

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

#include <Eigen/Core>
#include <Eigen/Dense>

#include <MultidimArrays/MultidimArrays.h>

#include "../utils/colors.h"
#include "../utils/contiguity.h"
#include "../utils/types_manipulations.h"

#include "../imageProcessing/connectedComponents.h"

#include "../optimization/l2optimization.h"

namespace StereoVision {
namespace ImageProcessing {

/*!
 * HexRgbTargets are target with color dots on an hexagon corners. This namespace contian functions to detect and isolate these targets in images.
 */
namespace HexRgbTarget {

constexpr bool validateHexTargetColors(Color::RedGreenBlue MC = Color::Blue,
									   Color::RedGreenBlue PC = Color::Red,
									   Color::RedGreenBlue NC = Color::Green) {
	return (MC != PC) and
			(MC != NC) and
			(PC != NC);
}

struct HexTargetPosition {

	Eigen::Vector2f posRefDot;

	bool dot1positive;
	Eigen::Vector2f posDot1;

	bool dot2positive;
	Eigen::Vector2f posDot2;

	bool dot3positive;
	Eigen::Vector2f posDot3;

	bool dot4positive;
	Eigen::Vector2f posDot4;

	bool dot5positive;
	Eigen::Vector2f posDot5;

};

template<typename T,
		 Color::RedGreenBlue MC = Color::Blue,
		 Color::RedGreenBlue PC = Color::Red,
		 Color::RedGreenBlue NC = Color::Green,
		 Contiguity::bidimensionalContiguity Cgt = Contiguity::Queen>
/*!
 * \brief detectHexTargets detect hexagon rgb targets in an image.
 * \param img The (RGB) image in which the targets are searched for.
 * \param threshold_min The threshold for the minimal R G or B value under which a pixel is selected as candidate.
 * \param threshold_diff The threshold for the difference betwen the minimal and maximal R G or B value above which a pixel is selected as candidate.
 * \param minArea The minimal area of a cluster.
 * \param maxArea The maximal area of a cluster.
 * \param minor_major_axis_ratio The minimal ratio between the minor and major axis length of a cluster.
 * \param relMaxHexDiameter Relative diameter of an hexagon, given as a proportion of the largest side of the image, above which a target will not be detected.
 * \param red_gain Gain for the red channel (in color attribution for a cluster).
 * \param green_gain Gain for the green channel (in color attribution for a cluster).
 * \param blue_gain Gain for the blue channel (in color attribution for a cluster).
 * \return
 */
std::vector<HexTargetPosition> detectHexTargets(Multidim::Array<T, 3> const& img,
												T threshold_min,
												T threshold_diff,
												int minArea = 10,
												int maxArea = 800,
												float minor_major_axis_ratio = 0.6,
												float relMaxHexDiameter = 0.2,
												float red_gain = 1.0,
												float green_gain = 1.0,
												float blue_gain = 1.0) {

	static_assert (validateHexTargetColors(MC, PC, NC), "Invalid color scheme provided");

	constexpr auto Nc = Multidim::AccessCheck::Nocheck;
	using AccT = TypesManipulations::accumulation_extended_t<T>;

	auto imshape = img.shape();

	if (imshape[2] != 3) {
		return {};
	}

	Multidim::Array<T,2> RedChannel = const_cast<Multidim::Array<T, 3>*>(&img)->subView(Multidim::DimSlice(), Multidim::DimSlice(), Multidim::DimIndex(0));
	Multidim::Array<T,2> GreenChannel = const_cast<Multidim::Array<T, 3>*>(&img)->subView(Multidim::DimSlice(), Multidim::DimSlice(), Multidim::DimIndex(1));
	Multidim::Array<T,2> BlueChannel = const_cast<Multidim::Array<T, 3>*>(&img)->subView(Multidim::DimSlice(), Multidim::DimSlice(), Multidim::DimIndex(2));

	Multidim::Array<bool, 2> selected(imshape[0], imshape[1]);

	for (int i = 0; i < imshape[0]; i++) {
		for (int j = 0; j < imshape[1]; j++) {

			T minVal = std::min(img.template value<Nc>(i,j,0),std::min(img.template value<Nc>(i,j,1), img.template value<Nc>(i,j,2)));
			T maxVal = std::max(img.template value<Nc>(i,j,0),std::max(img.template value<Nc>(i,j,1), img.template value<Nc>(i,j,2)));

			AccT diff = static_cast<AccT>(maxVal) - static_cast<AccT>(minVal);

			bool is_candidate = (minVal <= threshold_min) or (diff >= threshold_diff);

			selected.at<Nc>(i,j) = is_candidate;

		}
	}

	constexpr Contiguity::generalContiguity contGeneral = (Cgt == Contiguity::Rook) ? Contiguity::singleDimCanChange : Contiguity::allDimsCanChange;

	auto [clusters, clustersInfos] = connectedComponents<2, contGeneral>(selected);

	int nClusters = clustersInfos.size();
	std::vector<int> selectedClusters;
	selectedClusters.reserve(nClusters);

	std::vector<Eigen::Vector2f> centroids;

	for (int i = 0; i < nClusters; i++) {

		int area = clusterSize(clusters, clustersInfos[i]);

		if (area < minArea or area > maxArea) {
			continue;
		}

		auto [minor_axis, major_axis] = clusterMinorAndMajorAxis(clusters, clustersInfos[i]);

		if (minor_axis < minor_major_axis_ratio*major_axis) {
			continue;
		}

		selectedClusters.push_back(i);
		centroids.push_back(clusterCentroid(clusters, clustersInfos[i]));

	}

	int nSelected = selectedClusters.size();

	std::vector<HexTargetPosition> ret;
	Indexers::FixedSizeDisjointSetForest hexagonesUnions(selectedClusters.size());

	for (int si = 0; si < nSelected; si++) {

		if (hexagonesUnions.getGroup(si) != si) { //cluster is already in an hexagone.
			continue;
		}

		Eigen::Vector2f source = centroids[si];

		std::vector<std::tuple<float, int>> distancesToIdxs;
		distancesToIdxs.reserve(nSelected);

		for (int sj = 0; sj < nSelected; sj++) {

			Eigen::Vector2f target =  centroids[sj];
			Eigen::Vector2f diff = source - target;
			float dist = diff.norm();

			if (hexagonesUnions.getGroup(sj) != sj) { //cluster is already in an hexagone.
				dist = std::numeric_limits<float>::infinity();
			}

			distancesToIdxs.emplace_back(dist, sj);
		}

		std::sort(distancesToIdxs.begin(), distancesToIdxs.end(), [] (std::tuple<float, int> const& t1, std::tuple<float, int> const& t2) {
			auto [dist1, id1] = t1;
			auto [dist2, id2] = t2;

			if (dist1 < dist2) {
				return true;
			}

			return id1 < id2;
		});

		auto [opposite_dist, opposite_id] = distancesToIdxs[5];

		if (opposite_dist > relMaxHexDiameter*std::max(imshape[0], imshape[1])) {
			//hexagone is too large
			continue;
		}

		//Test if the points fit an ellipse well
		Eigen::Vector2f center = Eigen::Vector2f::Zero();

		for (int i = 0; i < 6; i++) {
			auto [dist, id] = distancesToIdxs[i];
			center += centroids[id];
		}

		center /= 6;

		Eigen::Matrix<float,6,3> A = Eigen::Matrix<float,6,3>::Zero();

		for (int i = 0; i < 6; i++) {
			auto [dist, id] = distancesToIdxs[i];
			Eigen::Vector2f coord0 = centroids[id] - center;
			A(i,0) = coord0[0]*coord0[0];
			A(i,1) = coord0[0]*coord0[1];
			A(i,2) = coord0[1]*coord0[1];
		}

		Eigen::Matrix<float,6,1> radiuses =  Eigen::Matrix<float,6,1>::Ones();

		Eigen::Vector3f params = Optimization::leastSquares(A, radiuses);

		Eigen::Matrix<float,6,1> residuals = A * params - radiuses;

		float maxRes = residuals.cwiseAbs().maxCoeff();

		if (maxRes > 0.1) { //misaligned with an ellipse
			continue;
		}

		if (params[2] < (params[1]/2)*(params[1]/2)) {
			continue;
		}

		//detect the color of the clusters
		std::array<int, 6> idxs;
		for (int i = 0; i < 6; i++) {
			auto [dist, id] = distancesToIdxs[i];
			idxs[i] = id;
		}

		std::array<Color::RedGreenBlue,6> nodesColors;

		int countMain = 0;
		int mainId = -1;

		for (int i = 0; i < 6; i++) {

			int clusterId = selectedClusters[idxs[i]];

			float meanRed = clusterMeanValue(clusters, RedChannel, clustersInfos[clusterId]);
			float meanGreen = clusterMeanValue(clusters, GreenChannel, clustersInfos[clusterId]);
			float meanBlue = clusterMeanValue(clusters, BlueChannel, clustersInfos[clusterId]);

			Color::RedGreenBlue col;

			meanRed *= red_gain;
			meanGreen *= green_gain;
			meanBlue *= blue_gain;

			if (meanRed > meanGreen and meanRed > meanBlue) {
				col = Color::Red;
			} else if (meanGreen > meanBlue) {
				col = Color::Green;
			} else {
				col = Color::Blue;
			}

			if (col == MC) {
				countMain++;
				mainId = i;
			}

			nodesColors[i] = col;

		}

		if (countMain != 1) { //a single main colour should be present
			continue;
		}

		// orient the target

		std::array<float, 6> angles;
		for (int i = 0; i < 6; i++) {

			Eigen::Vector2f coord0 = centroids[idxs[i]] - center;

			angles[i] = std::atan2(coord0.x(), coord0.y()); //inverted trigonometric direction, to match image coordinates.
		}

		for (int i = 0; i < 6; i++) {

			angles[i] -= angles[mainId];

			if (angles[i] < 0) {
				angles[i] = 2*M_PI + angles[i];
			}
		}

		angles[mainId] = 0;


		std::array<int, 6> hexIdxs;

		for (int i = 0; i < 6; i++) {
			hexIdxs[i] = i;
		}

		std::sort(hexIdxs.begin(), hexIdxs.end(), [&angles] (int id1, int id2) {
			return angles[id1] < angles[id2];
		});

		//emplace the hexagone
		HexTargetPosition hexPos;

		hexPos.posRefDot = centroids[idxs[hexIdxs[0]]];

		hexPos.posDot1 = centroids[idxs[hexIdxs[1]]];
		hexPos.dot1positive = nodesColors[hexIdxs[1]] == PC;

		hexPos.posDot2 = centroids[idxs[hexIdxs[2]]];
		hexPos.dot2positive = nodesColors[hexIdxs[2]] == PC;

		hexPos.posDot3 = centroids[idxs[hexIdxs[3]]];
		hexPos.dot3positive = nodesColors[hexIdxs[3]] == PC;

		hexPos.posDot4 = centroids[idxs[hexIdxs[4]]];
		hexPos.dot4positive = nodesColors[hexIdxs[4]] == PC;

		hexPos.posDot5 = centroids[idxs[hexIdxs[5]]];
		hexPos.dot5positive = nodesColors[hexIdxs[5]] == PC;

		ret.push_back(hexPos);

		for (int i = 1; i < 6; i++) {
			hexagonesUnions.joinNode(idxs[i], idxs[0]);
		}

	}

	return ret;

}

} // namespace HexRgbTarget
} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_HEXAGONALRGBTARGETSDETECTION_H
