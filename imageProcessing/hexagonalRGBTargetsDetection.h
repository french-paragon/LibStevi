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

#include "../io/image_io.h"

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

	std::array<bool, 5> dotsPositives;
	std::array<Eigen::Vector2f, 5> dotsPositions;

};

template<typename T>
Eigen::Vector2f clusterBlurryCentroid(int clusterId,
									  Multidim::Array<T, 3> const&  img,
									  Multidim::Array<int, 2> const& clusters,
									  std::vector<ConnectedComponentInfos<2>> const& clustersInfos,
									  int clusterDilationRadius = 3) {

	constexpr auto Nc = Multidim::AccessCheck::Nocheck;

	ConnectedComponentInfos<2> clusterInfos = clustersInfos[clusterId];

	int minExtClusteriCoord = clusterInfos.boundingBoxCornerMin[0] - clusterDilationRadius;

	if (minExtClusteriCoord < 0) {
		minExtClusteriCoord = 0;
	}

	int maxExtClusteriCoord = clusterInfos.boundingBoxCornerMax[0] + clusterDilationRadius;

	if (maxExtClusteriCoord >= img.shape()[0]) {
		maxExtClusteriCoord = img.shape()[0]-1;
	}

	int minExtClusterjCoord = clusterInfos.boundingBoxCornerMin[1] - clusterDilationRadius;

	if (minExtClusterjCoord < 0) {
		minExtClusterjCoord = 0;
	}

	int maxExtClusterjCoord = clusterInfos.boundingBoxCornerMax[1] + clusterDilationRadius;

	if (maxExtClusterjCoord >= img.shape()[1]) {
		maxExtClusterjCoord = img.shape()[1]-1;
	}

	int height = maxExtClusteriCoord - minExtClusteriCoord + 1;
	int width = maxExtClusterjCoord - minExtClusterjCoord + 1;


	Multidim::Array<bool, 2> extendedCluster(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			extendedCluster.at<Nc>(i,j) = (clusters.value<Nc>(i+minExtClusteriCoord,j+minExtClusterjCoord) == clusterInfos.idx);
		}
	}

	Eigen::Vector3f mean = Eigen::Vector3f::Zero();
	int count = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (extendedCluster.value<Nc>(i,j)) {
				for (int d = 0; d < 3; d++) {
					mean[d] += static_cast<float>(img.template value<Nc>(i+minExtClusteriCoord,j+minExtClusterjCoord, d));
				}
				count++;
			}
		}
	}

	mean /= count;

	Multidim::Array<bool, 2> previousECluster = extendedCluster;

	for (int it = 0; it < clusterDilationRadius; it++) {
		for (int i = 1; i < height-1; i++) {
			for (int j = 1; j < width-1; j++) {

				for (int di = -1; di <= 1; di++) {
					for (int dj = -1; dj <= 1; dj++) {

						if (di == 0 and dj == 0) {
							continue;
						}

						extendedCluster.at<Nc>(i,j) = previousECluster.value<Nc>(i,j) or previousECluster.value<Nc>(i+di,j+dj);
					}
				}
			}
		}

		previousECluster = extendedCluster;
	}

	float maxDistance2Mean = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (extendedCluster.value<Nc>(i,j)) {

				Eigen::Vector3f vec;
				for (int d = 0; d < 3; d++) {
					vec[d] = static_cast<float>(img.template value<Nc>(i+minExtClusteriCoord,j+minExtClusterjCoord, d));
				}

				float dist2mean = (mean - vec).norm();

				if (dist2mean > maxDistance2Mean) {
					maxDistance2Mean = dist2mean;
				}
			}
		}
	}

	Eigen::Vector2f weigthedMeanPos = Eigen::Vector2f::Zero();
	float weightsSum = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (extendedCluster.value<Nc>(i,j)) {

				Eigen::Vector2f pos(i+minExtClusteriCoord,j+minExtClusterjCoord);

				Eigen::Vector3f vec;
				for (int d = 0; d < 3; d++) {
					vec[d] = static_cast<float>(img.template value<Nc>(i+minExtClusteriCoord,j+minExtClusterjCoord, d));
				}

				float dist2mean = (mean - vec).norm();

				float w = 1 - (dist2mean/maxDistance2Mean);

				weigthedMeanPos += w*pos;
				weightsSum += w;
			}
		}
	}

	weigthedMeanPos /= weightsSum;

	return weigthedMeanPos;

}

template<typename T>
std::array<Eigen::Vector2f, 6> hexagoneRefinedCenter(std::array<int, 6> idxs,
													 Multidim::Array<T, 3> const&  img,
													 Multidim::Array<int, 2> const& clusters,
													 std::vector<ConnectedComponentInfos<2>> const& clustersInfos,
													 int clusterDilationRadius = 3) {

	std::array<Eigen::Vector2f, 6> refinedCentroids;

	for (int i = 0; i < 6; i++) {
		refinedCentroids[i] = clusterBlurryCentroid(idxs[i], img, clusters, clustersInfos, clusterDilationRadius);
	}

	return refinedCentroids;

}

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
												float blue_gain = 1.0,
												float hexRelResThreshold = 0.1) {

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

	IO::writeImage<uint8_t>("selected.bmp", selected);

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

		std::sort(distancesToIdxs.begin(), distancesToIdxs.end());

		float opposite_dist = std::get<0>(distancesToIdxs[5]);
		int opposite_id = std::get<1>(distancesToIdxs[5]);

		if (opposite_dist > relMaxHexDiameter*std::max(imshape[0], imshape[1])) {
			//hexagone is too large
			continue;
		}

		//Test if the points fit an ellipse well
		Eigen::Vector2f center = Eigen::Vector2f::Zero();

		for (int i = 0; i < 6; i++) {
			int id = std::get<1>(distancesToIdxs[i]);
			center += centroids[id];
		}

		center /= 6;

		Eigen::Matrix<float,6,3> A = Eigen::Matrix<float,6,3>::Zero();

		for (int i = 0; i < 6; i++) {
			int id = std::get<1>(distancesToIdxs[i]);
			Eigen::Vector2f coord0 = centroids[id] - center;
			A(i,0) = coord0[0]*coord0[0];
			A(i,1) = coord0[0]*coord0[1];
			A(i,2) = coord0[1]*coord0[1];
		}

		Eigen::Matrix<float,6,1> radiuses =  Eigen::Matrix<float,6,1>::Ones();

		Eigen::Vector3f params = Optimization::leastSquares(A, radiuses);

		Eigen::Matrix<float,6,1> residuals = A * params - radiuses;

		float maxRes = residuals.cwiseAbs().maxCoeff();

		if (maxRes > hexRelResThreshold) { //misaligned with an ellipse
			continue;
		}

		if (params[2] < (params[1]/2)*(params[1]/2)) { //not an ellipse
			continue;
		}
		//hexagone detected.

		std::array<int, 6> idxs;
		std::array<int, 6> clusts_idxs;
		for (int i = 0; i < 6; i++) {
			int id = std::get<1>(distancesToIdxs[i]);
			idxs[i] = id;
			clusts_idxs[i] = selectedClusters[id];
		}

		//detect the color of the clusters
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

		std::array<Eigen::Vector2f, 6> refinedCentroids = hexagoneRefinedCenter<T>(clusts_idxs, img, clusters, clustersInfos);

		// orient the target

		std::array<float, 6> angles;
		for (int i = 0; i < 6; i++) {

			Eigen::Vector2f coord0 = centroids[idxs[i]] - center;

			angles[i] = std::atan2(coord0.x(), coord0.y()); //inverted trigonometric direction, to match image coordinates.
		}

		float delta = angles[mainId];

		for (int i = 0; i < 6; i++) {

			angles[i] -= delta;

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

		hexPos.posRefDot = refinedCentroids[hexIdxs[0]];

		hexPos.dotsPositions[0] = refinedCentroids[hexIdxs[1]];
		hexPos.dotsPositives[0] = nodesColors[hexIdxs[1]] == PC;

		hexPos.dotsPositions[1] = refinedCentroids[hexIdxs[2]];
		hexPos.dotsPositives[1] = nodesColors[hexIdxs[2]] == PC;

		hexPos.dotsPositions[2] = refinedCentroids[hexIdxs[3]];
		hexPos.dotsPositives[2] = nodesColors[hexIdxs[3]] == PC;

		hexPos.dotsPositions[3] = refinedCentroids[hexIdxs[4]];
		hexPos.dotsPositives[3] = nodesColors[hexIdxs[4]] == PC;

		hexPos.dotsPositions[4] = refinedCentroids[hexIdxs[5]];
		hexPos.dotsPositives[4] = nodesColors[hexIdxs[5]] == PC;

		ret.push_back(hexPos);

		for (int i = 1; i < 6; i++) {
			hexagonesUnions.joinNode(idxs[hexIdxs[i]], idxs[hexIdxs[0]]);
		}

	}

	return ret;

}

} // namespace HexRgbTarget
} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_HEXAGONALRGBTARGETSDETECTION_H
