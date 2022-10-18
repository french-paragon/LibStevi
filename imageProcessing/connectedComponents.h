#ifndef LIBSTEVI_CONNECTEDCOMPONENTS_H
#define LIBSTEVI_CONNECTEDCOMPONENTS_H

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

#include <vector>
#include <tuple>
#include <set>
#include <map>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include "../utils/contiguity.h"
#include "../utils/indexers.h"

namespace StereoVision {
namespace ImageProcessing {

template<int nDims>
struct ConnectedComponentInfos {
	int idx;
	std::array<int, nDims> boundingBoxCornerMin;
	std::array<int, nDims> boundingBoxCornerMax;
};

template<int nDims, Contiguity::generalContiguity cont = Contiguity::allDimsCanChange>
std::tuple< Multidim::Array<int, nDims>, std::vector<ConnectedComponentInfos<nDims>> >
connectedComponents(Multidim::Array<bool, nDims> const& mask) {

	constexpr auto Nc = Multidim::AccessCheck::Nocheck;

	int possiblesMultiShifts = 1;

	for (int i = 0; i < nDims; i++) {
		possiblesMultiShifts *= 3;
	}

	using IdxBlock = typename Multidim::Array<bool,nDims>::IndexBlock;
	using ShpBlock = typename Multidim::Array<bool,nDims>::ShapeBlock;

	ShpBlock shape = mask.shape();
	ShpBlock allThreeShape;

	for (int i = 0 ; i < nDims; i++) {
		allThreeShape[i] = 3;
	}

	IdxBlock currentId;
	currentId.setZero();

	int next_cluster_id = 1;
	Multidim::Array<int, nDims> clusters(shape);

	int nPossibleCollision = nDims -1;
	if (cont == Contiguity::allDimsCanChange) {
		nPossibleCollision = possiblesMultiShifts-1;
	}

	std::vector<int> collision_clusters;
	collision_clusters.reserve(std::min(256,nPossibleCollision));

	Indexers::GrowingSizeDisjointSetForest unionFind;

	for (int it = 0; it < mask.flatLenght(); it++) {

		bool fg = mask.template value<Nc>(currentId);

		if (!fg) {
			clusters.template at<Nc>(currentId) = 0;
			currentId.moveToNextIndex(shape);
			continue;
		}

		int cluster_id = -1;
		collision_clusters.clear();

		if (cont == Contiguity::singleDimCanChange) {
			for (int d = 0; d < nDims; d++) {

				if (currentId[d]-1 >= 0) {
					IdxBlock shifted = currentId;
					shifted[d] -= 1;

					if (clusters.template at<Nc>(shifted) > 0) {

						if (cluster_id < 0) {
							cluster_id = clusters.template at<Nc>(shifted);
						} else {
							collision_clusters.push_back(clusters.template at<Nc>(shifted));
						}
					}
				}
			}
		} else {

			IdxBlock currentShift;
			currentShift.setZero();

			for (int itd = 0; itd < possiblesMultiShifts-1; itd++) {
				currentShift.moveToNextIndex(allThreeShape);

				IdxBlock shifted = currentId;

				for (int d = 0; d < nDims; d++) {
					int shift = 0;
					if (currentShift[d] == 1) {
						shift = -1;
					}
					if (currentShift[d] == 2) {
						shift = +1;
					}
					shifted[d] += shift;
				}

				bool isBigger = false;
				bool isEqual = true;

				for (int d = nDims-1; d >= 0; d--) {
					if (shifted[d] > currentId[d]) {
						isEqual = false;
						isBigger = true;
						break;
					}
					if (shifted[d] < currentId[d]) {
						isEqual = false;
						isBigger = false;
						break;
					}
				}

				if (isEqual or isBigger) {
					continue; //do not cover the shifts that were not already treated
				}

				int cand_id = clusters.valueOrAlt(shifted,-1);

				if (cand_id > 0) {
					if (cluster_id < 0) {
						cluster_id = cand_id;
					} else {
						collision_clusters.push_back(cand_id);
					}
				}
			}

		}

		if (cluster_id <= 0) {
			cluster_id = next_cluster_id;
			unionFind.addNode(); //add a node
			next_cluster_id++;
		}

		int cluster_gId = unionFind.getGroup(cluster_id-1);

		if (collision_clusters.size() > 0) {

			std::set<int> collidedGroups;

			for (int pxId : collision_clusters) {

				int gId = unionFind.getGroup(pxId-1);
				if (gId != cluster_gId) {
					collidedGroups.insert(gId);
				}

			}

			for (int gId : collidedGroups) {
				unionFind.joinNode(gId, cluster_gId);
			}

		}

		clusters.template at<Nc>(currentId) = cluster_gId+1;

		currentId.moveToNextIndex(shape);
	}


	currentId.setZero();

	for (int it = 0; it < mask.flatLenght(); it++) {

		int clusterId = clusters.template at<Nc>(currentId);

		if (clusterId > 0) {
			int cluster_gId = unionFind.getGroup(clusterId-1);
			clusters.template at<Nc>(currentId) = cluster_gId+1;
		} else {
			clusters.template at<Nc>(currentId) = 0;
		}

		currentId.moveToNextIndex(shape);
	}

	std::vector<ConnectedComponentInfos<nDims>> clustersInfos;
	std::map<int, int> _clusterIdx2ClusterInfosIdx;

	currentId.setZero();

	for (int it = 0; it < mask.flatLenght(); it++) {

		int clusterId = clusters.template at<Nc>(currentId);

		if (clusterId > 0) {

			if (_clusterIdx2ClusterInfosIdx.count(clusterId) > 0) {
				int infosIdx = _clusterIdx2ClusterInfosIdx[clusterId];

				for (int d = 0; d < nDims; d++) {
					clustersInfos[infosIdx].boundingBoxCornerMin[d] = std::min(clustersInfos[infosIdx].boundingBoxCornerMin[d], currentId[d]);
					clustersInfos[infosIdx].boundingBoxCornerMax[d] = std::max(clustersInfos[infosIdx].boundingBoxCornerMax[d], currentId[d]);
				}
			} else {
				int infosIdx = clustersInfos.size();

				clustersInfos.push_back({clusterId, currentId, currentId});
				_clusterIdx2ClusterInfosIdx[clusterId] = infosIdx;
			}
		}

		currentId.moveToNextIndex(shape);
	}

	return {clusters, clustersInfos};

}

template<int nDims>
int clusterSize(Multidim::Array<int, nDims> const& clusters, ConnectedComponentInfos<nDims> const& clusterInfos) {

	constexpr auto Nc = Multidim::AccessCheck::Nocheck;

	using IdxBlock = typename Multidim::Array<bool,nDims>::IndexBlock;
	using ShpBlock = typename Multidim::Array<bool,nDims>::ShapeBlock;

	ShpBlock shape = clusters.shape();

	IdxBlock min = clusterInfos.boundingBoxCornerMin;
	IdxBlock max = clusterInfos.boundingBoxCornerMax;

	IdxBlock delta = max - min;
	IdxBlock blockShape = delta + 1;

	int nPixels = 1;

	for (int i = 0; i < nDims; i++) {
		nPixels *= blockShape[i];
	}

	IdxBlock currentId;
	currentId.setZero();

	int count = 0;

	for (int i = 0; i < nPixels; i++) {

		IdxBlock shiftedId = min + currentId;

		if (clusters.template value<Nc>(shiftedId) == clusterInfos.idx) {
			count++;
		}

		currentId.moveToNextIndex(blockShape);
	}

	return count;

}

template<int nDims>
std::tuple<float, float> clusterMinorAndMajorAxis(Multidim::Array<int, nDims> const& clusters, ConnectedComponentInfos<nDims> const& clusterInfos) {

	constexpr auto Nc = Multidim::AccessCheck::Nocheck;

	using IdxBlock = typename Multidim::Array<bool,nDims>::IndexBlock;
	using ShpBlock = typename Multidim::Array<bool,nDims>::ShapeBlock;

	using LstMatrix = Eigen::Matrix<float, nDims, Eigen::Dynamic>;
	using SquareMatrix = Eigen::Matrix<float, nDims, nDims>;
	using Vector = Eigen::Matrix<float, nDims, 1>;

        int cSize = clusterSize(clusters, clusterInfos);

	ShpBlock shape = clusters.shape();

	IdxBlock min = clusterInfos.boundingBoxCornerMin;
	IdxBlock max = clusterInfos.boundingBoxCornerMax;

	IdxBlock delta = max - min;
	IdxBlock blockShape = delta + 1;

	int nPixels = 1;

	for (int i = 0; i < nDims; i++) {
		nPixels *= blockShape[i];
	}

	IdxBlock currentId;

	Vector mean = Vector::Zero();
	int count = 0;

	LstMatrix values;
        values.resize(nDims, cSize);

	for (int i = 0; i < nPixels; i++) {

		IdxBlock shiftedId = min + currentId;

                if (clusters.template value<Nc>(shiftedId) == clusterInfos.idx) {
			for (int d = 0; d < nDims; d++) {
				mean[d] += shiftedId[d];
				values(d,count) = shiftedId[d];
			}

			count++;
		}

		currentId.moveToNextIndex(blockShape);
	}

	mean /= count;

	for (int i = 0; i < count; i++) {
		values.col(i) -= mean;
	}

	SquareMatrix Var = values * values.transpose();

	auto eig = Var.eigenvalues();

	std::array<float, nDims> singulars;

	for (int d = 0; d < nDims; d++) {
		singulars[d] = std::sqrt(eig[d].real());
	}

	std::sort(singulars.begin(), singulars.end());

	return {singulars[0], singulars[nDims-1]};
}

template<int nDims>
Eigen::Matrix<float, nDims, 1> clusterCentroid(Multidim::Array<int, nDims> const& clusters, ConnectedComponentInfos<nDims> const& clusterInfos) {

	constexpr auto Nc = Multidim::AccessCheck::Nocheck;

	using IdxBlock = typename Multidim::Array<bool,nDims>::IndexBlock;
	using ShpBlock = typename Multidim::Array<bool,nDims>::ShapeBlock;

        using Vector = Eigen::Matrix<float, nDims, 1>;

	ShpBlock shape = clusters.shape();

	IdxBlock min = clusterInfos.boundingBoxCornerMin;
	IdxBlock max = clusterInfos.boundingBoxCornerMax;

	IdxBlock delta = max - min;
	IdxBlock blockShape = delta + 1;

	int nPixels = 1;

	for (int i = 0; i < nDims; i++) {
		nPixels *= blockShape[i];
	}

	IdxBlock currentId;

	Vector mean = Vector::Zero();
	int count = 0;

	for (int i = 0; i < nPixels; i++) {

		IdxBlock shiftedId = min + currentId;

		if (clusters.template value<Nc>(shiftedId) == clusterInfos.idx) {
			for (int d = 0; d < nDims; d++) {
				mean[d] += shiftedId[d];
			}

			count++;
		}

		currentId.moveToNextIndex(blockShape);
	}

	mean /= count;

	return mean;
}


template<int nDims, typename T>
float clusterMeanValue(Multidim::Array<int, nDims> const& clusters,
					   Multidim::Array<T, nDims> const& values,
					   ConnectedComponentInfos<nDims> const& clusterInfos) {

	constexpr auto Nc = Multidim::AccessCheck::Nocheck;

	using IdxBlock = typename Multidim::Array<bool,nDims>::IndexBlock;
	using ShpBlock = typename Multidim::Array<bool,nDims>::ShapeBlock;

	ShpBlock shape = clusters.shape();

	if (values.shape() != shape) {
		return std::nanf("");
	}

	IdxBlock min = clusterInfos.boundingBoxCornerMin;
	IdxBlock max = clusterInfos.boundingBoxCornerMax;

	IdxBlock delta = max - min;
	IdxBlock blockShape = delta + 1;

	int nPixels = 1;

	for (int i = 0; i < nDims; i++) {
		nPixels *= blockShape[i];
	}

	IdxBlock currentId;

	float mean = 0;
	int count = 0;

	for (int i = 0; i < nPixels; i++) {

		IdxBlock shiftedId = min + currentId;

		if (clusters.template value<Nc>(shiftedId) == clusterInfos.idx) {
			mean += values.template value<Nc>(shiftedId);
			count++;
		}

		currentId.moveToNextIndex(blockShape);
	}

	mean /= count;

	return mean;
}

} // namespace ImageProcessing
} // namespace StereoVision

#endif // LIBSTEVI_CONNECTEDCOMPONENTS_H
