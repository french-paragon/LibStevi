#ifndef INTRINSICIMAGEDECOMPOSITION_H
#define INTRINSICIMAGEDECOMPOSITION_H
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022-2023  Paragon<french.paragon@gmail.com>

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

#include <optional>

#include <MultidimArrays/MultidimArrays.h>
#include <MultidimArrays/MultidimIndexManipulators.h>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <assert.h>
#include <forward_list>
#include <iterator>
#include <limits>
#include <iostream>

#include "./colorConversions.h"
#include "./histogram.h"

#include "../correlation/matching_costs.h"
#include "../correlation/unfold.h"

#include "../utils/indexers.h"

#include "../io/image_io.h"

namespace StereoVision {
namespace ImageProcessing {

template<typename T, int nDim>
struct IntrinsicImageDecomposition {
	Multidim::Array<T, nDim> reflectance;
	Multidim::Array<T, nDim> shading;
};

template<typename T, typename ComputeType=float>
/*!
 * \brief retinexWithNonLocalTextureConstraint solve the intrisic image decomposition problem with a non local constraint on the texture
 * \param original the image to decompose
 * \return an Intrisic image decomposition structure.
 */
IntrinsicImageDecomposition<ComputeType, 3> retinexWithNonLocalTextureConstraint(Multidim::Array<T, 3> const& rgbImage,
																				 ComputeType diffThreshold,
																				 ComputeType lambdaRetinex = 1.0,
																				 ComputeType lambdaTexture = 1.0,
																				 ComputeType lambdaScaling = 1000.,
																				 ComputeType reflectanceToShadingWeight = 100.,
																				 float groupingThreshold = 0.01) {

	static_assert (std::is_floating_point_v<ComputeType>, "Compute type must be a floating point type");

	using MatrixAType = Eigen::SparseMatrix<ComputeType>;
	using VectorBType = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;
	using VectorSolType = Eigen::Matrix<ComputeType, Eigen::Dynamic, 1>;
	using SolverType = Eigen::ConjugateGradient<MatrixAType, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<ComputeType>>; //using SolverType = Eigen::BiCGSTAB<MatrixAType, Eigen::DiagonalPreconditioner<ComputeType>>;

	using fVecT = Multidim::Array<ComputeType, 1, Multidim::ConstView>;

	constexpr int channelDim = 2;
	constexpr int nColors = 3;
	constexpr int nDim = 3;

	if (rgbImage.shape()[channelDim] != nColors) {
		return IntrinsicImageDecomposition<ComputeType, nDim>{ Multidim::Array<ComputeType, nDim>(), Multidim::Array<ComputeType, nDim>() };
	}

	Multidim::Array<ComputeType, nDim> logImg = linear2logColorSpaceImg(rgbImage);

	Multidim::Array<ComputeType, nDim> rgChromaticity = normalizedIntensityRedGreenImage(logImg);

	Multidim::DimsExclusionSet<nDim> exlusionSet(channelDim);
	Multidim::IndexConverter<nDim> idxConverter(logImg.shape(), exlusionSet);

	int shadingVectorLength = idxConverter.numberOfPossibleIndices();

	MatrixAType A;
	VectorBType b;

	A.resize(shadingVectorLength, shadingVectorLength);
	b.resize(shadingVectorLength);

	//retinex constraint

	ComputeType diffThresh = diffThreshold;

	MatrixAType Aretinex;
	Aretinex.resize(shadingVectorLength, shadingVectorLength);
	Aretinex.reserve(Eigen::VectorXi::Constant(shadingVectorLength, nDim*2+1));


	VectorBType b_retinex = VectorBType::Zero(shadingVectorLength);

	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		for (int dim = 0; dim < nDim; dim++) {

			if (dim == channelDim) {
				continue;
			}

			int d = dim;

			if (dim < channelDim) {
				d += 1;
			}

			for (int delta : {-1, +1}) {
				auto dIdx = idx;
				dIdx[d] += delta;

				if (dIdx[d] < 0 or dIdx[d] >= rgbImage.shape()[d]) {
					continue;
				}

				int j = idxConverter.getPseudoFlatIdFromIndex(dIdx);

				ComputeType diffNormSquared = 0;

				for (int k = 0; k < rgChromaticity.shape()[channelDim]; k++) {
					idx[channelDim] = k;
					dIdx[channelDim] = k;
					ComputeType diff = rgChromaticity.valueUnchecked(idx) - rgChromaticity.valueUnchecked(dIdx);
					diffNormSquared += diff*diff;
				}

				ComputeType diffThreshSquared = diffThresh*diffThresh;

				ComputeType omega = (diffNormSquared > diffThreshSquared) ? 0 : reflectanceToShadingWeight;

				Aretinex.coeffRef(i,i) += 2*nColors*(1+omega);
				Aretinex.coeffRef(i,j) -= 2*nColors*(1+omega);

				for (int c = 0; c < nColors; c++) {

					idx[channelDim] = c;
					dIdx[channelDim] = c;
					ComputeType diffCol = logImg.valueUnchecked(idx) - logImg.valueUnchecked(dIdx);

					b_retinex[i] += 2*omega*diffCol;
				}

			}
		}
	}


	//global texture constraint

	constexpr int searchRadius = 1;
	constexpr int searchWindowSide = 2*searchRadius+1;
	constexpr int searchWindowSize = searchWindowSide*searchWindowSide;

	Multidim::Array<ComputeType, 3> featureVolumeR0 =
			Correlation::unfold(searchRadius, searchRadius, rgChromaticity, PaddingMargins(), Correlation::Rotate0);

	Multidim::Array<ComputeType, 3> featureVolumeR90 =
			Correlation::unfold(searchRadius, searchRadius, rgChromaticity, PaddingMargins(), Correlation::Rotate90);

	Multidim::Array<ComputeType, 3> featureVolumeR180 =
			Correlation::unfold(searchRadius, searchRadius, rgChromaticity, PaddingMargins(), Correlation::Rotate180);

	Multidim::Array<ComputeType, 3> featureVolumeR270 =
			Correlation::unfold(searchRadius, searchRadius, rgChromaticity, PaddingMargins(), Correlation::Rotate270);

	Indexers::FixedSizeDisjointSetForest nodesClusters(idxConverter.numberOfPossibleIndices());

	std::forward_list<int> toTreat;
	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		if (idx[0] < searchRadius or idx[0] >= rgbImage.shape()[0]-searchRadius) {
			continue;
		}

		if (idx[1] < searchRadius or idx[1] >= rgbImage.shape()[1]-searchRadius) {
			continue;
		}

		//we insert only the windows that are fully within the image to be compared.
		toTreat.push_front(i);
	}

	std::vector<std::optional<int>> matchedOrientationIdx(idxConverter.numberOfPossibleIndices());
	std::fill(matchedOrientationIdx.begin(), matchedOrientationIdx.end(), std::nullopt);
	std::vector<float> matchedCost(idxConverter.numberOfPossibleIndices());

	std::array<Correlation::UnfoldPatchOrientation, 4> consideredOrientations = {Correlation::UnfoldPatchOrientation::Rotate0,
																				 Correlation::UnfoldPatchOrientation::Rotate90,
																				 Correlation::UnfoldPatchOrientation::Rotate180,
																				 Correlation::UnfoldPatchOrientation::Rotate270};

	std::array<Multidim::Array<ComputeType, 3>*, 4> featureVolumeRefForOrientation = {&featureVolumeR0,
																					  &featureVolumeR90,
																					  &featureVolumeR180,
																					  &featureVolumeR270};


	//compute the groups
	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		if (idx[0] < searchRadius or idx[0] >= rgbImage.shape()[0]-searchRadius) {
			continue;
		}

		if (idx[1] < searchRadius or idx[0] >= rgbImage.shape()[1]-searchRadius) {
			continue;
		}

		std::array<int,2> fVecPos = {idx[0], idx[1]};

		std::array<fVecT, 4> fVecs;

		for (int o = 0; o < 4; o++) {
			fVecs[o] = featureVolumeRefForOrientation[o]->indexDimView(2,fVecPos);
		}

		std::forward_list<int>::iterator previous = toTreat.before_begin();
		std::forward_list<int>::iterator current = std::next(previous);

		int nToTreat = std::distance(toTreat.begin(), toTreat.end());
		int counter = 0;
		while (std::next(previous) != toTreat.end()) {

			counter++;

			current = std::next(previous);

			if (*current == i) {
				toTreat.erase_after(previous); //always remove the current item from items to treat later.
				current = std::next(previous);

				if (current == toTreat.end()) {
					break;
				}
			}

			auto targetIdx = idxConverter.getIndexFromPseudoFlatId(*current);
			std::array<int,2> fTargetVecPos = {targetIdx[0], targetIdx[1]};

			fVecT fVecTarget = featureVolumeR0.indexDimView(2,fTargetVecPos);
			std::array<float, 4> costs;

			for (int o = 0; o < 4; o++) {
				costs[o] = Correlation::MatchingFunctionTraits<Correlation::matchingFunctions::SSD>::featureComparison(fVecs[0], fVecTarget)/(searchWindowSize*2);
			}

			int minOrientation = 0;
			float minCost = costs[0];

			for (int o = 1; o < 4; o++) {

				if (costs[o] < minCost) {
					minOrientation = o;
					minCost = costs[o];
				}
			}

			if (minCost < groupingThreshold) {
				nodesClusters.joinNode(*current, i);
				matchedOrientationIdx[*current] = minOrientation;
				matchedCost[*current] = minCost;
				toTreat.erase_after(previous);
			}

			previous = std::next(previous);

			if (previous == toTreat.end()) {
				break;
			}

		}

	}

	// build the median of each group and then prepare the triplets
	// and build the texture matrix

	MatrixAType Atexture;
	Atexture.resize(shadingVectorLength, shadingVectorLength);
	Atexture.reserve(Eigen::VectorXi::Constant(shadingVectorLength, 3));

	VectorBType b_texture = VectorBType::Constant(shadingVectorLength, 0);

	std::vector<int> groups = nodesClusters.getGroups();
	std::vector<std::vector<int>> groupsElements = nodesClusters.getGroupsElements();

	using Triplet = Eigen::Triplet<ComputeType>;
	std::vector<Triplet> triplets;
	triplets.reserve(3*idxConverter.numberOfPossibleIndices());

	for (int g : groups) {

		int groupSize = nodesClusters.getGroupSize(g);

		if (groupSize <= 1) {
			continue;
		}

		int P2Radius = searchRadius+2;
		int P2size = 2*P2Radius+1;
		Multidim::Array<ComputeType, 3> medianP2Patch(P2size, P2size, 2);

		for (int d0 = -P2Radius; d0 <= P2Radius; d0++) {
			for (int d1 = -P2Radius; d1 <= P2Radius; d1++) {
				for (int c = 0; c < 2; c++) {
					std::vector<ComputeType> values;
					values.reserve(groupSize);

					medianP2Patch.at(P2Radius+d0, P2Radius+d1, c) = 0;

					for (int i : groupsElements[g]) {
						auto idx = idxConverter.getIndexFromPseudoFlatId(i);

						int rotationId = matchedOrientationIdx[i].value_or(0);

						std::tuple<int,int> coords = Correlation::rotatedOffsetsFromOrientation(d0, d1, consideredOrientations[rotationId]);

						int di = std::get<0>(coords);
						int dj = std::get<1>(coords);

						idx[0] += di;
						idx[1] += dj;

						if (idx[0] < 0 or idx[0] >= rgbImage.shape()[0]) {
							continue;
						}

						if (idx[1] < 0 or idx[1] >= rgbImage.shape()[1]) {
							continue;
						}

						idx[2] = c;
						values.push_back(rgChromaticity.valueUnchecked(idx));
					}

					int medianId = values.size()/2;
					std::nth_element(values.begin(), values.begin()+medianId, values.end());

					medianP2Patch.at(P2Radius+d0, P2Radius+d1, c) = values[medianId];
				}

			}
		}

		std::array<fVecT, 4> vFecsMedianP2;
		std::array<fVecT, 4> vFecsMedianP1;
		std::array<fVecT, 4> vFecsMedianP0;

		for (int i = 0; i < 4; i++) {
			Multidim::Array<ComputeType, 3> featureVolMedianP2 =
					Correlation::unfold(searchRadius+2, searchRadius+2, medianP2Patch, PaddingMargins(0), consideredOrientations[i]);
			Multidim::Array<ComputeType, 3> featureVolMedianP1 =
					Correlation::unfold(searchRadius+1, searchRadius+1, medianP2Patch, PaddingMargins(0), consideredOrientations[i]);
			Multidim::Array<ComputeType, 3> featureVolMedianP0 =
					Correlation::unfold(searchRadius, searchRadius, medianP2Patch, PaddingMargins(0), consideredOrientations[i]);

			vFecsMedianP2[i] = featureVolMedianP2.indexDimView(2,{0,0});
			vFecsMedianP1[i] = featureVolMedianP1.indexDimView(2,{1,1});
			vFecsMedianP0[i] = featureVolMedianP0.indexDimView(2,{2,2});

		}

		Multidim::Array<ComputeType, 3> featureVolP2 = Correlation::unfold(searchRadius+2, searchRadius+2, rgChromaticity, PaddingMargins());
		Multidim::Array<ComputeType, 3> featureVolP1 = Correlation::unfold(searchRadius+1, searchRadius+1, rgChromaticity, PaddingMargins());
		Multidim::Array<ComputeType, 3> featureVolP0 = Correlation::unfold(searchRadius, searchRadius, rgChromaticity, PaddingMargins());


		std::array<Multidim::Array<ComputeType, 3>*, 3> fVols = {&featureVolP0, &featureVolP1, &featureVolP2};
		std::array<std::array<fVecT, 4>*, 3> fVecsMediansPs = {&vFecsMedianP0, &vFecsMedianP1, &vFecsMedianP2};

		int p;

		std::array<int,3> idxP;
		std::array<int,2> fTargetVecPosP;

		int rotationIdP;

		int Kp;
		ComputeType cp;


		for (int e = 0; e < groupsElements[g].size()-1; e++) {

			int i = groupsElements[g][e];

			auto idx = idxConverter.getIndexFromPseudoFlatId(i);
			std::array<int,2> fTargetVecPos = {idx[0], idx[1]};

			int rotationId = matchedOrientationIdx[i].value_or(0);

			int K = 3;
			float gCost = 1;

			std::array<int, 3> Ks = {3,5,7};

			for (int k = 0; k < 3; k++) {
				fVecT fvec = fVols[k]->indexDimView(2,fTargetVecPos);
				float cost = Correlation::MatchingFunctionTraits<Correlation::matchingFunctions::SSD>::featureComparison(fvec, (*fVecsMediansPs[k])[rotationId])/(fvec.shape()[0]);

				if (cost < groupingThreshold) {
					K = Ks[k];
				}

				if (cost < gCost) {
					gCost = cost;
				}
			}

			ComputeType cq = K*(1-gCost);

			ComputeType diff;

			if (e > 0) {

				for (int c = 0; c < nColors; c++) {

					idx[2] = c;
					idxP[2] = c;

					ComputeType coeff = 2*cp*cq;

					ComputeType bVal = coeff*(logImg.valueUnchecked(idx) - logImg.valueUnchecked(idxP));
					b_texture[p] += -bVal;
					b_texture[i] += bVal;

					Atexture.coeffRef(p, i) -= coeff;
					Atexture.coeffRef(i, p) -= coeff;

					Atexture.coeffRef(p, p) += coeff;
					Atexture.coeffRef(i, i) += coeff;
				}

			}

			p = i;

			idxP = idx;
			fTargetVecPosP = fTargetVecPos;

			rotationIdP = rotationId;

			Kp = K;
			cp = cq;

		}
	}

	//scale constraint

	//determine brighthest pixels

	Multidim::Array<ComputeType, nDim-1> grayImg = img2gray(rgbImage, {1.,1.,1.});

	ComputeType bright = 0;


	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);
		std::array<int,2> gidx = {idx[0], idx[1]};

		ComputeType val = grayImg.valueUnchecked(gidx);

		if (val > bright) {
			bright = val;
		}
	}

	//build the matrix

	MatrixAType Ascale;
	Ascale.resize(shadingVectorLength, shadingVectorLength);
	Ascale.reserve(Eigen::VectorXi::Constant(shadingVectorLength, 1));

	VectorBType b_scale = VectorBType::Constant(shadingVectorLength, 0);

	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);
		std::array<int,2> gidx = {idx[0], idx[1]};

		ComputeType val = grayImg.valueUnchecked(gidx);

		if (val >= 0.95*bright) {
			Ascale.coeffRef(i, i) += 2;
			b_scale[i] += 2*exp(1);
		}
	}

	//compute the total matrix and vector
	A = lambdaRetinex*Aretinex;
	A += lambdaTexture*Atexture;
	A += lambdaScaling*Ascale;
	b = lambdaRetinex*b_retinex + lambdaTexture*b_texture + lambdaScaling*b_scale;

	SolverType solver;
	solver.compute(A);

	VectorSolType solution = solver.solve(b);

	if(solver.info()!=Eigen::Success) {
		return IntrinsicImageDecomposition<ComputeType, 3>{ Multidim::Array<ComputeType, 3>(), Multidim::Array<ComputeType, 3>() };
	}

	Multidim::Array<ComputeType, 3> logR(rgbImage.shape());
	Multidim::Array<ComputeType, 3> logS(rgbImage.shape());

	#pragma omp parallel for
	for (int i = 0; i < idxConverter.numberOfPossibleIndices(); i++) {
		auto idx = idxConverter.getIndexFromPseudoFlatId(i);

		idx[channelDim] = 0;
		logS.atUnchecked(idx) = solution[i];
		logR.atUnchecked(idx) = logImg.valueUnchecked(idx) - logS.valueUnchecked(idx);
		idx[channelDim] = 1;
		logS.atUnchecked(idx) = solution[i];
		logR.atUnchecked(idx) = logImg.valueUnchecked(idx) - logS.valueUnchecked(idx);
		idx[channelDim] = 2;
		logS.atUnchecked(idx) = solution[i];
		logR.atUnchecked(idx) = logImg.valueUnchecked(idx) - logS.valueUnchecked(idx);
	}

	Multidim::Array<ComputeType, 3> R = linear2logColorSpaceImg(logR);
	Multidim::Array<ComputeType, 3> S = linear2logColorSpaceImg(logS);

	return IntrinsicImageDecomposition<ComputeType, 3>{ R, S };

}

template<typename T, typename ComputeType=float>
IntrinsicImageDecomposition<ComputeType, 3> autoRetinexWithNonLocalTextureConstraint(Multidim::Array<T, 3> const& rgbImage,
																					 ComputeType lambdaRetinex = 1.0,
																					 ComputeType lambdaTexture = 1.0,
																					 ComputeType lambdaScaling = 1000.,
																					 ComputeType reflectanceToShadingWeight = 100.,
																					 float groupingThreshold = 0.01,
																					 ComputeType histBinSize = 1.,
																					 ComputeType minVal = 0.,
																					 ComputeType maxVal = 255.) {

	static_assert (std::is_floating_point_v<ComputeType>, "Compute type must be a floating point type");

	constexpr int nDiffThresh = 12;

	std::array<ComputeType, nDiffThresh> test;

	constexpr ComputeType minDiffThresh = 0.00001;
	constexpr ComputeType maxDiffThresh = 0.005;
	constexpr ComputeType expandDiffThresh = maxDiffThresh - minDiffThresh;
	constexpr ComputeType binDiffThresh = expandDiffThresh/nDiffThresh;


	for (int i = 0; i < nDiffThresh; i++) {
		test[i] = i*binDiffThresh;
	}

	double minEntropy = std::numeric_limits<double>::infinity();

	IntrinsicImageDecomposition<ComputeType, 3> ret;

	for (ComputeType diffThresh : test) {

		IntrinsicImageDecomposition<ComputeType, 3> cand =  retinexWithNonLocalTextureConstraint(rgbImage,
																								 diffThresh,
																								 lambdaRetinex,
																								 lambdaTexture,
																								 lambdaScaling,
																								 reflectanceToShadingWeight,
																								 groupingThreshold);

		if (cand.shading.empty() or cand.reflectance.empty()) {
			std::cerr << "intrisic image decomposition failed!" << std::endl;
			continue;
		}

		Multidim::Array<ComputeType, 2> view = cand.shading.sliceView(2,0);


		Histogram<ComputeType> hist(view, histBinSize, minVal, maxVal);

		double cand_entropy = hist.entropy();

		if (cand_entropy < minEntropy) {
			minEntropy = cand_entropy;
			ret.shading = std::move(cand.shading);
			ret.reflectance = std::move(cand.reflectance);
		}

	}

	return ret;

}

template<typename T, int nDim, typename ComputeType=float>
IntrinsicImageDecomposition<ComputeType, nDim> performIntrinsicImageDecomposition(Multidim::Array<T, nDim> const& original,
																				  ComputeType lambda,
																				  std::optional<int> channelDim= (nDim > 2) ? std::optional<int>(-1) : std::nullopt,
																				  int maxIterations = 100,
																				  Multidim::Array<ComputeType, nDim> initialRefl = Multidim::Array<ComputeType, nDim>())
{

	typedef Multidim::Array<T, nDim> MDArray;
	typedef Multidim::Array<ComputeType, nDim> RMDArray;
	typedef IntrinsicImageDecomposition<ComputeType, nDim> RStruct;

	int excludedDim = -1;

	if (channelDim.has_value()) {
		if (channelDim.value() >= 0) {
			excludedDim = channelDim.value();
		} else {
			excludedDim = nDim + channelDim.value();
		}
	}

	int nPixs = 1;

	for (int i = 0; i < nDim; i++) {
		if (i != excludedDim) {
			nPixs *= original.shape()[i];
		}
	}


}

} // namespace StereoVision
} //namespace ImageProcessing

#endif // INTRINSICIMAGEDECOMPOSITION_H
