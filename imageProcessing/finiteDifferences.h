#ifndef LIBSTEVI_FINITEDIFFERENCES_H
#define LIBSTEVI_FINITEDIFFERENCES_H

#include "../geometry/imagecoordinates.h"
#include "../utils/types_manipulations.h"

#include <MultidimArrays/MultidimArrays.h>

namespace StereoVision {

template<typename T, Geometry::ImageAxis axis, typename O = TypesManipulations::accumulation_extended_t<T>>
Multidim::Array<O, 3> finiteDifference(Multidim::Array<T, 3> const& img) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = img.shape();

	Multidim::Array<O, 3> interm(shape);
	Multidim::Array<O, 3> out(shape);

	for (int i = 0; i < shape[0]; i++) {

		//constant value on edges
		int iPrev = (i > 0) ? i-1: i;
		int iNext = (i+1 < shape[0]) ? i+1: i;

		for (int j = 0; j < shape[1]; j++) {

			//constant value on edges
			int jPrev = (j > 0) ? j-1: j;
			int jNext = (j+1 < shape[1]) ? j+1: j;

			for (int c = 0; c < shape[2]; c++) {

				if (axis == Geometry::ImageAxis::X) {
					interm.template at<Nc>(i,j,c) = static_cast<O>(img.template value<Nc>(i,jNext,c)) - static_cast<O>(img.template value<Nc>(i,jPrev,c));
				} else {
					interm.template at<Nc>(i,j,c) = static_cast<O>(img.template value<Nc>(iNext,j,c)) - static_cast<O>(img.template value<Nc>(iPrev,j,c));
				}

			}
		}
	}

	for (int i = 0; i < shape[0]; i++) {

		//constant value on edges
		int iPrev = (i > 0) ? i-1: i;
		int iNext = (i+1 < shape[0]) ? i+1: i;

		for (int j = 0; j < shape[1]; j++) {

			//constant value on edges
			int jPrev = (j > 0) ? j-1: j;
			int jNext = (j+1 < shape[1]) ? j+1: j;

			for (int c = 0; c < shape[2]; c++) {

				if (axis == Geometry::ImageAxis::X) {
					out.template at<Nc>(i,j,c) = interm.template at<Nc>(iNext,j,c) + 2*interm.template at<Nc>(i,j,c) + interm.template at<Nc>(iPrev,j,c);
				} else {
					out.template at<Nc>(i,j,c) = interm.template at<Nc>(i,jNext,c) + 2*interm.template at<Nc>(i,j,c) + interm.template at<Nc>(i,jPrev,c);
				}

			}
		}
	}

	return out;

}

template<typename T, Geometry::ImageAxis axis, typename O = TypesManipulations::accumulation_extended_t<T>>
Multidim::Array<O, 2> finiteDifference(Multidim::Array<T, 2> const& img) {

	constexpr Multidim::AccessCheck Nc = Multidim::AccessCheck::Nocheck;

	auto shape = img.shape();

	Multidim::Array<O, 2> interm(shape);

	for (int i = 0; i < shape[0]; i++) {

		//constant value on edges
		int iPrev = (i > 0) ? i-1: i;
		int iNext = (i+1 < shape[0]) ? i+1: i;

		for (int j = 0; j < shape[1]; j++) {

			//constant value on edges
			int jPrev = (j > 0) ? j-1: j;
			int jNext = (j+1 < shape[1]) ? j+1: j;

			if (axis == Geometry::ImageAxis::X) {
				interm.template at<Nc>(i,j) = static_cast<O>(img.template value<Nc>(i,jNext)) - static_cast<O>(img.template value<Nc>(i,jPrev));
			} else {
				interm.template at<Nc>(i,j) = static_cast<O>(img.template value<Nc>(iNext,j)) - static_cast<O>(img.template value<Nc>(iPrev,j));
			}
		}
	}

	Multidim::Array<O, 2> out(shape);

	for (int i = 0; i < shape[0]; i++) {

		//constant value on edges
		int iPrev = (i > 0) ? i-1: i;
		int iNext = (i+1 < shape[0]) ? i+1: i;

		for (int j = 0; j < shape[1]; j++) {

			//constant value on edges
			int jPrev = (j > 0) ? j-1: j;
			int jNext = (j+1 < shape[1]) ? j+1: j;

			if (axis == Geometry::ImageAxis::X) {
				out.template at<Nc>(i,j) = interm.template at<Nc>(iNext,j) + 2*interm.template at<Nc>(i,j) + interm.template at<Nc>(iPrev,j);
			} else {
				out.template at<Nc>(i,j) = interm.template at<Nc>(i,jNext) + 2*interm.template at<Nc>(i,j) + interm.template at<Nc>(i,jPrev);
			}
		}
	}

	return out;
}

} //namespace StereoVision

#endif // FINITEDIFFERENCES_H
