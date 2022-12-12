#ifndef STEREOVISION_LENSDISTORTION_H
#define STEREOVISION_LENSDISTORTION_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021-2022  Paragon<french.paragon@gmail.com>

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

#include "../geometry/core.h"
#include "../geometry/imagecoordinates.h"
#include "../utils/stevimath.h"

#include <optional>
#include <Eigen/Dense>

namespace StereoVision {
namespace Geometry {

template<typename Pos_T, typename K_T>
Eigen::Matrix<Pos_T, 2, 1> radialDistortion(Eigen::Matrix<Pos_T, 2, 1> pos, Eigen::Matrix<K_T, 3, 1> k123) {

	static_assert (std::is_floating_point_v<Pos_T> and std::is_floating_point_v<K_T>, "The position and k parameters type should be float point types");

	Pos_T r2 = pos(0)*pos(0) + pos(1)*pos(1);

	Eigen::Matrix<Pos_T, 3, 1> vr;
	vr << r2, r2*r2, r2*r2*r2;

	return k123.template cast<Pos_T>().dot(vr)*pos;
}

template<typename Pos_T, typename T_T>
Eigen::Matrix<Pos_T, 2, 1> tangentialDistortion(Eigen::Matrix<Pos_T, 2, 1> pos, Eigen::Matrix<T_T, 2, 1> t12) {

	static_assert (std::is_floating_point_v<Pos_T> and std::is_floating_point_v<T_T>, "The position and t parameters type should be float point types");

	Pos_T r2 = pos(0)*pos(0) + pos(1)*pos(1);

	Eigen::Matrix<Pos_T, 2, 1> td;
	td << t12(1)*(r2 + 2*pos(0)*pos(0)) + 2*t12(0)*pos(0)*pos(1),
			t12(0)*(r2 + 2*pos(1)*pos(1)) + 2*t12(1)*pos(0)*pos(1);

	return td;

}

template<typename Pos_T, typename K_T>
Eigen::Matrix<Pos_T, 2, 1> invertRadialDistorstion(Eigen::Matrix<Pos_T, 2, 1> pos,
										Eigen::Matrix<K_T, 3, 1> k123,
										int iters = 5) {

	static_assert (std::is_floating_point_v<Pos_T> and std::is_floating_point_v<K_T>, "The position and k parameters type should be float point types");

	Pos_T rb = pos.norm();
	Pos_T r = rb;

	Pos_T& k1 = k123[0];
	Pos_T& k2 = k123[1];
	Pos_T& k3 = k123[2];

	for (int i = 0; i < iters; i++) {
		r = r - (r + k1*Math::iPow<3>(r) + k2*Math::iPow<5>(r) + k3*Math::iPow<7>(r) - rb)/
				(1 + 3*k1*Math::iPow<2>(r) + 5*k2*Math::iPow<4>(r) + 7*k3*Math::iPow<6>(r));
	}

	return pos * (r/rb);

}

template<typename Pos_T, typename T_T>
Eigen::Matrix<Pos_T, 2, 1> invertTangentialDistorstion(Eigen::Matrix<Pos_T, 2, 1> pos,
											Eigen::Matrix<T_T, 2, 1> t12,
											int iters = 5) {

	static_assert (std::is_floating_point_v<Pos_T> and std::is_floating_point_v<T_T>, "The position and t parameters type should be float point types");

	Eigen::Matrix<Pos_T, 2, 1> npos = pos;

	T_T& t1 = t12[0];
	T_T& t2 = t12[1];

	for (int i = 0; i < iters; i++) {

		Pos_T r2 = Math::iPow<2>(npos[0]) + Math::iPow<2>(npos[1]);

		Eigen::Matrix<Pos_T, 2, 1> f;
		f << npos(0) + t2*(r2 + 2*npos(0)*npos(0)) + 2*t1*npos(0)*npos(1) - pos(0),
				npos(1) + t1*(r2 + 2*npos(1)*npos(1)) + 2*t2*npos(0)*npos(1) - pos(1);

		Eigen::Matrix<Pos_T, 2, 2> df;
		df << 1 + 6*t2*npos(0) + 2*t1*npos(1), 2*t2*npos(1) + 2*t1*npos(0),
			  2*t2*npos(1) + 2*t1*npos(0), 1 + 6*t1*npos(1) + 2*t2*npos(0);

		npos = npos - df.inverse()*f;

	}

	return npos;
}

template<typename Pos_T, typename K_T, typename T_T>
Eigen::Matrix<Pos_T, 2, 1> invertRadialTangentialDistorstion(Eigen::Matrix<Pos_T, 2, 1> pos,
												  Eigen::Matrix<K_T, 3, 1> k123,
												  Eigen::Matrix<T_T, 2, 1> t12,
												  int iters = 5) {

	static_assert (std::is_floating_point_v<Pos_T> and std::is_floating_point_v<K_T> and std::is_floating_point_v<T_T>,
			"The position , k and t parameters type should be float point types");

	Eigen::Matrix<Pos_T, 2, 1> npos = pos;

	K_T& k1 = k123[0];
	K_T& k2 = k123[1];
	K_T& k3 = k123[2];

	T_T& t1 = t12[0];
	T_T& t2 = t12[1];

	for (int i = 0; i < iters; i++) {

		Pos_T r2 = Math::iPow<2>(npos[0]) + Math::iPow<2>(npos[1]);

		Pos_T dr = (k1*r2 + k2*Math::iPow<2>(r2) + k3*Math::iPow<3>(r2));
		Pos_T dx_r = npos(0)*dr;
		Pos_T dy_r = npos(1)*dr;

		Eigen::Matrix<Pos_T, 2, 1> f;
		f << npos(0) + dx_r + t2*(r2 + 2*npos(0)*npos(0)) + 2*t1*npos(0)*npos(1) - pos(0),
				npos(1) + dy_r + t1*(r2 + 2*npos(1)*npos(1)) + 2*t2*npos(0)*npos(1) - pos(1);

		Pos_T drdr2 = k1 + 2*k2*r2 + 3*k3*Math::iPow<2>(r2);
		Pos_T drdx = 2*drdr2*npos(0);
		Pos_T drdy = 2*drdr2*npos(1);

		Eigen::Matrix<Pos_T, 2, 2> df;
		df << 1 + (dr + npos(0)*drdx) + 6*t2*npos(0) + 2*t1*npos(1), 2*t2*npos(1) + 2*t1*npos(0) + (npos(0)*drdy),
			  2*t2*npos(1) + 2*t1*npos(0) + (npos(1)*drdx), 1 + (dr + npos(1)*drdy) + 6*t1*npos(1) + 2*t2*npos(0);

		npos = npos - df.inverse()*f;

	}

	return npos;

}

template<typename Pos_T, typename B_T>
Eigen::Matrix<Pos_T, 2, 1> skewDistortion(Eigen::Matrix<Pos_T, 2, 1> const& pos,
										  Eigen::Matrix<B_T, 2, 1> const&  B12,
										  Eigen::Matrix<Pos_T, 2, 1> const&  f,
										  Eigen::Matrix<Pos_T, 2, 1> const&  pp) {

	static_assert (std::is_floating_point_v<Pos_T> and std::is_floating_point_v<B_T>, "The position and B parameters type should be float point types");

	Eigen::Matrix<Pos_T, 2, 1> r = Homogeneous2ImageCoordinates(pos, f, pp, ImageAnchors::TopLeft);
	r[0] += B12[0]*pos[0] + B12[1]*pos[1];
	return r;

}

template<typename Pos_T, typename B_T>
Eigen::Matrix<Pos_T, 2, 1> skewDistortion(Eigen::Matrix<Pos_T, 2, 1> const& pos,
										  Eigen::Matrix<B_T, 2, 1> const&  B12,
										  Pos_T f,
										  Eigen::Matrix<Pos_T, 2, 1> const&  pp) {

	return skewDistortion(pos, B12, Eigen::Matrix<Pos_T, 2, 1>(f,f), pp);
}

template<typename Pos_T, typename B_T>
Eigen::Matrix<Pos_T, 2, 1> inverseSkewDistortion(Eigen::Matrix<Pos_T, 2, 1> const& pos,
									  Eigen::Matrix<B_T, 2, 1> const& B12,
									  Eigen::Matrix<Pos_T, 2, 1> const& f,
									  Eigen::Matrix<Pos_T, 2, 1> const& pp) {

	static_assert (std::is_floating_point_v<Pos_T> and std::is_floating_point_v<B_T>, "The position and B parameters type should be float point types");

	Eigen::Matrix<Pos_T, 2, 1> r = pos;
	r[1] -= pp[1];
	r[1] /= f[1];
	r[0] -= B12[1]*r[1];
	r[0] -= pp[0];
	r[0] /= (f[0] + B12[0]);

	return r;
}

template<typename Pos_T, typename B_T>
Eigen::Matrix<Pos_T, 2, 1> inverseSkewDistortion(Eigen::Matrix<Pos_T, 2, 1> pos,
									  Eigen::Matrix<B_T, 2, 1> B12,
									  Pos_T f,
									  Eigen::Matrix<Pos_T, 2, 1> pp) {
	return inverseSkewDistortion(pos, B12, Eigen::Matrix<Pos_T, 2, 1>(f,f), pp);
}


template<typename Pos_T, typename K_T, typename T_T, typename B_T>
/*!
 * \brief fullLensDistortionHomogeneousCoordinates represent a full distortion model were the radial and tangential distortion coefficients are measured in homogeneous coordinates.
 * \return The distorted point in pixel coordinate
 */
Eigen::Matrix<Pos_T, 2, 1> fullLensDistortionHomogeneousCoordinates(Eigen::Matrix<Pos_T, 2, 1> const& pos,
																	Eigen::Matrix<Pos_T, 2, 1> const& f,
																	Eigen::Matrix<Pos_T, 2, 1> const& pp,
																	std::optional<Eigen::Matrix<K_T, 3, 1>> k123 = std::nullopt,
																	std::optional<Eigen::Matrix<T_T, 2, 1>> t12 = std::nullopt,
																	std::optional<Eigen::Matrix<B_T, 2, 1>>  B12 = std::nullopt) {

	static_assert (std::is_floating_point_v<Pos_T> and
			std::is_floating_point_v<K_T> and
			std::is_floating_point_v<T_T> and
			std::is_floating_point_v<B_T>,
			"The position , k, t and B, parameters type should be float point types");

	Eigen::Matrix<Pos_T, 2, 1> drpos = Eigen::Matrix<Pos_T, 2, 1>::Zero();
	if (k123.has_value()) {
		drpos = radialDistortion(pos, k123.value());
	}

	Eigen::Matrix<Pos_T, 2, 1> dtpos = Eigen::Matrix<Pos_T, 2, 1>::Zero();
	if (t12.has_value()) {
		dtpos = tangentialDistortion(pos, t12.value());
	}

	Eigen::Matrix<Pos_T, 2, 1> mpos = pos + drpos + dtpos;

	if (B12.has_value()) {
		return skewDistortion(mpos, B12.value(), f, pp);
	}

	return Homogeneous2ImageCoordinates(mpos, f, pp, ImageAnchors::TopLeft);

}

template<typename Pos_T, typename K_T, typename T_T, typename B_T>
Eigen::Matrix<Pos_T, 2, 1> fullLensDistortionHomogeneousCoordinates(Eigen::Matrix<Pos_T, 2, 1> const& pos,
																	Pos_T f,
																	Eigen::Matrix<Pos_T, 2, 1> const& pp,
																	std::optional<Eigen::Matrix<K_T, 3, 1>> k123 = std::nullopt,
																	std::optional<Eigen::Matrix<T_T, 2, 1>> t12 = std::nullopt,
																	std::optional<Eigen::Matrix<B_T, 2, 1>>  B12 = std::nullopt) {

	return fullLensDistortionHomogeneousCoordinates(pos, Eigen::Matrix<Pos_T, 2, 1>(f,f), pp, k123, t12, B12);
}

template<typename Pos_T, typename K_T, typename T_T, typename B_T>
Eigen::Matrix<Pos_T, 2, 1> invertFullLensDistortionHomogeneousCoordinates(Eigen::Matrix<Pos_T, 2, 1> const& pos,
																		  Eigen::Matrix<Pos_T, 2, 1> const& f,
																		  Eigen::Matrix<Pos_T, 2, 1> const& pp,
																		  std::optional<Eigen::Matrix<K_T, 3, 1>> k123 = std::nullopt,
																		  std::optional<Eigen::Matrix<T_T, 2, 1>> t12 = std::nullopt,
																		  std::optional<Eigen::Matrix<B_T, 2, 1>>  B12 = std::nullopt,
																		  int iters = 5) {

	static_assert (std::is_floating_point_v<Pos_T> and
			std::is_floating_point_v<K_T> and
			std::is_floating_point_v<T_T> and
			std::is_floating_point_v<B_T>,
			"The position , k, t and B, parameters type should be float point types");

	Eigen::Matrix<Pos_T, 2, 1> invSkew;

	if (B12.has_value()) {
		invSkew = inverseSkewDistortion(pos, B12.value(), f, pp);
	} else {
		invSkew = (pos-pp);
		invSkew[0] /= f[0];
		invSkew[1] /= f[1];
	}

	if (!k123.has_value() and !t12.has_value()) {
		return invSkew;
	} else if (!k123.has_value()) {
		return invertTangentialDistorstion(invSkew, t12.value(), iters);
	} else if (!t12.has_value()) {
		return invertRadialDistorstion(invSkew, k123.value(), iters);
	}

	return invertRadialTangentialDistorstion(invSkew, k123.value(), t12.value(), iters);

}

template<typename Pos_T, typename K_T, typename T_T, typename B_T>
Eigen::Matrix<Pos_T, 2, 1> invertFullLensDistortionHomogeneousCoordinates(Eigen::Matrix<Pos_T, 2, 1> const& pos,
																		  Pos_T f,
																		  Eigen::Matrix<Pos_T, 2, 1> const& pp,
																		  std::optional<Eigen::Matrix<K_T, 3, 1>> k123 = std::nullopt,
																		  std::optional<Eigen::Matrix<T_T, 2, 1>> t12 = std::nullopt,
																		  std::optional<Eigen::Matrix<B_T, 2, 1>>  B12 = std::nullopt,
																		  int iters = 5) {
	return invertFullLensDistortionHomogeneousCoordinates(pos, Eigen::Matrix<Pos_T, 2, 1>(f,f), pp, k123, t12, B12, iters);
}

} // namespace Geometry
} // namespace StereoVision

#endif // STEREOVISIONAPP_LENSDISTORTION_H
