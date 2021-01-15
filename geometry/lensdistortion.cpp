/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2021  Paragon<french.paragon@gmail.com>

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

#include "lensdistortion.h"

#include <Eigen/Dense>

namespace StereoVision {
namespace Geometry {

Eigen::Vector2f radialDistortion(Eigen::Vector2f pos, Eigen::Vector3f k123) {

	float r2 = pos(0)*pos(0) + pos(1)*pos(1);

	Eigen::Vector3f vr;
	vr << r2, r2*r2, r2*r2*r2;

	return k123.dot(vr)*pos;
}

Eigen::Vector2d radialDistortionD(Eigen::Vector2d pos, Eigen::Vector3d k123) {

	double r2 = pos(0)*pos(0) + pos(1)*pos(1);

	Eigen::Vector3d vr;
	vr << r2, r2*r2, r2*r2*r2;

	return k123.dot(vr)*pos;
}

Eigen::Vector2f tangentialDistortion(Eigen::Vector2f pos, Eigen::Vector2f t12) {

	float r2 = pos(0)*pos(0) + pos(1)*pos(1);

	Eigen::Vector2f td;
	td << t12(1)*(r2 + 2*pos(0)*pos(0)) + 2*t12(0)*pos(0)*pos(1),
			t12(0)*(r2 + 2*pos(1)*pos(1)) + 2*t12(1)*pos(0)*pos(1);

	return td;

}
Eigen::Vector2d tangentialDistortionD(Eigen::Vector2d pos, Eigen::Vector2d t12) {

	double r2 = pos(0)*pos(0) + pos(1)*pos(1);

	Eigen::Vector2d td;
	td << t12(1)*(r2 + 2*pos(0)*pos(0)) + 2*t12(0)*pos(0)*pos(1),
			t12(0)*(r2 + 2*pos(1)*pos(1)) + 2*t12(1)*pos(0)*pos(1);

	return td;
}

template<int n>
float iPow(float f) {
	float r = 1;

	for (int i = 0; i < n; i++) {
		r *= f;
	}

	return r;
}

Eigen::Vector2f invertRadialDistorstion(Eigen::Vector2f pos, Eigen::Vector3f k123, int iters) {

	float rb = pos.norm();
	float r = rb;

	float& k1 = k123[0];
	float& k2 = k123[1];
	float& k3 = k123[2];

	for (int i = 0; i < iters; i++) {
		r = r - (r + k1*iPow<3>(r) + k2*iPow<5>(r) + k3*iPow<7>(r) - rb)/(1 + 3*k1*iPow<2>(r) + 5*k2*iPow<4>(r) + 7*k3*iPow<6>(r));
	}

	return pos * (r/rb);

}
Eigen::Vector2f invertTangentialDistorstion(Eigen::Vector2f pos, Eigen::Vector2f t12, int iters) {

	Eigen::Vector2f npos = pos;

	float& t1 = t12[0];
	float& t2 = t12[1];

	for (int i = 0; i < iters; i++) {

		float r2 = iPow<2>(npos[0]) + iPow<2>(npos[1]);

		Eigen::Vector2f f;
		f << npos(0) + t2*(r2 + 2*npos(0)*npos(0)) + 2*t1*npos(0)*npos(1) - pos(0),
				npos(1) + t1*(r2 + 2*npos(1)*npos(1)) + 2*t2*npos(0)*npos(1) - pos(1);

		Eigen::Matrix2f df;
		df << 1 + 6*t2*npos(0) + 2*t1*npos(1), 2*t2*npos(1) + 2*t1*npos(0),
			  2*t2*npos(1) + 2*t1*npos(0), 1 + 6*t1*npos(1) + 2*t2*npos(0);

		npos = npos - df.inverse()*f;

	}

	return npos;

}
Eigen::Vector2f invertRadialTangentialDistorstion(Eigen::Vector2f pos, Eigen::Vector3f k123, Eigen::Vector2f t12, int iters) {

	Eigen::Vector2f npos = pos;

	float& k1 = k123[0];
	float& k2 = k123[1];
	float& k3 = k123[2];

	float& t1 = t12[0];
	float& t2 = t12[1];

	for (int i = 0; i < iters; i++) {

		float r2 = iPow<2>(npos[0]) + iPow<2>(npos[1]);

		float dr = (k1*r2 + k2*iPow<2>(r2) + k3*iPow<3>(r2));
		float dx_r = npos(0)*dr;
		float dy_r = npos(1)*dr;

		Eigen::Vector2f f;
		f << npos(0) + dx_r + t2*(r2 + 2*npos(0)*npos(0)) + 2*t1*npos(0)*npos(1) - pos(0),
				npos(1) + dy_r + t1*(r2 + 2*npos(1)*npos(1)) + 2*t2*npos(0)*npos(1) - pos(1);

		float drdr2 = k1 + 2*k2*r2 + 3*k3*iPow<2>(r2);
		float drdx = 2*drdr2*npos(0);
		float drdy = 2*drdr2*npos(1);

		Eigen::Matrix2f df;
		df << 1 + (dr + npos(0)*drdx) + 6*t2*npos(0) + 2*t1*npos(1), 2*t2*npos(1) + 2*t1*npos(0) + (npos(0)*drdy),
			  2*t2*npos(1) + 2*t1*npos(0) + (npos(1)*drdx), 1 + (dr + npos(1)*drdy) + 6*t1*npos(1) + 2*t2*npos(0);

		npos = npos - df.inverse()*f;

	}

	return npos;

}


Eigen::Vector2f skewDistortion(Eigen::Vector2f pos, Eigen::Vector2f B12, float f, Eigen::Vector2f pp) {

	Eigen::Vector2f r = f*pos + pp;
	r[0] += B12[0]*pos[0] + B12[1]*pos[1];
	return r;
}
Eigen::Vector2d skewDistortionD(Eigen::Vector2d pos, Eigen::Vector2d B12, double f, Eigen::Vector2d pp) {

	Eigen::Vector2d r = f*pos + pp;
	r[0] += B12[0]*pos[0] + B12[1]*pos[1];
	return r;
}

Eigen::Vector2f inverseSkewDistortion(Eigen::Vector2f pos, Eigen::Vector2f B12, float f, Eigen::Vector2f pp) {

	float y = (pos[1] - pp[1])/f;
	float x = (pos[0] - B12[1]*y - pp[0])/(f + B12[0]);

	Eigen::Vector2f r(x,y);

	return r;
}

} // namespace Geometry
} // namespace StereoVision
