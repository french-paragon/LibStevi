#ifndef STEREOVISION_UTILS_CONTIGUITY_H
#define STEREOVISION_UTILS_CONTIGUITY_H

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

#include <array>

namespace StereoVision {

class Contiguity {

public:
	enum generalContiguity {
		singleDimCanChange,
                allDimsCanChange
	};

	enum bidimensionalContiguity {
		Rook,
		Bishop,
		Queen
	};

	template<bidimensionalContiguity contiguity>
	class BidimensionalContiguityTraits {

	};

	constexpr static int nDirections(bidimensionalContiguity contiguity) {
		return (contiguity == bidimensionalContiguity::Queen) ? 8 : 4;
	}

	constexpr static int nCornerDirections(bidimensionalContiguity contiguity) {
		return (contiguity == bidimensionalContiguity::Queen) ? 3 : ((contiguity == bidimensionalContiguity::Rook) ? 2 : 1);
	}

	constexpr static int nTilingDirections(bidimensionalContiguity contiguity) {
		return (contiguity == bidimensionalContiguity::Queen) ? 4 : 2;
	}

	template<bidimensionalContiguity contiguity>
	constexpr static std::array<std::array<int, 2>, BidimensionalContiguityTraits<contiguity>::nDir> getDirections() {
		return std::array<std::array<int, 2>, BidimensionalContiguityTraits<contiguity>::nDir>();
	}

	template<bidimensionalContiguity contiguity>
	constexpr static std::array<std::array<int, 2>, nCornerDirections(contiguity)> getCornerDirections() {
		return std::array<std::array<int, 2>, nCornerDirections(contiguity)>();
	}

	template<bidimensionalContiguity contiguity>
	constexpr static std::array<std::array<int, 2>, nTilingDirections(contiguity)> getTilingDirections() {
		return std::array<std::array<int, 2>, nTilingDirections(contiguity)>();
	}

};

template<>
class Contiguity::BidimensionalContiguityTraits<Contiguity::Queen> {
public:
	static constexpr int nDir = 8;
	static constexpr int nCornerDir = 3;
	static constexpr int nTilingDir = 4;
};

template<>
class Contiguity::BidimensionalContiguityTraits<Contiguity::Rook> {
public:
	static constexpr int nDir = 4;
	static constexpr int nCornerDir = 2;
	static constexpr int nTilingDir = 2;
};

template<>
class Contiguity::BidimensionalContiguityTraits<Contiguity::Bishop> {
public:
	static constexpr int nDir = 4;
	static constexpr int nCornerDir = 1;
	static constexpr int nTilingDir = 2;
};

template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Queen>::nDir> Contiguity::getDirections<Contiguity::Queen>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, nDirections(bidimensionalContiguity::Queen)> returnArrayType;

	return returnArrayType({deltasArrayType({1,1}),
							deltasArrayType({1,0}),
							deltasArrayType({1,-1}),
							deltasArrayType({0,1}),
							deltasArrayType({0,-1}),
							deltasArrayType({-1,1}),
							deltasArrayType({-1,0}),
							deltasArrayType({-1,-1})});
}
template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Rook>::nDir> Contiguity::getDirections<Contiguity::Rook>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, BidimensionalContiguityTraits<Rook>::nDir> returnArrayType;

	return returnArrayType({deltasArrayType({1,0}),
							deltasArrayType({0,1}),
							deltasArrayType({0,-1}),
							deltasArrayType({-1,0})});
}

template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Bishop>::nDir> Contiguity::getDirections<Contiguity::Bishop>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, BidimensionalContiguityTraits<Bishop>::nDir> returnArrayType;

	return returnArrayType({deltasArrayType({1,1}),
							deltasArrayType({1,-1}),
							deltasArrayType({-1,1}),
							deltasArrayType({-1,-1})});
}



template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Queen>::nCornerDir> Contiguity::getCornerDirections<Contiguity::Queen>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, Contiguity::BidimensionalContiguityTraits<Contiguity::Queen>::nCornerDir> returnArrayType;

	return returnArrayType({deltasArrayType({1,1}), deltasArrayType({1,0}), deltasArrayType({0,1})});
}

template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Rook>::nCornerDir> Contiguity::getCornerDirections<Contiguity::Rook>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, Contiguity::BidimensionalContiguityTraits<Contiguity::Rook>::nCornerDir> returnArrayType;

	return returnArrayType({deltasArrayType({1,0}), deltasArrayType({0,1})});
}

template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Bishop>::nCornerDir> Contiguity::getCornerDirections<Contiguity::Bishop>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, Contiguity::BidimensionalContiguityTraits<Contiguity::Bishop>::nCornerDir> returnArrayType;

	return returnArrayType({deltasArrayType({1,1})});
}



template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Queen>::nTilingDir> Contiguity::getTilingDirections<Contiguity::Queen>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, Contiguity::BidimensionalContiguityTraits<Contiguity::Queen>::nTilingDir> returnArrayType;

	return returnArrayType({deltasArrayType({1,1}), deltasArrayType({1,0}), deltasArrayType({0,1}), deltasArrayType({1,-1})});
}

template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Rook>::nTilingDir> Contiguity::getTilingDirections<Contiguity::Rook>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, Contiguity::BidimensionalContiguityTraits<Contiguity::Rook>::nTilingDir> returnArrayType;

	return returnArrayType({deltasArrayType({1,0}), deltasArrayType({0,1})});
}

template<>
constexpr std::array<std::array<int, 2>, Contiguity::BidimensionalContiguityTraits<Contiguity::Bishop>::nTilingDir> Contiguity::getTilingDirections<Contiguity::Bishop>() {

	typedef std::array<int, 2> deltasArrayType;
	typedef std::array<deltasArrayType, Contiguity::BidimensionalContiguityTraits<Contiguity::Bishop>::nTilingDir> returnArrayType;

	return returnArrayType({deltasArrayType({1,1}), deltasArrayType({1,-1})});
}

} // namespace StereoVision

#endif // CONTIGUITY_H
