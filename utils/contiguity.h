#ifndef CONTIGUITY_H
#define CONTIGUITY_H

#include <array>

namespace StereoVision {

class Contiguity {

public:
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

	template<bidimensionalContiguity contiguity>
	constexpr static std::array<std::array<int, 2>, BidimensionalContiguityTraits<contiguity>::nDir> getDirections() {
		return std::array<std::array<int, 2>, BidimensionalContiguityTraits<contiguity>::nDir>();
	}

	template<bidimensionalContiguity contiguity>
	constexpr static std::array<std::array<int, 2>, nCornerDirections(contiguity)> getCornerDirections() {
		return std::array<std::array<int, 2>, nCornerDirections(contiguity)>();
	}

};

template<>
class Contiguity::BidimensionalContiguityTraits<Contiguity::Queen> {
public:
	static constexpr int nDir = 8;
	static constexpr int nCornerDir = 3;
};

template<>
class Contiguity::BidimensionalContiguityTraits<Contiguity::Rook> {
public:
	static constexpr int nDir = 4;
	static constexpr int nCornerDir = 2;
};

template<>
class Contiguity::BidimensionalContiguityTraits<Contiguity::Bishop> {
public:
	static constexpr int nDir = 4;
	static constexpr int nCornerDir = 1;
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

} // namespace StereoVision

#endif // CONTIGUITY_H
