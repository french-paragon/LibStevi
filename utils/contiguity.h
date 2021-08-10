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

	constexpr static int nDirections(bidimensionalContiguity contiguity) {
		return (contiguity == bidimensionalContiguity::Queen) ? 8 : 4;
	}

	constexpr static int nCornerDirections(bidimensionalContiguity contiguity) {
		return (contiguity == bidimensionalContiguity::Queen) ? 3 : ((contiguity == bidimensionalContiguity::Rook) ? 2 : 1);
	}

	template<bidimensionalContiguity contiguity>
	constexpr static std::array<std::array<int, 2>, nDirections(contiguity)> getDirections() {

		typedef std::array<int, 2> deltasArrayType;
		typedef std::array<deltasArrayType, nDirections(contiguity)> returnArrayType;

		return (contiguity == bidimensionalContiguity::Queen) ?
					returnArrayType({deltasArrayType({1,1}),
									 deltasArrayType({1,0}),
									 deltasArrayType({1,-1}),
									 deltasArrayType({0,1}),
									 deltasArrayType({0,-1}),
									 deltasArrayType({-1,1}),
									 deltasArrayType({-1,0}),
									 deltasArrayType({-1,-1})}) :
					((contiguity == bidimensionalContiguity::Rook) ?
						 returnArrayType({deltasArrayType({1,0}),
										  deltasArrayType({0,1}),
										  deltasArrayType({0,-1}),
										  deltasArrayType({-1,0})}) :
						 returnArrayType({deltasArrayType({1,1}),
										  deltasArrayType({1,-1}),
										  deltasArrayType({-1,1}),
										  deltasArrayType({-1,-1})})
					);
	}

	template<bidimensionalContiguity contiguity>
	constexpr static std::array<std::array<int, 2>, nCornerDirections(contiguity)> getCornerDirections() {

		typedef std::array<int, 2> deltasArrayType;
		typedef std::array<deltasArrayType, nCornerDirections(contiguity)> returnArrayType;

		return (contiguity == bidimensionalContiguity::Queen) ?
					returnArrayType({deltasArrayType({1,1}), deltasArrayType({1,0}), deltasArrayType({0,1})}) :
					((contiguity == bidimensionalContiguity::Rook) ?
						 returnArrayType({deltasArrayType({1,0}), deltasArrayType({0,1})}) :
						 returnArrayType({deltasArrayType({1,1})})
					);
	}

};

} // namespace StereoVision

#endif // CONTIGUITY_H
