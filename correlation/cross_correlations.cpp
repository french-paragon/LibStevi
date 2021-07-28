#include "cross_correlations.h"

namespace StereoVision {
namespace Correlation {

const std::string MatchingFunctionTraits<matchingFunctions::NCC>::Name = "NCC";
const std::string MatchingFunctionTraits<matchingFunctions::SSD>::Name = "SSD";
const std::string MatchingFunctionTraits<matchingFunctions::SAD>::Name = "SAD";

} //namespace Correlation
} //namespace StereoVision
