#include "cross_correlations.h"

namespace StereoVision {
namespace Correlation {

const std::string MatchingFunctionTraits<matchingFunctions::CC>::Name = "CC";
const std::string MatchingFunctionTraits<matchingFunctions::NCC>::Name = "NCC";
const std::string MatchingFunctionTraits<matchingFunctions::SSD>::Name = "SSD";
const std::string MatchingFunctionTraits<matchingFunctions::SAD>::Name = "SAD";
const std::string MatchingFunctionTraits<matchingFunctions::ZCC>::Name = "ZCC";
const std::string MatchingFunctionTraits<matchingFunctions::ZNCC>::Name = "ZNCC";
const std::string MatchingFunctionTraits<matchingFunctions::ZSSD>::Name = "ZSSD";
const std::string MatchingFunctionTraits<matchingFunctions::ZSAD>::Name = "ZSAD";

} //namespace Correlation
} //namespace StereoVision
