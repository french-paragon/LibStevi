#include "geometricexception.h"

#include <cstring>

namespace StereoVision {
namespace Geometry {

GeometricException::GeometricException(std::string const& what)
{
	_what = new char[what.size()];
	memcpy (_what, what.c_str(), what.size());
}


GeometricException::GeometricException(GeometricException const& other) :
	GeometricException(other.what())
{

}

GeometricException::~GeometricException() {
	delete _what;
}

const char* GeometricException::what() const noexcept
{
	return _what;
}

} // namespace Geometry
} // namespace StereoVision
