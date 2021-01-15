#ifndef STEREOVISIONAPP_GEOMETRICEXCEPTION_H
#define STEREOVISIONAPP_GEOMETRICEXCEPTION_H

#include <string>
#include <exception>

namespace StereoVision {
namespace Geometry {

class GeometricException : public std::exception
{
public:
	GeometricException(std::string const& what);
	GeometricException(GeometricException const& other);

	virtual ~GeometricException();

	const char* what() const noexcept override;

protected:

	char* _what;
};

} // namespace Geometry
} // namespace StereoVision

#endif // STEREOVISIONAPP_GEOMETRICEXCEPTION_H
