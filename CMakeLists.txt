cmake_minimum_required(VERSION 3.5)

set(STEREOVISION_LIB stevi)
set(STEREOVISION_LIB ${STEREOVISION_LIB} PARENT_SCOPE)
project(lib${STEREOVISION_LIB} LANGUAGES CXX)

option(buildTests "Build the test suits for the library" ON)
option(buildExamples "Build the examples for the library" ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -march=native -mavx -ftree-vectorize")

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(submodules)
include_directories(${MODULES_INCLUDE_DIRECTORIES})

set(LIB_SRC 
        stevi_global.h
	geometry/core.cpp
	geometry/core.h
	geometry/geometricexception.cpp
	geometry/geometricexception.h
	geometry/alignement.cpp
	geometry/alignement.h
	geometry/imagecoordinates.cpp
	geometry/imagecoordinates.h
	geometry/lensdistortion.cpp
	geometry/lensdistortion.h
	geometry/pointcloudalignment.cpp
	geometry/pointcloudalignment.h
	geometry/rotations.cpp
        geometry/rotations.h
		geometry/stereorigrectifier.cpp
		geometry/stereorigrectifier.h
        interpolation/interpolation.cpp
        interpolation/interpolation.h
        interpolation/lensdistortionsmap.cpp
        interpolation/lensdistortionsmap.h
        interpolation/downsampling.h
        correlation/matching_costs.h
        correlation/correlation_base.h
        correlation/unfold.h
        correlation/unfold.cpp
        correlation/cross_correlations.h
        correlation/cross_correlations.cpp
        correlation/census.h
        correlation/cost_based_refinement.h
        correlation/sgm.h
        correlation/hierarchical.h
        correlation/patchmatch.h
        optimization/l0optimization.h
        optimization/l1optimization.h
        optimization/l2optimization.h
        optimization/sphericaloptimization.h
        optimization/affine_utils.h
        utils/contiguity.h
        utils/margins.h
        utils/indexers.h
        utils/propagation_direction.h
        utils/randomcache.h
        utils/types_manipulations.h
        statistics/stereo_covering.h
        imageProcessing/meanShiftClustering.h
        io/image_io.h
)

add_library(
    stevi
    ${LIB_SRC}
)

target_link_libraries(stevi PRIVATE ${PNP_LIBS})


if (buildTests)

enable_testing()

find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Test REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Test REQUIRED)

set(CMAKE_AUTOMOC ON)
add_subdirectory(test)

endif(buildTests)

if (buildExamples)

    find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Gui REQUIRED)
    find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Gui REQUIRED)

    set(CMAKE_AUTOMOC ON)

add_subdirectory(examples)

endif (buildExamples)
