cmake_minimum_required(VERSION 3.5)

include(CheckCXXCompilerFlag)
get_directory_property(hasParent PARENT_DIRECTORY)

set(STEREOVISION_LIB stevi)
if(hasParent)
    set(STEREOVISION_LIB ${STEREOVISION_LIB} PARENT_SCOPE)
endif()
project(lib${STEREOVISION_LIB} LANGUAGES CXX)

#
include(CMakePackageConfigHelpers)

include(GNUInstallDirs)

set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
set(CMAKE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})

set(STEREOVISION_NAME "StereoVision")
set(STEREOVISION_EXPORTNAME "StereoVisionTargets")

option(buildTests "Build the test suits for the library" ON)
option(buildExamples "Build the examples for the library" ON)
option(buildGui "Build some additional tools for building gui with Qt" OFF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -march=native -ftree-vectorize")

#SIMD for x86
check_cxx_compiler_flag("-mavx" AVXSUPPORTED)
check_cxx_compiler_flag("-mavx2" AVX2SUPPORTED)
check_cxx_compiler_flag("-mfma" FMASUPPORTED)

if (AVXSUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx")
endif()

if (AVX2SUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")
endif()

if (FMASUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mfma")
endif()

#SIMD for ARM not supported yet

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(JPEG)
find_package(PNG)
find_package(TIFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_DEBUG_POSTFIX d)

add_subdirectory(submodules)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/submodules)
include_directories(${MODULES_INCLUDE_DIRECTORIES})

set(LIB_SRC
        stevi_global.h
        geometry/core.h
        geometry/geometricexception.cpp
        geometry/geometricexception.h
        geometry/alignement.h
        geometry/imagecoordinates.h
        geometry/lensdistortion.h
        geometry/pointcloudalignment.cpp
        geometry/pointcloudalignment.h
        geometry/genericraysalignement.h
        geometry/rotations.h
        geometry/posesmath.h
        geometry/stereorigrectifier.cpp
        geometry/stereorigrectifier.h
        geometry/genericbinarypartitioningtree.h
        graphs/graphs.h
        graphs/doublyConnectedEdgeLists.h
        graphs/graph_flow.h
        graphs/graph_cut.h
        optimization/assignement_problems.h
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
        correlation/image_based_refinement.h
        correlation/sgm.h
        correlation/hierarchical.h
        correlation/patchmatch.h
        correlation/on_demand_cost_volume.h
        correlation/fast_approximate_matching.h
        correlation/template_matching.h
        correlation/dynamic_programing_stereo.h
        correlation/disparity_plus_background_segmentation.h
        correlation/guided_cost_filtering.h
        optimization/l0optimization.h
        optimization/l1optimization.h
        optimization/l2optimization.h
        optimization/huber_kernel.h
        optimization/leastmedianoptimization.h
        optimization/sphericaloptimization.h
        optimization/affine_utils.h
        optimization/principalComponentsAnalysis.h
        optimization/gaussnewtownproblem.h
        utils/contiguity.h
        utils/margins.h
        utils/indexers.h
        utils/propagation_direction.h
        utils/randomcache.h
        utils/types_manipulations.h
        utils/colors.h
        utils/array_utils.h
        utils/stevimath.h
        utils/hash_utils.h
        utils/iterative_numerical_algorithm_output.h
        statistics/stereo_covering.h
        imageProcessing/colorConversions.h
        imageProcessing/histogram.h
        imageProcessing/meanShiftClustering.h
        imageProcessing/intrinsicImageDecomposition.h
        imageProcessing/checkBoardDetection.h
        imageProcessing/checkBoardDetection.cpp
        imageProcessing/hexagonalRGBTargetsDetection.h
        imageProcessing/finiteDifferences.h
        imageProcessing/connectedComponents.h
        imageProcessing/morphologicalOperators.h
        imageProcessing/foregroundSegmentation.h
        imageProcessing/shapeFromShading.h
        imageProcessing/convolutions.h
        imageProcessing/standardConvolutionFilters.h
        imageProcessing/pixelsLines.h
        imageProcessing/pixelsTriangles.h
        imageProcessing/edgesDetection.h
        io/image_io.h
        io/image_io.cpp
        io/pointcloud_io.h
        io/pointcloud_io.cpp
        io/sdc_pointcloud_io.h
        io/sdc_pointcloud_io.cpp
        io/pcd_pointcloud_io.h
        io/pcd_pointcloud_io.cpp
        sparseMatching/cornerDetectors.h
        sparseMatching/nonLocalMaximumPointSelection.h
        sparseMatching/pointsOrientation.h
        sparseMatching/pointsDescriptors.h
)

if (buildGui)
    set(LIB_SRC ${LIB_SRC} gui/arraydisplayadapter.h)
endif(buildGui)

add_library(
    ${STEREOVISION_LIB}
    ${LIB_SRC}
)

target_compile_definitions(${STEREOVISION_LIB} PRIVATE LIBSTEREOVISION_BUILDING)

if (JPEG_FOUND)
    message("Using native jpg library")
    target_link_libraries(${STEREOVISION_LIB} PUBLIC JPEG::JPEG)
    target_compile_definitions(stevi PRIVATE STEVI_IO_USE_JPEG)
endif (JPEG_FOUND)

if (PNG_FOUND)
    message("Using native png library")
    target_link_libraries(${STEREOVISION_LIB} PUBLIC PNG::PNG)
    target_compile_definitions(stevi PRIVATE STEVI_IO_USE_PNG)
endif (PNG_FOUND)

if (TIFF_FOUND)
    message("Using native tiff library")
    target_link_libraries(${STEREOVISION_LIB} PUBLIC TIFF::TIFF)
    target_compile_definitions(stevi PRIVATE STEVI_IO_USE_TIFF)
endif (TIFF_FOUND)

if (buildGui)
    find_package(QT NAMES Qt5 COMPONENTS Widgets REQUIRED)
    find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)
    target_link_libraries(stevi PUBLIC Qt${QT_VERSION_MAJOR}::Widgets)
endif(buildGui)

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


install(TARGETS ${STEREOVISION_LIB}
          EXPORT ${STEREOVISION_EXPORTNAME}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )

if (buildGui)
  install(TARGETS ${QIMAGEDISPLAYWIDGET_LIB}
            EXPORT ${STEREOVISION_EXPORTNAME}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
endif()

foreach(FILE IN LISTS LIB_SRC)
        if(FILE MATCHES "^((.*)/)?([^/]+)\\.h$")
                install(FILES ${FILE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${STEREOVISION_NAME}/${CMAKE_MATCH_2})
        endif()
endforeach()

if (buildGui)
    get_target_property(QDISPLAYWIDGET_SRC_LIST ${QIMAGEDISPLAYWIDGET_LIB} SOURCES)
    get_target_property(QDISPLAYWIDGET_SOURCE_DIR ${QIMAGEDISPLAYWIDGET_LIB} SOURCE_DIR)
    foreach(FILE IN LISTS QDISPLAYWIDGET_SRC_LIST)
            if(FILE MATCHES "^((.*)/)?([^/]+)\\.h$")
                    install(FILES ${QDISPLAYWIDGET_SOURCE_DIR}/${FILE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${STEREOVISION_NAME}/QImageDisplayWidget)
            endif()
    endforeach()
endif()


install(
  EXPORT ${STEREOVISION_EXPORTNAME}
  DESTINATION ${CMAKE_INSTALL_CMAKEDIR}/cmake/${STEREOVISION_NAME}
  NAMESPACE ${STEREOVISION_NAME}::
  FILE ${STEREOVISION_NAME}Targets.cmake # ensure the name is different than the AlineConfig.cmake file
  )

configure_package_config_file(
  "cmake/Config.cmake.in"
  "${STEREOVISION_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_CMAKEDIR}/cmake/${STEREOVISION_NAME}
  PATH_VARS
    CMAKE_INSTALL_LIBDIR
  )

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${STEREOVISION_NAME}ConfigVersion.cmake"
  VERSION "0.0"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${STEREOVISION_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${STEREOVISION_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_CMAKEDIR}/cmake/${STEREOVISION_NAME} )
