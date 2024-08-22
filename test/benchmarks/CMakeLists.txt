add_executable(benchmarkCrossCorrelationAlgorithms benchmarkCrossCorrelationAlgorithms.cpp ../test_correlation_utils.h)
target_link_libraries(benchmarkCrossCorrelationAlgorithms ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)

add_executable(benchmarkUnfoldOperators benchmarkUnfoldOperators.cpp)
target_link_libraries(benchmarkUnfoldOperators ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)

add_executable(benchmarkNormSolverAlgorithms benchmarkNormSolverAlgorithms.cpp)
target_link_libraries(benchmarkNormSolverAlgorithms ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)

add_executable(benchmarkL0ApproximationFunctions benchmarkL0ApproximationFunctions.cpp)
target_link_libraries(benchmarkL0ApproximationFunctions ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)

configure_file(benchmarkStereoMatchingModels.cpp benchmarkStereoMatchingModels.cpp)
add_executable(benchmarkStereoMatchingModels ${CMAKE_CURRENT_BINARY_DIR}/benchmarkStereoMatchingModels.cpp)
target_link_libraries(benchmarkStereoMatchingModels ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test Threads::Threads)

add_executable(benchmarkRandomUtils benchmarkRandomUtils.cpp)
target_link_libraries(benchmarkRandomUtils ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)

add_executable(benchmarkImageSaving benchmarkImageSaving.cpp)
target_link_libraries(benchmarkImageSaving ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test Threads::Threads)

if (TARGET JPEG::JPEG)
    target_link_libraries(benchmarkImageSaving JPEG::JPEG)
    target_compile_definitions(benchmarkImageSaving PUBLIC JPEGAVAILABLE)
endif()
if (TARGET JPEG)
    target_link_libraries(benchmarkImageSaving JPEG)
    target_compile_definitions(benchmarkImageSaving PUBLIC JPEGAVAILABLE)
endif()

if (TARGET PNG::PNG)
    target_link_libraries(benchmarkImageSaving PNG::PNG)
    target_compile_definitions(benchmarkImageSaving PUBLIC PNGAVAILABLE)
endif()
if (TARGET PNG)
    target_link_libraries(benchmarkImageSaving PNG)
    target_compile_definitions(benchmarkImageSaving PUBLIC PNGAVAILABLE)
endif()

if (TARGET TIFF::TIFF)
    target_link_libraries(benchmarkImageSaving TIFF::TIFF)
    target_compile_definitions(benchmarkImageSaving PUBLIC TIFFAVAILABLE)
endif()
if (TARGET TIFF)
    target_link_libraries(benchmarkImageSaving TIFF)
    target_compile_definitions(benchmarkImageSaving PUBLIC TIFFAVAILABLE)
endif()

add_executable(benchmarkForegroundSegmentation benchmarkForegroundSegmentation.cpp)
target_link_libraries(benchmarkForegroundSegmentation ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)

add_executable(benchmarkRaysAlignementAlgorithms benchmarkRaysAlignementAlgorithms.cpp)
target_link_libraries(benchmarkRaysAlignementAlgorithms ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
