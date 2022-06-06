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
target_link_libraries(benchmarkStereoMatchingModels ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)

add_executable(benchmarkRandomUtils benchmarkRandomUtils.cpp)
target_link_libraries(benchmarkRandomUtils ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
