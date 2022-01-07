
add_executable(testCore testCore.cpp)
target_link_libraries(testCore ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestCore testCore)

add_executable(testReprojection testProjection.cpp)
target_link_libraries(testReprojection ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestReprojection testReprojection)

add_executable(testRotations testRotations.cpp)
target_link_libraries(testRotations ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestRotations testRotations)


add_executable(testPointCloudAlignement testPointCloudAlignement.cpp)
target_link_libraries(testPointCloudAlignement ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestPointCloudAlignement testPointCloudAlignement)


add_executable(testLenseDistortion testLenseDistortion.cpp)
target_link_libraries(testLenseDistortion ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestLenseDistortion testLenseDistortion)


add_executable(testCorrelation testCorrelationFilters.cpp ../test_correlation_utils.h)
target_link_libraries(testCorrelation ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestCorrelation testCorrelation)


add_executable(testCorrelation2d testCorrelation2d.cpp ../test_correlation_utils.h)
target_link_libraries(testCorrelation2d ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestCorrelation2d testCorrelation2d)


add_executable(testCorrelationHierarchical testCorrelationHierarchical.cpp ../test_correlation_utils.h)
target_link_libraries(testCorrelationHierarchical ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestCorrelationHierarchical testCorrelationHierarchical)


add_executable(testL0Optimization testL0Optimization.cpp)
target_link_libraries(testL0Optimization ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestL0Optimization testL0Optimization)


add_executable(testL1Optimization testL1Optimization.cpp)
target_link_libraries(testL1Optimization ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestL1Optimization testL1Optimization)


add_executable(testOptimizationUtils testOptimizationUtils.cpp)
target_link_libraries(testOptimizationUtils ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestOptimizationUtils testOptimizationUtils)


add_executable(testCostRefinement testCostRefinement.cpp)
target_link_libraries(testCostRefinement ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestCostRefinement testCostRefinement)


add_executable(testStereoStatistics testStereoStatistics.cpp)
target_link_libraries(testStereoStatistics ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestStereoStatistics testStereoStatistics)

add_executable(testIndexers testIndexers.cpp)
target_link_libraries(testIndexers ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestIndexers testIndexers)
