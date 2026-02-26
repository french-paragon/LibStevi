
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


add_executable(testLeastMedianOptimization testLeastMedianOptimization.cpp)
target_link_libraries(testLeastMedianOptimization ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestLeastMedianOptimization testLeastMedianOptimization)



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

add_executable(testStereoRigRectifier testStereoRigRectifier.cpp)
target_link_libraries(testStereoRigRectifier ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestStereoRigRectifier testStereoRigRectifier)

add_executable(testImageClustering testImageClusteringAlgorithms.cpp)
target_link_libraries(testImageClustering ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestImageClustering testImageClustering)

add_executable(testImageIO testImageIO.cpp)
target_link_libraries(testImageIO ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test Threads::Threads)
if (TIFF_FOUND)
        target_compile_definitions(testImageIO PRIVATE STEVI_IO_USE_TIFF)
endif (TIFF_FOUND)

add_test(TestImageIO testImageIO)

add_executable(testFiniteDifferences testFiniteDifferences.cpp)
target_link_libraries(testFiniteDifferences ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestFiniteDifferences testFiniteDifferences)

add_executable(testCheckboardDetection testCheckboardDetection.cpp)
target_link_libraries(testCheckboardDetection ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestCheckboardDetection testCheckboardDetection)

add_executable(testHexTargetDetection testHexTargetDetection.cpp)
target_link_libraries(testHexTargetDetection ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test Threads::Threads)
add_test(TestHexTargetDetection testHexTargetDetection)

add_executable(testConnectedComponents testConnectedComponents.cpp)
target_link_libraries(testConnectedComponents ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestConnectedComponents testConnectedComponents)

add_executable(testMatchPattern testMatchPattern.cpp)
target_link_libraries(testMatchPattern ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestMatchPattern testMatchPattern)

add_executable(testOnDemandCostVolume testOnDemandCostVolume.cpp)
target_link_libraries(testOnDemandCostVolume ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestOnDemandCostVolume testOnDemandCostVolume)

add_executable(testGraphs testGraphs.cpp)
target_link_libraries(testGraphs ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestGraphs testGraphs)

add_executable(testSegmentation testSegmentation.cpp)
target_link_libraries(testSegmentation ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestSegmentation testSegmentation)

add_executable(testConvolutions testConvolution.cpp)
target_link_libraries(testConvolutions ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestConvolutions testConvolutions)

add_executable(testPixelsSets testPixelsSets.cpp)
target_link_libraries(testPixelsSets ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestPixelsSets testPixelsSets)

add_executable(testPCA testPCA.cpp)
target_link_libraries(testPCA ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestPCA testPCA)

add_executable(testSparseMatchingUtils testSparseMatchingUtils.cpp)
target_link_libraries(testSparseMatchingUtils ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestSparseMatchingUtils testSparseMatchingUtils)

add_executable(testAssignementProblems testAssignementProblems.cpp)
target_link_libraries(testAssignementProblems ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestAssignementProblems testAssignementProblems)

add_executable(testInterpolation testInterpolation.cpp)
target_link_libraries(testInterpolation ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestInterpolation testInterpolation)

add_executable(testPartitionTrees testPartitionTrees.cpp)
target_link_libraries(testPartitionTrees ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestPartitionTrees testPartitionTrees)

add_executable(testTypesUtils testTypesUtils.cpp)
target_link_libraries(testTypesUtils ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestTypesUtils testTypesUtils)

configure_file(testPointCloudIO.cpp testPointCloudIO.cpp)
add_executable(testPointCloudIO ${CMAKE_CURRENT_BINARY_DIR}/testPointCloudIO.cpp)
target_link_libraries(testPointCloudIO ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test Qt${QT_VERSION_MAJOR}::Core)
add_test(TestPointCloudIO testPointCloudIO)

add_executable(testInpaintingAlgorithms testInpaintingAlgorithms.cpp)
target_link_libraries(testInpaintingAlgorithms ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestInpaintingAlgorithms testInpaintingAlgorithms)

add_executable(testCovarianceKernels testCovarianceKernels.cpp)
target_link_libraries(testCovarianceKernels ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestCovarianceKernels testCovarianceKernels)

add_executable(testCombinatorial testCombinatorial.cpp)
target_link_libraries(testCombinatorial ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Test)
add_test(TestCombinatorial testCombinatorial)



