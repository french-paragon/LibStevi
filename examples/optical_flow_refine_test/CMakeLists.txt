set(EXEC_NAME optical_flow_refine_test)

set(PROG_SRC main.cpp)

add_executable(
	${EXEC_NAME}
	${PROG_SRC}
)

target_link_libraries(${EXEC_NAME} ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Core Threads::Threads)

if (buildGui)
	add_compile_definitions(-DWITH_GUI)

        target_compile_definitions(${EXEC_NAME} PRIVATE LIBSTEREOVISION_BUILDING)
	target_link_libraries(${EXEC_NAME} Qt${QT_VERSION_MAJOR}::Widgets ${QIMAGEDISPLAYWIDGET_LIB})

endif(buildGui)
