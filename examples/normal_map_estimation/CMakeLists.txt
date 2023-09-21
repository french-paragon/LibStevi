set(EXEC_NAME normal_map_estimation)

set(PROG_SRC main.cpp)

add_executable(
    ${EXEC_NAME}
    ${PROG_SRC}
)

target_link_libraries(${EXEC_NAME} ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Core)

if (buildGui)
    add_compile_definitions(-DWITH_GUI)

    target_link_libraries(${EXEC_NAME} Qt${QT_VERSION_MAJOR}::Widgets ${QIMAGEDISPLAYWIDGET_LIB})

endif(buildGui)
