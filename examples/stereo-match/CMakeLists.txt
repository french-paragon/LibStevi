set(EXEC_NAME stereo-match)

set(PROG_SRC main.cpp)

add_executable(
    ${EXEC_NAME}
    ${PROG_SRC}
)

target_compile_definitions(${EXEC_NAME} PRIVATE LIBSTEREOVISION_BUILDING)
target_link_libraries(${EXEC_NAME} ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Core)

if (buildGui)
    target_compile_definitions(${EXEC_NAME} PUBLIC -DWITH_GUI)

    target_link_libraries(${EXEC_NAME} Qt${QT_VERSION_MAJOR}::Widgets ${QIMAGEDISPLAYWIDGET_LIB})

endif(buildGui)
