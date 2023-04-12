#pnp
set(WITH_PYBIND11 OFF)

add_subdirectory(pnp)

set(PNP_LIBS libpnp PARENT_SCOPE)

set (MODULES_INCLUDE_DIRECTORIES ${MODULES_INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/pnp PARENT_SCOPE)


if (buildGui)
    set(buildExamples OFF)
    add_subdirectory(qImageDisplayWidget)

    set(QIMAGEDISPLAYWIDGET_LIB ${QIMAGEDISPLAYWIDGET_LIB} PARENT_SCOPE)

endif(buildGui)
