#pnp
set(WITH_PYBIND11 OFF)


if (buildGui)
    set(buildExamples OFF)
    add_subdirectory(qImageDisplayWidget)

    set(QIMAGEDISPLAYWIDGET_LIB ${QIMAGEDISPLAYWIDGET_LIB} PARENT_SCOPE)

endif(buildGui)
