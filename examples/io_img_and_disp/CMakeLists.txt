
set(EXEC_NAME io_img_and_disp)

configure_file(main.cpp main.cpp)
set(PROG_SRC ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)

add_executable(
    ${EXEC_NAME}
    ${PROG_SRC}
)

target_link_libraries(${EXEC_NAME} ${STEREOVISION_LIB} Qt${QT_VERSION_MAJOR}::Core)
