
set(EXEC_NAME io_pcd)

configure_file(main.cpp main.cpp)
set(PROG_SRC ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)

add_executable(
    ${EXEC_NAME}
    ${PROG_SRC}
)

target_link_libraries(${EXEC_NAME} ${STEREOVISION_LIB})
