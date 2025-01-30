
set(EXEC_NAME io_remove_attributes)

configure_file(main.cpp main.cpp)
set(PROG_SRC ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)

add_executable(
    ${EXEC_NAME}
    ${PROG_SRC}
)

target_link_libraries(${EXEC_NAME} ${STEREOVISION_LIB})
