<<<<<<< HEAD
set(EXEC_NAME io_sdc_read)

set(PROG_SRC main.cpp)
=======

set(EXEC_NAME io_sdc_read)

configure_file(main.cpp main.cpp)
set(PROG_SRC ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)
>>>>>>> 9d9c74f5471b06134e732c86c48a73fa01005c1b

add_executable(
    ${EXEC_NAME}
    ${PROG_SRC}
)

target_link_libraries(${EXEC_NAME} ${STEREOVISION_LIB})
