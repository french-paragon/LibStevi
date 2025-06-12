set(EXEC_NAME generic_ray_alignment_evaluation)

set(PROG_SRC main.cpp)

add_executable(
    ${EXEC_NAME}
    ${PROG_SRC}
)

target_link_libraries(${EXEC_NAME} ${STEREOVISION_LIB})
