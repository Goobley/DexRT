cmake_minimum_required(VERSION 3.13)
project(DexRT CXX)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
    message("GOTTEM")
    enable_language(CUDA)
endif()

SET(CMAKE_CXX_FLAGS "${CXX_FLAGS}")
add_subdirectory(YAKL)
if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    set_property(TARGET yakl PROPERTY CUDA_ARCHITECTURES OFF)
endif()
include_directories(YAKL)

add_executable(dexrt "source/main.cpp")
target_include_directories(dexrt SYSTEM PUBLIC "${GCC_INCLUDE_PATH}" "${MPI_INCLUDE_PATH}")
# target_include_directories(dexrt SYSTEM PUBLIC "${MPI_INCLUDE_PATH}")
target_link_libraries(dexrt "${LDFLAGS}")

include(YAKL/yakl_utils.cmake)
yakl_process_target(dexrt)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_target_properties(dexrt PROPERTIES LINKER_LANGUAGE CXX)
    if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
        set_target_properties(dexrt PROPERTIES CUDA_ARCHITECTURES OFF)
    endif()
endif()