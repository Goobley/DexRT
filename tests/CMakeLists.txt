cmake_minimum_required(VERSION 3.18)
project(DexRTTests CXX)
enable_testing()

add_subdirectory(../fmt ./fmt)
add_subdirectory(../yaml-cpp ./yaml-cpp)

get_filename_component(YAKL_PATH "../YAKL" ABSOLUTE)
add_subdirectory(${YAKL_PATH} ./YAKL)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 84)
    endif()
endif()
include_directories(${YAKL_PATH})

list(APPEND UNIT_TEST_SOURCE_FILES
    test_raymarch.cpp
    test_crtaf.cpp
)

add_executable(dexrt_test 
    catch_amalgamated.cpp
    ${UNIT_TEST_SOURCE_FILES}
)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_source_files_properties(${UNIT_TEST_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
target_include_directories(dexrt_test SYSTEM PUBLIC "${GCC_INCLUDE_PATH}" "${MPI_INCLUDE_PATH}")
target_include_directories(dexrt_test PUBLIC "../source/")
target_include_directories(dexrt_test PUBLIC "../YAKL/src")
target_include_directories(dexrt_test PUBLIC "../yaml-cpp/include")
target_link_libraries(dexrt_test "${LDFLAGS}")
target_link_libraries(dexrt_test yaml-cpp::yaml-cpp)
set_target_properties(dexrt_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES
)
include(../dex_yakl_utils.cmake)
yakl_process_target(dexrt_test)

add_executable(lw_comparison_2d
    demos/lightweaver_comparison_2d/main.cpp
)
target_include_directories(lw_comparison_2d SYSTEM PUBLIC 
    "${GCC_INCLUDE_PATH}" 
    "${MPI_INCLUDE_PATH}" 
    "${NETCDF_INCLUDE_PATH}"
)
target_include_directories(lw_comparison_2d PUBLIC "../source/")
target_include_directories(lw_comparison_2d PUBLIC "../YAKL/src")
target_include_directories(lw_comparison_2d PUBLIC ../fmt)
target_link_libraries(lw_comparison_2d "${LDFLAGS}")
target_link_libraries(lw_comparison_2d fmt::fmt)
target_link_libraries(lw_comparison_2d yaml-cpp::yaml-cpp)
set_target_properties(lw_comparison_2d PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES
)
yakl_process_target(lw_comparison_2d)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_target_properties(dexrt_test PROPERTIES LINKER_LANGUAGE CXX)
endif()

add_test(NAME dexrt_unit_test COMMAND $<TARGET_FILE:dexrt_test>)