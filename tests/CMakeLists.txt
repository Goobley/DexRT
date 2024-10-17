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

list(APPEND DEX_SOURCE_FILES
    "../source/StaticFormalSolution.cpp"
    "../source/DynamicFormalSolution.cpp"
)

list(APPEND UNIT_TEST_SOURCE_FILES
    test_raymarch.cpp
    test_crtaf.cpp
    test_lte.cpp
    test_voigt.cpp
    test_utils.cpp
    test_emis_opac.cpp
    test_gesv.cpp
    "../source/MiscSparse.cpp"
)

add_executable(dexrt_test
    catch_amalgamated.cpp
    ${UNIT_TEST_SOURCE_FILES}
    ${DEX_SOURCE_FILES}
)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_source_files_properties(${UNIT_TEST_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
target_include_directories(dexrt_test SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${MAGMA_INCLUDE_PATH}"
)
target_include_directories(dexrt_test PUBLIC "../source/")
target_include_directories(dexrt_test PUBLIC "../YAKL/src")
target_include_directories(dexrt_test PUBLIC "../yaml-cpp/include")
target_include_directories(dexrt_test PUBLIC ../fmt/include)
target_link_libraries(dexrt_test "${LDFLAGS}")
target_link_libraries(dexrt_test fmt::fmt)
target_link_libraries(dexrt_test yaml-cpp::yaml-cpp)
set_target_properties(dexrt_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES
)
include(../dex_yakl_utils.cmake)
yakl_process_target(dexrt_test)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_target_properties(dexrt_test PROPERTIES LINKER_LANGUAGE CXX)
endif()

#############################

# add_executable(lw_comparison_2d
#     demos/lightweaver_comparison_2d/main.cpp
#     ${DEX_SOURCE_FILES}
# )
# target_include_directories(lw_comparison_2d SYSTEM PUBLIC
#     "${GCC_INCLUDE_PATH}"
#     "${MPI_INCLUDE_PATH}"
#     "${NETCDF_INCLUDE_PATH}"
#     "${MAGMA_INCLUDE_PATH}"
# )
# target_include_directories(lw_comparison_2d PUBLIC "../source/")
# target_include_directories(lw_comparison_2d PUBLIC "../YAKL/src")
# target_include_directories(lw_comparison_2d PUBLIC ../fmt/include)
# target_link_libraries(lw_comparison_2d "${LDFLAGS}")
# target_link_libraries(lw_comparison_2d fmt::fmt)
# target_link_libraries(lw_comparison_2d yaml-cpp::yaml-cpp)
# set_target_properties(lw_comparison_2d PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS YES
# )
# yakl_process_target(lw_comparison_2d)
# if ("${YAKL_ARCH}" STREQUAL "CUDA")
#     set_target_properties(lw_comparison_2d PROPERTIES LINKER_LANGUAGE CXX)
# endif()

# #############################

# add_executable(diffuse_source
#     demos/diffuse_source/main.cpp
#     ${DEX_SOURCE_FILES}
# )
# target_include_directories(diffuse_source SYSTEM PUBLIC
#     "${GCC_INCLUDE_PATH}"
#     "${MPI_INCLUDE_PATH}"
#     "${NETCDF_INCLUDE_PATH}"
#     "${MAGMA_INCLUDE_PATH}"
# )
# target_include_directories(diffuse_source PUBLIC "../source/")
# target_include_directories(diffuse_source PUBLIC "../YAKL/src")
# target_include_directories(diffuse_source PUBLIC ../fmt/include)
# target_link_libraries(diffuse_source "${LDFLAGS}")
# target_link_libraries(diffuse_source fmt::fmt)
# target_link_libraries(diffuse_source yaml-cpp::yaml-cpp)
# set_target_properties(diffuse_source PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS YES
# )
# yakl_process_target(diffuse_source)
# if ("${YAKL_ARCH}" STREQUAL "CUDA")
#     set_target_properties(diffuse_source PROPERTIES LINKER_LANGUAGE CXX)
# endif()

# #############################

# add_executable(solid_source
#     demos/solid_source/main.cpp
#     ${DEX_SOURCE_FILES}
# )
# target_include_directories(solid_source SYSTEM PUBLIC
#     "${GCC_INCLUDE_PATH}"
#     "${MPI_INCLUDE_PATH}"
#     "${NETCDF_INCLUDE_PATH}"
#     "${MAGMA_INCLUDE_PATH}"
# )
# target_include_directories(solid_source PUBLIC "../source/")
# target_include_directories(solid_source PUBLIC "../YAKL/src")
# target_include_directories(solid_source PUBLIC ../fmt/include)
# target_link_libraries(solid_source "${LDFLAGS}")
# target_link_libraries(solid_source fmt::fmt)
# target_link_libraries(solid_source yaml-cpp::yaml-cpp)
# set_target_properties(solid_source PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS YES
# )
# yakl_process_target(solid_source)
# if ("${YAKL_ARCH}" STREQUAL "CUDA")
#     set_target_properties(solid_source PROPERTIES LINKER_LANGUAGE CXX)
# endif()

# #############################

# add_executable(static_lte_fs
#     demos/static_lte_fs/main.cpp
#     ${DEX_SOURCE_FILES}
# )
# target_include_directories(static_lte_fs SYSTEM PUBLIC
#     "${GCC_INCLUDE_PATH}"
#     "${MPI_INCLUDE_PATH}"
#     "${NETCDF_INCLUDE_PATH}"
#     "${MAGMA_INCLUDE_PATH}"
# )
# target_include_directories(static_lte_fs PUBLIC "../source/")
# target_include_directories(static_lte_fs PUBLIC "../YAKL/src")
# target_include_directories(static_lte_fs PUBLIC ../fmt/include)
# target_link_libraries(static_lte_fs "${LDFLAGS}")
# target_link_libraries(static_lte_fs fmt::fmt)
# target_link_libraries(static_lte_fs yaml-cpp::yaml-cpp)
# set_target_properties(static_lte_fs PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS YES
# )
# yakl_process_target(static_lte_fs)
# if ("${YAKL_ARCH}" STREQUAL "CUDA")
#     set_target_properties(static_lte_fs PROPERTIES LINKER_LANGUAGE CXX)
# endif()

# #############################

add_test(NAME dexrt_unit_test COMMAND $<TARGET_FILE:dexrt_test>)