cmake_minimum_required(VERSION 3.18)
project(DexRT CXX)

add_subdirectory(fmt)
add_subdirectory(yaml-cpp)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
    enable_language(CUDA)
endif()

SET(CMAKE_CXX_FLAGS "${CXX_FLAGS}")
add_subdirectory(YAKL)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 84)
endif()
include_directories(YAKL)

list(APPEND DEX_SOURCE_FILES
    "source/main.cpp"
    "source/StaticFormalSolution.cpp"
    "source/DynamicFormalSolution.cpp"
    "source/PromweaverBoundary.cpp"
)
list(APPEND DEX_RAY_SOURCE_FILES
    "source/dexrt_ray.cpp"
)

add_executable(dexrt
    "${DEX_SOURCE_FILES}"
)
add_executable(dexrt_ray
    "${DEX_RAY_SOURCE_FILES}"
)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

include(dex_yakl_utils.cmake)

############## dexrt ##############
target_include_directories(dexrt SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${MAGMA_INCLUDE_PATH}"
)
target_link_libraries(dexrt "${LDFLAGS}")
target_link_libraries(dexrt fmt::fmt)
target_link_libraries(dexrt yaml-cpp::yaml-cpp)
target_link_libraries(dexrt argparse)
set_target_properties(dexrt PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES
)
yakl_process_target(dexrt)
if (DEFINED DEXRT_CUDA_ARCHITECTURES)
    set_target_properties(dexrt PROPERTIES CUDA_ARCHITECTURES "${DEXRT_CUDA_ARCHITECTURES}")
endif()
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_target_properties(dexrt PROPERTIES LINKER_LANGUAGE CXX)
endif()

############## dexrt_ray ##############
target_include_directories(dexrt_ray SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${MAGMA_INCLUDE_PATH}"
)
target_link_libraries(dexrt_ray "${LDFLAGS}")
target_link_libraries(dexrt_ray fmt::fmt)
target_link_libraries(dexrt_ray yaml-cpp::yaml-cpp)
target_link_libraries(dexrt_ray argparse)
set_target_properties(dexrt_ray PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES
)
yakl_process_target(dexrt_ray)
if (DEFINED DEXRT_CUDA_ARCHITECTURES)
    set_target_properties(dexrt_ray PROPERTIES CUDA_ARCHITECTURES "${DEXRT_CUDA_ARCHITECTURES}")
endif()
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_target_properties(dexrt_ray PROPERTIES LINKER_LANGUAGE CXX)
endif()