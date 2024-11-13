cmake_minimum_required(VERSION 3.18)
project(DexRT CXX)

add_subdirectory(fmt)
add_subdirectory(yaml-cpp)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
    enable_language(CUDA)
endif()

include("CheckGit.cmake")
CheckGitSetup()

SET(CMAKE_CXX_FLAGS "${CXX_FLAGS}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_subdirectory(YAKL)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 84)
endif()
include_directories(YAKL)

list(APPEND DEX_SOURCE_FILES
    "source/main.cpp"
    "source/StaticFormalSolution.cpp"
    "source/DynamicFormalSolution.cpp"
    "source/CascadeState.cpp"
    "source/PromweaverBoundary.cpp"
    "source/MiscSparse.cpp"
    "source/Collisions.cpp"
    "source/ProfileNormalisation.cpp"
    "source/Populations.cpp"
    "source/ChargeConservation.cpp"
    "source/Mipmaps.cpp"
    "source/DirectionalEmisOpacInterp.cpp"
    "source/CoreAndVoigtEmisOpac.cpp"
)
list(APPEND DEX_RAY_SOURCE_FILES
    "source/dexrt_ray.cpp"
    "source/PromweaverBoundary.cpp"
    "source/MiscSparse.cpp"
)

add_executable(dexrt
    "${DEX_SOURCE_FILES}"
)
add_executable(dexrt_ray
    "${DEX_RAY_SOURCE_FILES}"
)
# NOTE(cmo): Override language type for cuda, may need to do this for HIP too.
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS}")
# NOTE(cmo): Storing all the files in a list seems to prevent the LANGUAGE attribute for being set
else()
    set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES LANGUAGE CXX)
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
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_link_libraries(dexrt "${LDLIBS}")
target_link_libraries(dexrt fmt::fmt)
target_link_libraries(dexrt yaml-cpp::yaml-cpp)
target_link_libraries(dexrt argparse)
target_link_libraries(dexrt git_version)
set_target_properties(dexrt PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINK_FLAGS "${LDFLAGS}"
)
yakl_process_target(dexrt)
get_source_file_property(flags "source/main.cpp" COMPILE_FLAGS)
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
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_link_libraries(dexrt_ray "${LDLIBS}")
target_link_libraries(dexrt_ray fmt::fmt)
target_link_libraries(dexrt_ray yaml-cpp::yaml-cpp)
target_link_libraries(dexrt_ray argparse)
target_link_libraries(dexrt_ray git_version)
set_target_properties(dexrt_ray PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINK_FLAGS "${LDFLAGS}"
)
yakl_process_target(dexrt_ray)
if (DEFINED DEXRT_CUDA_ARCHITECTURES)
    set_target_properties(dexrt_ray PROPERTIES CUDA_ARCHITECTURES "${DEXRT_CUDA_ARCHITECTURES}")
endif()
if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_target_properties(dexrt_ray PROPERTIES LINKER_LANGUAGE CXX)
endif()
