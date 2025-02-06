cmake_minimum_required(VERSION 3.18)
if ("${DEX_ARCH}" STREQUAL "CUDA")
    project(DexRT LANGUAGES CXX CUDA)
else ()
    project(DexRT LANGUAGES CXX)
endif()

add_subdirectory(fmt)
add_subdirectory(yaml-cpp)
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

include("CheckGit.cmake")
CheckGitSetup()

SET(CMAKE_CXX_FLAGS "${CXX_FLAGS}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED GLOBAL)
find_package(KokkosKernels)
if (${DEXRT_USE_MPI})
    find_package(MPI REQUIRED)
endif()

add_library(yakl INTERFACE)
target_include_directories(yakl INTERFACE YAKL/src YAKL/external YAKL/src/extensions)
set(YAKL_DEFS "")
if (${YAKL_HAVE_MPI} OR ${DEXRT_USE_MPI})
    set(YAKL_DEFS "${YAKL_DEFS} -DHAVE_MPI")
endif()
if (${YAKL_AUTO_PROFILE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_AUTO_PROFILE")
endif()
if (${YAKL_PROFILE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_PROFILE")
endif()
if (${YAKL_AUTO_FENCE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_AUTO_FENCE")
endif()


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
    "source/NgAcceleration.cpp"
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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if ("${DEX_ARCH}" STREQUAL "CUDA")
    set(CMAKE_CXX_STANDARD 20)
endif()

# NOTE(cmo): Override language type for cuda, may need to do this for HIP too.
set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
if ("${DEX_ARCH}" STREQUAL "CUDA")
    set(DEX_EXTRA_CUDA_STD_FLAG "")
    if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
        set(DEX_EXTRA_CUDA_STD_FLAG "-std=c++20")
    endif()
    set_target_properties(dexrt PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})
    set_target_properties(dexrt_ray PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})

    set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${DEX_EXTRA_CUDA_STD_FLAG}")
endif()

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
target_link_libraries(dexrt Kokkos::kokkos)
target_link_libraries(dexrt yakl)
if (KokkosKernels_FOUND)
    target_link_libraries(dexrt Kokkos::kokkoskernels)
endif()
if (${DEXRT_USE_MPI})
    target_include_directories(dexrt PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(dexrt MPI::MPI_CXX)
    target_compile_definitions(dexrt PUBLIC HAVE_MPI)
endif()

set_target_properties(dexrt PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)

message("Compiling with: ${CMAKE_CXX_FLAGS} ${YAKL_DEFS} ${DEX_CXX_FLAGS}")
############# dexrt_ray ##############
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
target_link_libraries(dexrt_ray Kokkos::kokkos)
target_link_libraries(dexrt_ray yakl)
if (KokkosKernels_FOUND)
    target_link_libraries(dexrt_ray Kokkos::kokkoskernels)
endif()
set_target_properties(dexrt_ray PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)