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
find_package(KokkosKernels REQUIRED)
if ("${YAKL_HAVE_MPI}" OR "${DEXRT_USE_MPI}")
    set(DEXRT_USE_MPI "On")
    set(YAKL_DEFS "${YAKL_DEFS} -DHAVE_MPI")
    find_package(MPI REQUIRED)
endif()

add_library(yakl INTERFACE)
target_include_directories(yakl INTERFACE YAKL/src YAKL/external YAKL/src/extensions)
set(YAKL_DEFS "")
if ("${DEXRT_USE_MPI}")
    set(YAKL_DEFS "${YAKL_DEFS} -DHAVE_MPI")
endif()
# NOTE(cmo): dexrt_sc's parallel output (output.io: Parallel) needs netCDF-4
# built with parallel HDF5, against the same MPI.
if ("${DEXRT_USE_MPI}")
    execute_process(COMMAND nc-config --has-parallel4 OUTPUT_VARIABLE DEX_NC_PAR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if ("${DEX_NC_PAR}" STREQUAL "yes")
        set(YAKL_DEFS "${YAKL_DEFS} -DDEXRT_HAVE_NC_PAR")
        message(STATUS "dexrt_sc: parallel netCDF output available")
    endif()
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
option(DEX_FORCE_DOUBLE_PREC "Compile with fp_t = double" OFF)
option(DEX_HALF_RADIATION_STORAGE "Store radiation arrays (J, wphi) in 16 bits" ON)
if (${DEX_FORCE_DOUBLE_PREC})
    set(DEX_CXX_FLAGS "${DEX_CXX_FLAGS} -DDEXRT_FORCE_DOUBLE_PREC")
endif()
if (${DEX_HALF_RADIATION_STORAGE})
    set(DEX_CXX_FLAGS "${DEX_CXX_FLAGS} -DDEXRT_HALF_RADIATION_STORAGE")
endif()

list(APPEND DEX_2D_SOURCE_FILES
    "source/ScCells.cpp"
    "source/ScIo.cpp"
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
    "source/LineSweepSetup.cpp"
    "source/BlockMap.cpp"
    "source/ProbesToCompute.cpp"
    "source/InitialPops.cpp"
    "source/BackgroundOpacity.cpp"
)
list(APPEND DEX_SOURCE_FILES
    "source/main.cpp"
    "source/StateSetup.cpp"
    ${DEX_2D_SOURCE_FILES}
)
list(APPEND DEX_RE_SOURCE_FILES
    "source/main_rad_eq.cpp"
    ${DEX_2D_SOURCE_FILES}
)
# NOTE(cmo): dexrt_sc and dexrt_sc_3d share every SC object but the main.
list(APPEND DEX_SC_COMMON_SOURCE_FILES
    "source/StateSetup.cpp"
    "source/ScStateSetup.cpp"
    "source/ScAngularQuadrature.cpp"
    "source/ScRadiation.cpp"
    "source/ScDecomposition.cpp"
    "source/ScHalo.cpp"
    "source/ScMediumCoefficients.cpp"
    "source/ScDynamicScheduler.cpp"
    "source/ScFormalSolution.cpp"
    "source/ScFormal.cpp"
    "source/ScMedium.cpp"
    "source/ScPreparedGeometry.cpp"
    "source/ScLongerFormal.cpp"
    "source/ScBoundary.cpp"
    "source/ScShortFormal.cpp"
    "source/ScPrd.cpp"
    ${DEX_2D_SOURCE_FILES}
)
list(APPEND DEX_SC_SOURCE_FILES
    "source/ScMain.cpp"
    ${DEX_SC_COMMON_SOURCE_FILES}
)
list(APPEND DEX_SC_3D_SOURCE_FILES
    "source/ScMain3d.cpp"
)
list(APPEND DEX_3D_SOURCE_FILES
    "source/main_3d.cpp"
    "source/StaticFormalSolution3d.cpp"
    "source/CascadeState3d.cpp"
    "source/BlockMap.cpp"
    "source/Mipmaps3d.cpp"
    "source/CoreAndVoigtEmisOpac.cpp"
    "source/ProbesToCompute.cpp"
    "source/DynamicFormalSolution3d.cpp"
    "source/MiscSparse.cpp"
    "source/Populations.cpp"
    "source/PromweaverBoundary.cpp"
    "source/ProfileNormalisation.cpp"
    "source/Collisions.cpp"
    "source/NgAcceleration.cpp"
    "source/ChargeConservation.cpp"
    "source/InitialPops.cpp"
)
list(APPEND DEX_RAY_SOURCE_FILES
    "source/dexrt_ray.cpp"
    "source/PromweaverBoundary.cpp"
    "source/MiscSparse.cpp"
    "source/BlockMap.cpp"
)
list(APPEND DEX_RAY_3D_SOURCE_FILES
    "source/dexrt_ray_3d.cpp"
    "source/PromweaverBoundary.cpp"
    "source/MiscSparse.cpp"
    "source/BlockMap.cpp"
)

add_executable(dexrt
    "${DEX_SOURCE_FILES}"
)
add_executable(dexrt_rad_eq
    "${DEX_RE_SOURCE_FILES}"
)
add_library(dexrt_sc_common OBJECT
    "${DEX_SC_COMMON_SOURCE_FILES}"
)
add_executable(dexrt_sc
    "source/ScMain.cpp"
    $<TARGET_OBJECTS:dexrt_sc_common>
)
add_executable(dexrt_sc_3d
    "${DEX_SC_3D_SOURCE_FILES}"
    $<TARGET_OBJECTS:dexrt_sc_common>
)
add_executable(dexrt_3d
    "${DEX_3D_SOURCE_FILES}"
)
add_executable(dexrt_ray
    "${DEX_RAY_SOURCE_FILES}"
)
add_executable(dexrt_ray_3d
    "${DEX_RAY_3D_SOURCE_FILES}"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if ("${DEX_ARCH}" STREQUAL "CUDA")
    set(CMAKE_CXX_STANDARD 20)
endif()

# NOTE(cmo): Override language type for cuda, may need to do this for HIP too.
set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_RE_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_SC_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_SC_3D_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_3D_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_RAY_3D_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
set_source_files_properties(${DEX_RE_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
set_source_files_properties(${DEX_SC_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
set_source_files_properties(${DEX_SC_3D_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
set_source_files_properties(${DEX_3D_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
set_source_files_properties(${DEX_RAY_3D_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
if ("${DEX_ARCH}" STREQUAL "CUDA")
    set(DEX_EXTRA_CUDA_STD_FLAG "")
    if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
        set(DEX_EXTRA_CUDA_STD_FLAG "-std=c++20")
    endif()
    set_target_properties(dexrt PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})
    set_target_properties(dexrt_rad_eq PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})
    foreach(target dexrt_sc_common dexrt_sc dexrt_sc_3d)
        set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})
    endforeach()
    set_target_properties(dexrt_3d PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})
    set_target_properties(dexrt_ray PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})
    set_target_properties(dexrt_ray_3d PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})

    set_source_files_properties(${DEX_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set_source_files_properties(${DEX_RE_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set_source_files_properties(${DEX_SC_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set_source_files_properties(${DEX_SC_3D_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set_source_files_properties(${DEX_3D_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set_source_files_properties(${DEX_RAY_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set_source_files_properties(${DEX_RAY_3D_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${DEX_EXTRA_CUDA_STD_FLAG}")
endif()

############## dexrt ##############
target_include_directories(dexrt SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
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
    target_include_directories(dexrt SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(dexrt MPI::MPI_CXX)
    target_compile_definitions(dexrt PUBLIC HAVE_MPI)
endif()

set_target_properties(dexrt PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)

message("Compiling with: ${CMAKE_CXX_FLAGS} ${YAKL_DEFS} ${DEX_CXX_FLAGS}")
############## dexrt_rad_eq ##############
target_include_directories(dexrt_rad_eq SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_link_libraries(dexrt_rad_eq "${LDLIBS}")
target_link_libraries(dexrt_rad_eq fmt::fmt)
target_link_libraries(dexrt_rad_eq yaml-cpp::yaml-cpp)
target_link_libraries(dexrt_rad_eq argparse)
target_link_libraries(dexrt_rad_eq git_version)
target_link_libraries(dexrt_rad_eq Kokkos::kokkos)
target_link_libraries(dexrt_rad_eq yakl)
if (KokkosKernels_FOUND)
    target_link_libraries(dexrt_rad_eq Kokkos::kokkoskernels)
endif()
if (${DEXRT_USE_MPI})
    target_include_directories(dexrt_rad_eq SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(dexrt_rad_eq MPI::MPI_CXX)
    target_compile_definitions(dexrt_rad_eq PUBLIC HAVE_MPI)
endif()

set_target_properties(dexrt_rad_eq PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)
############## dexrt_sc, dexrt_sc_3d ##############
foreach(target dexrt_sc_common dexrt_sc dexrt_sc_3d)
    target_include_directories(${target} SYSTEM PUBLIC
        "${GCC_INCLUDE_PATH}"
        "${MPI_INCLUDE_PATH}"
        "${NETCDF_INCLUDE_PATH}"
        "${MAGMA_INCLUDE_PATH}"
        "${CUDA_MATH_INCLUDE_PATH}"
    )
    target_link_libraries(${target} "${LDLIBS}")
    target_link_libraries(${target} fmt::fmt)
    target_link_libraries(${target} yaml-cpp::yaml-cpp)
    target_link_libraries(${target} argparse)
    target_link_libraries(${target} git_version)
    target_link_libraries(${target} Kokkos::kokkos)
    target_link_libraries(${target} yakl)
    if (KokkosKernels_FOUND)
        target_link_libraries(${target} Kokkos::kokkoskernels)
    endif()
    if (${DEXRT_USE_MPI})
        target_include_directories(${target} SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})
        target_link_libraries(${target} MPI::MPI_CXX)
        target_compile_definitions(${target} PUBLIC HAVE_MPI)
    endif()

    set_target_properties(${target} PROPERTIES
        COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}"
        LINK_FLAGS "${LDFLAGS}"
    )
    if (NOT "${DEX_ARCH}" STREQUAL "CUDA" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # NOTE(cmo): Lets the per-cell group loops vectorise (FP exceptions and
        # errno are never inspected).
        target_compile_options(${target} PRIVATE -fno-trapping-math -fno-math-errno)
    endif()
endforeach()
############## dexrt_sc_io_test (MPI builds) ##############
if (${DEXRT_USE_MPI})
    set_source_files_properties("tests/sc/sc_io_test.cpp" PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
    # Build this standalone regression explicitly, not as part of the solver binaries.
    add_executable(dexrt_sc_io_test EXCLUDE_FROM_ALL
        "tests/sc/sc_io_test.cpp"
        "source/ScIo.cpp"
    )
    target_include_directories(dexrt_sc_io_test PUBLIC "source")
    # FindMPI can leave include dirs empty with an MPI C++ wrapper; CUDA still
    # needs the explicit MPI path, just as the production targets do.
    target_include_directories(dexrt_sc_io_test SYSTEM PUBLIC
        "${MPI_INCLUDE_PATH}"
        "${NETCDF_INCLUDE_PATH}"
        ${MPI_CXX_INCLUDE_DIRS}
    )
    target_link_libraries(dexrt_sc_io_test "${LDLIBS}" fmt::fmt yaml-cpp::yaml-cpp Kokkos::kokkos yakl MPI::MPI_CXX)
    target_compile_definitions(dexrt_sc_io_test PUBLIC HAVE_MPI)
    set_target_properties(dexrt_sc_io_test PROPERTIES COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}" LINK_FLAGS "${LDFLAGS}")
    if ("${DEX_ARCH}" STREQUAL "CUDA")
        set_target_properties(dexrt_sc_io_test PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})
    endif()
endif()
############## dexrt_3d ##############
target_include_directories(dexrt_3d SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_include_directories(dexrt_3d PUBLIC
    "thirdparty"
)
target_link_libraries(dexrt_3d "${LDLIBS}")
target_link_libraries(dexrt_3d fmt::fmt)
target_link_libraries(dexrt_3d yaml-cpp::yaml-cpp)
target_link_libraries(dexrt_3d argparse)
target_link_libraries(dexrt_3d git_version)
target_link_libraries(dexrt_3d Kokkos::kokkos)
target_link_libraries(dexrt_3d yakl)
if (KokkosKernels_FOUND)
    target_link_libraries(dexrt_3d Kokkos::kokkoskernels)
endif()
if (${DEXRT_USE_MPI})
    target_include_directories(dexrt_3d SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(dexrt_3d MPI::MPI_CXX)
    target_compile_definitions(dexrt_3d PUBLIC HAVE_MPI)
endif()

set_target_properties(dexrt_3d PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)
############# dexrt_ray ##############
target_include_directories(dexrt_ray SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_include_directories(dexrt_ray PUBLIC
    "thirdparty"
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
if (${DEXRT_USE_MPI})
    target_include_directories(dexrt_ray SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(dexrt_ray MPI::MPI_CXX)
    target_compile_definitions(dexrt_ray PUBLIC HAVE_MPI)
endif()

set_target_properties(dexrt_ray PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)
############# dexrt_ray_3d ##############
target_include_directories(dexrt_ray_3d SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_include_directories(dexrt_ray PUBLIC
    "thirdparty"
)
target_link_libraries(dexrt_ray_3d "${LDLIBS}")
target_link_libraries(dexrt_ray_3d fmt::fmt)
target_link_libraries(dexrt_ray_3d yaml-cpp::yaml-cpp)
target_link_libraries(dexrt_ray_3d argparse)
target_link_libraries(dexrt_ray_3d git_version)
target_link_libraries(dexrt_ray_3d Kokkos::kokkos)
target_link_libraries(dexrt_ray_3d yakl)
if (KokkosKernels_FOUND)
    target_link_libraries(dexrt_ray_3d Kokkos::kokkoskernels)
endif()
if (${DEXRT_USE_MPI})
    target_include_directories(dexrt_ray_3d SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(dexrt_ray_3d MPI::MPI_CXX)
    target_compile_definitions(dexrt_ray_3d PUBLIC HAVE_MPI)
endif()

set_target_properties(dexrt_ray_3d PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${DEX_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)
