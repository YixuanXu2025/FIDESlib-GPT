cmake_minimum_required(VERSION 3.25)
project(FIDESlibGPT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_NATIVE_SMOKE "Build native smoke test executable" ON)

set(FIDESLIB_ROOT "/root/fideslib" CACHE PATH "Path to FIDESlib source/build tree")
set(OPENFHE_ROOT "/root/fideslib/deps/openfhe-install" CACHE PATH "Path to OpenFHE install")

# Locate pybind11 from the current Python environment
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYBIND11_CMAKE_DIR)
    message(FATAL_ERROR "Failed to locate pybind11 CMake directory. Did you install pybind11?")
endif()

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenFHE 1.4.2 CONFIG REQUIRED
    PATHS "${OPENFHE_ROOT}/lib/OpenFHE"
    NO_DEFAULT_PATH
)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(NCCL_LIB nccl REQUIRED)

set(COMMON_INCLUDE_DIRS
    "${FIDESLIB_ROOT}/api"
)

set(COMMON_LINK_LIBS
    "${FIDESLIB_ROOT}/build/fideslib.a"
    OPENFHEpke
    OPENFHEbinfhe
    OPENFHEcore
    TBB::tbb
    OpenMP::OpenMP_CXX
    Threads::Threads
    CUDA::cudart
    CUDA::cuda_driver
    ${NCCL_LIB}
)

# Python extension module
pybind11_add_module(_fideslib MODULE
    bindings/bindings.cpp
)

target_include_directories(_fideslib PRIVATE
    ${COMMON_INCLUDE_DIRS}
)

target_link_libraries(_fideslib PRIVATE
    ${COMMON_LINK_LIBS}
)

set_target_properties(_fideslib PROPERTIES
    BUILD_RPATH "${OPENFHE_ROOT}/lib;/usr/local/cuda/lib64"
    INSTALL_RPATH "${OPENFHE_ROOT}/lib;/usr/local/cuda/lib64"
)

# Native smoke test executable
if(BUILD_NATIVE_SMOKE AND EXISTS "${CMAKE_SOURCE_DIR}/cpp/native_smoke_test.cpp")
    add_executable(fideslib_gpt_minimal
        cpp/native_smoke_test.cpp
    )

    target_include_directories(fideslib_gpt_minimal PRIVATE
        ${COMMON_INCLUDE_DIRS}
    )

    target_link_libraries(fideslib_gpt_minimal PRIVATE
        ${COMMON_LINK_LIBS}
    )

    set_target_properties(fideslib_gpt_minimal PROPERTIES
        BUILD_RPATH "${OPENFHE_ROOT}/lib;/usr/local/cuda/lib64"
        INSTALL_RPATH "${OPENFHE_ROOT}/lib;/usr/local/cuda/lib64"
    )
endif()

message(STATUS "FIDESLIB_ROOT      = ${FIDESLIB_ROOT}")
message(STATUS "OPENFHE_ROOT       = ${OPENFHE_ROOT}")
message(STATUS "PYBIND11_CMAKE_DIR = ${PYBIND11_CMAKE_DIR}")
message(STATUS "BUILD_NATIVE_SMOKE = ${BUILD_NATIVE_SMOKE}")
