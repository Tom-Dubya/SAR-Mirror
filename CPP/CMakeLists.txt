set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fopenmp -march=native -ffast-math")

set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/Library/lib/pkgconfig")

cmake_minimum_required(VERSION 3.21)
project(CPP)

add_executable(CPP main.cpp
        src/utils/string_utils.h
        src/utils/matrix_math.h
        src/constants.h
        src/utils/stopwatch.h
        src/utils/stopwatch.cpp
        src/polarization_types.h
        src/algs/target_cp_corr_bp.cpp
        src/algs/target_cp_corr_bp.h
        src/algs/base_correlated_back_projection.h
        src/algs/af_dome_corr_bp.cpp
        src/algs/af_dome_corr_bp.h
        src/extern/mstar2raw.c
        src/extern/mstar2raw.h
        src/utils/file_utils.h
        src/utils/misc_utils.h
        src/algs/mstar_aggregator.cpp
        src/algs/mstar_aggregator.h
        src/algs/ph_mstar_corr_bp.cpp
        src/algs/ph_mstar_corr_bp.h
        src/algs/sample_corr_bp.cpp
        src/algs/sample_corr_bp.h
        src/utils/io_utils.h
)

target_compile_definitions(CPP PRIVATE
        ARMA_DONT_USE_WRAPPER
        ARMA_USE_HDF5
        ARMA_USE_OPENMP
        ARMA_DONT_PRINT_FAST_MATH_WARNING)

# Ideally, use Conda for dependency management.
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    target_compile_definitions(CPP PRIVATE
            ARMA_OPENMP_THREADS=42
            OPENBLAS_NUM_THREADS=42)

    pkg_check_modules(PKG_Open_BLAS REQUIRED IMPORTED_TARGET openblas)
    pkg_check_modules(PKG_Armadillo REQUIRED IMPORTED_TARGET armadillo)
    pkg_check_modules(PKG_ZLib REQUIRED IMPORTED_TARGET zlib)
    pkg_check_modules(PKG_FFTW REQUIRED IMPORTED_TARGET fftw3)

    include_directories($ENV{CONDA_PREFIX}/include)
    find_library(HDF5_LIBRARIES NAMES libhdf5 hdf5 HINTS $ENV{CONDA_PREFIX}/lib)
    find_library(HDF5_CPP_LIBRARIES NAMES libhdf5_cpp hdf5_cpp HINTS $ENV{CONDA_PREFIX}/lib)

    find_package(OpenMP COMPONENTS CXX REQUIRED)

    target_link_libraries(CPP
            PkgConfig::PKG_Open_BLAS
            PkgConfig::PKG_ZLib
            ${HDF5_LIBRARIES}
            ${HDF5_CPP_LIBRARIES}
            PkgConfig::PKG_Armadillo
            PkgConfig::PKG_FFTW
            ${OpenMP_CXX_LIBRARIES})
else()
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)

    set(ARMADILLO_INCLUDE_DIR ${LIB_DIR}/armadillo-14.0.3/include)
    include_directories(${LIB_DIR}/armadillo-14.0.3/include/)

    find_library(BLAS_LIB NAMES libopenblas PATHS LIB_DIR)

    find_package(Armadillo REQUIRED)
    find_package(HDF5 COMPONENTS CXX REQUIRED)
    find_package(OpenMP COMPONENTS CXX REQUIRED)

    include_directories(OpenMP_CXX_INCLUDE_DIRS)
    target_link_libraries(CPP ${BLAS_LIB} ${HDF5_LIBRARIES} ${ARMADILLO_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
endif()