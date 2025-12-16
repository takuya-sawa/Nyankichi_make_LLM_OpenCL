cmake_minimum_required(VERSION 3.15)
project(make_llm_high LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "Found OpenMP")
endif()

add_executable(make_llm_high_bench
  src/main_bench.cpp
  src/dense_tile.cpp
)

target_include_directories(make_llm_high_bench PRIVATE include)

if(MSVC)
  # MSVC: enable OpenMP experimental simd pragma and force UTF-8 source interpretation
  target_compile_options(make_llm_high_bench PRIVATE /openmp:experimental /utf-8 /arch:AVX2)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(make_llm_high_bench PRIVATE OpenMP::OpenMP_CXX)
endif()

# small helper to detect CPU features on Windows
add_executable(cpuid_check tools/cpuid_check.cpp)
if(MSVC)
  target_compile_options(cpuid_check PRIVATE /utf-8)
endif()

# micro-kernel unit test
add_executable(microkernel_test src/test_microkernel.cpp src/dense_tile.cpp)
target_include_directories(microkernel_test PRIVATE include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(microkernel_test PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
  target_compile_options(microkernel_test PRIVATE /openmp:experimental /utf-8 /arch:AVX2)
endif()

# PoC batched GEMM test
add_executable(batched_gemm_test src/test_batched_gemm.cpp src/dense_tile.cpp)
target_include_directories(batched_gemm_test PRIVATE include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(batched_gemm_test PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
  target_compile_options(batched_gemm_test PRIVATE /openmp:experimental /utf-8 /arch:AVX2)
endif()

# CTest integration
include(CTest)
enable_testing()
add_test(NAME microkernel_test COMMAND microkernel_test)
set_tests_properties(microkernel_test PROPERTIES TIMEOUT 120)
add_test(NAME batched_gemm_test COMMAND batched_gemm_test)
set_tests_properties(batched_gemm_test PROPERTIES TIMEOUT 120)