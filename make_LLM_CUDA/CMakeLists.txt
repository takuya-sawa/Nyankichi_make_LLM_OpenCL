cmake_minimum_required(VERSION 3.18)
project(TinyLLM LANGUAGES CXX)

option(USE_OPENCL "Enable OpenCL backend instead of CUDA" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use UTF-8 source encoding with MSVC to avoid C4819 and console/locale issues
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 共通インクルードディレクトリ
include_directories(${CMAKE_SOURCE_DIR}/include)

if(USE_OPENCL)
    # OpenCL バックエンド
    # まず vcpkg が提供する CONFIG ベースのビンディングを試し、見つからなければ従来の FindOpenCL CMake モジュールを使う
    find_package(OpenCL CONFIG QUIET)
    if(TARGET OpenCL::OpenCL)
        message(STATUS "Found OpenCL (CONFIG) via vcpkg or package config")
        set(OPENCL_TARGET OpenCL::OpenCL)
        set(OPENCL_HEADERS OpenCL::Headers)
    else()
        find_package(OpenCL REQUIRED)
    endif()

    message(STATUS "Building with OpenCL backend")

    set(SOURCES
        src/main_opencl.cpp
        src/tensor_opencl.cpp
        src/math_opencl.cpp
        src/transformer_opencl.cpp
    )

    add_executable(TinyLLM_OPENCL ${SOURCES})
    # test executable for gradient checks
    add_executable(TinyLLM_GRADTEST src/test_gradients.cpp src/transformer_opencl.cpp src/math_opencl.cpp src/tensor_opencl.cpp)
    target_compile_definitions(TinyLLM_GRADTEST PUBLIC USE_OPENCL)

    # Link OpenCL targets/includes for the test executable as well
    if(TARGET OpenCL::OpenCL)
        target_link_libraries(TinyLLM_GRADTEST PUBLIC OpenCL::OpenCL)
    endif()
    if(TARGET OpenCL::Headers)
        target_link_libraries(TinyLLM_GRADTEST PUBLIC OpenCL::Headers OpenCL::HeadersCpp)
    elseif(DEFINED OpenCL_INCLUDE_DIRS)
        target_include_directories(TinyLLM_GRADTEST PUBLIC ${OpenCL_INCLUDE_DIRS})
    endif()
    target_compile_definitions(TinyLLM_OPENCL PUBLIC USE_OPENCL)

    # CONFIG ベースのターゲットが存在すればそれらをリンク（ヘッダ/ライブラリを伝播）
    if(TARGET OpenCL::OpenCL)
        target_link_libraries(TinyLLM_OPENCL PUBLIC OpenCL::OpenCL)
    endif()
    if(TARGET OpenCL::Headers)
        target_link_libraries(TinyLLM_OPENCL PUBLIC OpenCL::Headers OpenCL::HeadersCpp)
    elseif(DEFINED OpenCL_INCLUDE_DIRS)
        include_directories(${OpenCL_INCLUDE_DIRS})
    endif()
    if(TARGET OpenCL::Utils)
        target_link_libraries(TinyLLM_OPENCL PUBLIC OpenCL::Utils OpenCL::UtilsCpp)
    endif()

    set_target_properties(TinyLLM_OPENCL PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")

    # PoC test: transformer layer forward using CPU AVX2 path
    add_executable(transformer_poc_test src/test_transformer_poc.cpp src/transformer_opencl.cpp src/math_opencl.cpp)
    target_include_directories(transformer_poc_test PRIVATE include)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(transformer_poc_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MSVC)
      target_compile_options(transformer_poc_test PRIVATE /openmp:experimental /utf-8 /arch:AVX2)
    endif()
    # Link or add OpenCL headers if available
    if(TARGET OpenCL::Headers)
      target_link_libraries(transformer_poc_test PUBLIC OpenCL::Headers OpenCL::HeadersCpp)
    elseif(DEFINED OpenCL_INCLUDE_DIRS)
      target_include_directories(transformer_poc_test PUBLIC ${OpenCL_INCLUDE_DIRS})
    endif()
    # Include paths for make_llm_High (for PoC rec_gemm include)
    target_include_directories(transformer_poc_test PRIVATE ${CMAKE_SOURCE_DIR}/../make_llm_High/include)
    include(CTest)
    add_test(NAME transformer_poc_test COMMAND transformer_poc_test)
    set_tests_properties(transformer_poc_test PROPERTIES TIMEOUT 240)
else()
    # CUDA バックエンド（既存）
    find_package(CUDA REQUIRED)

    include_directories(${CUDA_INCLUDE_DIRS})

    set(SOURCES
        src/main.cu
        src/tensor_cuda.cu
        src/math_cuda.cu
        src/transformer_cuda.cu
    )

    add_executable(TinyLLM_CUDA ${SOURCES})

    # CUDA オプション設定
    set_target_properties(TinyLLM_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86;89;90"
    )

    # cuBLAS ライブラリをリンク
    target_link_libraries(TinyLLM_CUDA
        PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CURAND_LIBRARY}
    )

    # ビルドディレクトリ
    set_target_properties(TinyLLM_CUDA PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # メッセージ表示
    message(STATUS "CUDA Toolkit Version: ${CUDA_VERSION}")
    message(STATUS "CUDA Libraries: ${CUDA_CUBLAS_LIBRARIES}")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
