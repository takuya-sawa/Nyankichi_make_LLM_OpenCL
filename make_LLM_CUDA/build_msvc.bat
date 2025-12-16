@echo off
setlocal enabledelayedexpansion

echo.
echo ======================================================
echo      TinyLLM CUDA Version - MSVC Build
echo ======================================================
echo.

REM Find Visual Studio 2022
set "VS_PATH="
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
)

if not defined VS_PATH (
    echo ERROR: Visual Studio 2022 not found
    echo Please install Visual Studio 2022 with C++ development tools
    pause
    exit /b 1
)

echo [VS] Found: !VS_PATH!

REM Initialize MSVC environment
echo [Build] Setting up MSVC environment...
call "!VS_PATH!\VC\Auxiliary\Build\vcvarsall.bat" x64

if errorlevel 1 (
    echo ERROR: Failed to initialize MSVC environment
    pause
    exit /b 1
)

REM Check CUDA_PATH
if not defined CUDA_PATH (
    echo [CUDA] Searching for CUDA installation...
    
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
    ) else (
        echo ERROR: CUDA Toolkit not found
        pause
        exit /b 1
    )
)

echo [CUDA] CUDA_PATH: !CUDA_PATH!

REM Verify NVCC
if not exist "!CUDA_PATH!\bin\nvcc.exe" (
    echo ERROR: NVCC compiler not found at !CUDA_PATH!\bin\nvcc.exe
    pause
    exit /b 1
)
echo [CUDA] NVCC: OK

REM Verify cuBLAS
if not exist "!CUDA_PATH!\lib\x64\cublas.lib" (
    echo ERROR: cuBLAS library not found at !CUDA_PATH!\lib\x64\cublas.lib
    pause
    exit /b 1
)
echo [CUDA] cuBLAS: OK

REM Create output directory
if not exist "bin" mkdir bin

echo.
echo [Build] Compiling with NVCC and MSVC...
echo.

REM Compile with NVCC
"!CUDA_PATH!\bin\nvcc.exe" ^
    -std=c++17 ^
    -O3 ^
    -I"!CUDA_PATH!\include" ^
    -Iinclude ^
    -L"!CUDA_PATH!\lib\x64" ^
    -lcublas ^
    -lcurand ^
    -o bin\TinyLLM_CUDA.exe ^
    src\main.cu ^
    src\tensor_cuda.cu ^
    src\math_cuda.cu ^
    src\transformer_cuda.cu

if errorlevel 1 (
    echo.
    echo ERROR: Compilation failed
    pause
    exit /b 1
)

echo.
echo ======================================================
echo          Build Completed Successfully!
echo ======================================================
echo.

echo [Output]
echo Executable: .\bin\TinyLLM_CUDA.exe
echo.

echo [How to Run]
echo Train + Inference:  .\bin\TinyLLM_CUDA.exe
echo Train only:         .\bin\TinyLLM_CUDA.exe train
echo Inference only:     .\bin\TinyLLM_CUDA.exe infer
echo.

pause
