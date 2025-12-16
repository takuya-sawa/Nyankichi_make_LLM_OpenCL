@echo off
REM TinyLLM CUDA - MSVC + NVCC Build Script
REM x64 Native Tools Command Prompt for VS 環境をセットアップしてコンパイル

setlocal enabledelayedexpansion

echo.
echo ======================================================
echo     TinyLLM CUDA Version - MSVC + NVCC Build
echo ======================================================
echo.

REM Visual Studio のパスを検索
set "VS_PATH="
set "VSVARS="

for %%v in (Community Professional Enterprise) do (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\%%v\VC\Auxiliary\Build\vcvars64.bat" (
        set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\%%v"
        set "VSVARS=C:\Program Files\Microsoft Visual Studio\2022\%%v\VC\Auxiliary\Build\vcvars64.bat"
        echo Found Visual Studio 2022 %%v at: !VS_PATH!
        goto :found_vs
    )
)

if not defined VSVARS (
    echo ERROR: Visual Studio 2022 not found
    echo Please install Visual Studio 2022 with C++ development tools
    pause
    exit /b 1
)

:found_vs

REM CUDA パスを検索
if defined CUDA_PATH (
    echo CUDA_PATH found: !CUDA_PATH!
) else (
    echo Searching for CUDA Toolkit...
    
    for %%v in (v13.1 v13.0 v12.4 v12.3 v12.2 v12.1 v12.0 v11.8) do (
        if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%v" (
            set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%v"
            echo Found CUDA at: !CUDA_PATH!
            goto :found_cuda
        )
    )
    
    echo ERROR: CUDA Toolkit not found
    pause
    exit /b 1
)

:found_cuda

REM 必須ファイルの確認
if not exist "!CUDA_PATH!\bin\nvcc.exe" (
    echo ERROR: NVCC not found at !CUDA_PATH!\bin\nvcc.exe
    pause
    exit /b 1
)
echo NVCC compiler: OK

if not exist "!CUDA_PATH!\lib\x64\cublas.lib" (
    echo ERROR: cuBLAS not found at !CUDA_PATH!\lib\x64\cublas.lib
    pause
    exit /b 1
)
echo cuBLAS library: OK

echo.
echo [Build] Setting up Visual Studio build environment...
echo Executing: "!VSVARS!"
call "!VSVARS!" x64 >nul 2>&1

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to setup Visual Studio environment
    pause
    exit /b 1
)

echo Visual Studio environment: OK

echo.
echo [Build] Creating output directory...
if not exist "bin" mkdir bin

echo [Build] Compiling with NVCC...
echo.

REM コンパイル（デバッグ情報付き）
"!CUDA_PATH!\bin\nvcc.exe" ^
    -std=c++17 ^
    -O3 ^
    -G ^
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

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Compilation failed
    pause
    exit /b 1
)

echo.
echo ======================================================
echo         Build Completed Successfully!
echo ======================================================
echo.

echo [Output]
echo Executable: .\bin\TinyLLM_CUDA.exe
echo Size: 
for %%F in (bin\TinyLLM_CUDA.exe) do echo %%~zF bytes
echo.

echo [How to Run]
echo Train + Inference:  .\bin\TinyLLM_CUDA.exe
echo Train only:         .\bin\TinyLLM_CUDA.exe train
echo Inference only:     .\bin\TinyLLM_CUDA.exe infer
echo.

pause

endlocal
