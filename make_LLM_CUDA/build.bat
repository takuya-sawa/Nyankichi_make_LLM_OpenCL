@echo off
REM TinyLLM CUDA - Direct NVCC Compile (for x64 Native Tools Command Prompt)
REM x64 Native Tools Command Prompt で実行することを想定

setlocal enabledelayedexpansion

echo.
echo ======================================================
echo     TinyLLM CUDA Version - NVCC Compile
echo ======================================================
echo.

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
    echo Please set CUDA_PATH environment variable
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
echo [Build] Creating output directory...
if not exist "bin" mkdir bin

echo.
echo [Build] Compiling with NVCC...
echo Command:
echo "!CUDA_PATH!\bin\nvcc.exe" -std=c++17 -O3 -I"!CUDA_PATH!\include" -Iinclude -L"!CUDA_PATH!\lib\x64" -lcublas -lcurand -o bin\TinyLLM_CUDA.exe src\main.cu src\tensor_cuda.cu src\math_cuda.cu src\transformer_cuda.cu
echo.

REM コンパイル
"!CUDA_PATH!\bin\nvcc.exe" ^
    -std=c++17 ^
    -O3 ^
    -allow-unsupported-compiler ^
    -Xcompiler "/utf-8" ^
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
    echo ERROR: Compilation failed with error code %ERRORLEVEL%
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
if exist "bin\TinyLLM_CUDA.exe" (
    for %%F in (bin\TinyLLM_CUDA.exe) do (
        echo Size: %%~zF bytes
    )
)
echo.

echo [How to Run]
echo Train + Inference:  .\bin\TinyLLM_CUDA.exe
echo Train only:         .\bin\TinyLLM_CUDA.exe train
echo Inference only:     .\bin\TinyLLM_CUDA.exe infer
echo.

pause

endlocal
