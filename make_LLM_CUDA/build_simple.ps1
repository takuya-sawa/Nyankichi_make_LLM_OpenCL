#!/usr/bin/env powershell
# TinyLLM CUDA Version - Simple Build Script using NVCC directly

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "     TinyLLM CUDA Version - Direct NVCC Build" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Set CUDA_PATH
if (Test-Path env:CUDA_PATH) {
    Write-Host "CUDA_PATH found: $env:CUDA_PATH" -ForegroundColor Green
} else {
    $cuda_search_paths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    )
    
    $cuda_found = $false
    foreach ($path in $cuda_search_paths) {
        if (Test-Path $path) {
            $env:CUDA_PATH = $path
            Write-Host "Found CUDA at: $path" -ForegroundColor Green
            $cuda_found = $true
            break
        }
    }
    
    if (-not $cuda_found) {
        Write-Host "ERROR: CUDA Toolkit not found" -ForegroundColor Red
        exit 1
    }
}

# Verify NVCC
$nvcc_path = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
if (-not (Test-Path $nvcc_path)) {
    Write-Host "ERROR: NVCC compiler not found at: $nvcc_path" -ForegroundColor Red
    exit 1
}
Write-Host "NVCC compiler: OK" -ForegroundColor Green

# Verify cuBLAS
$cublas_path = Join-Path $env:CUDA_PATH "lib\x64\cublas.lib"
if (-not (Test-Path $cublas_path)) {
    Write-Host "ERROR: cuBLAS library not found at: $cublas_path" -ForegroundColor Red
    exit 1
}
Write-Host "cuBLAS library: OK" -ForegroundColor Green

Write-Host ""
Write-Host "[Build] Compiling with NVCC..." -ForegroundColor Yellow
Write-Host ""

# Set up environment
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

# Create output directory
if (-not (Test-Path "bin")) {
    New-Item -ItemType Directory -Path "bin" | Out-Null
}

# CUDA compile flags
$CUDA_INCLUDE = "$env:CUDA_PATH\include"
$CUDA_LIB = "$env:CUDA_PATH\lib\x64"
$INCLUDE_DIR = "include"
$OUTPUT = "bin\TinyLLM_CUDA.exe"

# Compile source files
$source_files = @(
    "src\main.cu",
    "src\tensor_cuda.cu",
    "src\math_cuda.cu",
    "src\transformer_cuda.cu"
)

# Build argument array properly
$compile_args = @(
    "-std=c++17",
    "-O3",
    "-I`"$CUDA_INCLUDE`"",
    "-I$INCLUDE_DIR",
    "-L`"$CUDA_LIB`"",
    "-lcublas",
    "-lcurand",
    "-o",
    $OUTPUT
) + $source_files

Write-Host "Compiling: $nvcc_path" -ForegroundColor Gray
Write-Host "Arguments: $compile_args" -ForegroundColor Gray
Write-Host ""

& $nvcc_path $compile_args

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Compilation failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "         Build Completed Successfully!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

Write-Host "[Output]" -ForegroundColor Cyan
Write-Host "Executable: .\bin\TinyLLM_CUDA.exe"
Write-Host ""

Write-Host "[How to Run]" -ForegroundColor Cyan
Write-Host "Train + Inference:  .\bin\TinyLLM_CUDA.exe"
Write-Host "Train only:         .\bin\TinyLLM_CUDA.exe train"
Write-Host "Inference only:     .\bin\TinyLLM_CUDA.exe infer"
Write-Host ""
