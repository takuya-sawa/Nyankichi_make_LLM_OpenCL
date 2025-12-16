#!/usr/bin/env powershell
# TinyLLM CUDA Version - Setup & Build Script

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "     TinyLLM CUDA Version - Setup & Build" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if CUDA_PATH is already set
if (Test-Path env:CUDA_PATH) {
    Write-Host "CUDA_PATH found: $env:CUDA_PATH" -ForegroundColor Green
} else {
    Write-Host "CUDA_PATH not found. Searching for CUDA installation..." -ForegroundColor Yellow
    
    # Common CUDA installation paths
    $cuda_search_paths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0"
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
        Write-Host "Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
        exit 1
    }
}

# Verify cuBLAS
$cublas_path = Join-Path $env:CUDA_PATH "lib\x64\cublas.lib"
if (-not (Test-Path $cublas_path)) {
    Write-Host "ERROR: cuBLAS library not found at: $cublas_path" -ForegroundColor Red
    exit 1
}
Write-Host "cuBLAS library: OK" -ForegroundColor Green

# Verify NVCC
$nvcc_path = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
if (-not (Test-Path $nvcc_path)) {
    Write-Host "ERROR: NVCC compiler not found at: $nvcc_path" -ForegroundColor Red
    exit 1
}
Write-Host "NVCC compiler: OK" -ForegroundColor Green

# Check CMake
try {
    $cmake_version = cmake --version 2>&1 | Select-Object -First 1
    Write-Host "CMake: $cmake_version" -ForegroundColor Green
} catch {
    Write-Host "ERROR: CMake not found or not in PATH" -ForegroundColor Red
    Write-Host "Please install CMake from: https://cmake.org/download/" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "All prerequisites verified!" -ForegroundColor Green
Write-Host ""

# Create build directory
if (-not (Test-Path "build")) {
    Write-Host "[Build] Creating build directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Set-Location build

# Check for Ninja or fall back to NMake
$ninja_available = $false
try {
    $ninja_version = ninja --version 2>&1
    $ninja_available = $true
    Write-Host "[CMake] Using Ninja generator..." -ForegroundColor Yellow
} catch {
    Write-Host "[CMake] Ninja not found, using NMake Makefiles..." -ForegroundColor Yellow
}

# Run CMake
if ($ninja_available) {
    Write-Host "[CMake] Generating Ninja project..." -ForegroundColor Yellow
    cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
} else {
    Write-Host "[CMake] Generating NMake Makefiles..." -ForegroundColor Yellow
    cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: CMake generation failed" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "[Build] Starting Release build..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow
Write-Host ""

# Build Release
if ($ninja_available) {
    ninja
} else {
    nmake
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Set-Location ..

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "         Build Completed Successfully!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

Write-Host "[Output]" -ForegroundColor Cyan
Write-Host "Executable: .\build\bin\Release\TinyLLM_CUDA.exe"
Write-Host ""

Write-Host "[How to Run]" -ForegroundColor Cyan
Write-Host "Train + Inference:  .\build\bin\Release\TinyLLM_CUDA.exe"
Write-Host "Train only:         .\build\bin\Release\TinyLLM_CUDA.exe train"
Write-Host "Inference only:     .\build\bin\Release\TinyLLM_CUDA.exe infer"
Write-Host ""
