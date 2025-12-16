@echo off
cd /d "%~dp0"
echo.
echo DEBUG: Current directory is %CD%
echo.
echo ======================================================
echo     TinyLLM OpenCL - CMake Build
echo ======================================================
echo.

if not exist "build" mkdir build

REM --- vcpkg detection and OpenCL install (if vcpkg available) ---
set "CMAKE_TOOLCHAIN_OPTION="

REM Simple check for vcpkg in C:\vcpkg or repo-local vcpkg
if exist "C:\vcpkg\vcpkg.exe" (
    echo DEBUG: Using vcpkg at C:\vcpkg
    set "CMAKE_TOOLCHAIN_OPTION=-DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake"
) else if exist "%~dp0vcpkg\vcpkg.exe" (
    echo DEBUG: Using repo-local vcpkg at %~dp0vcpkg
    set "CMAKE_TOOLCHAIN_OPTION=-DCMAKE_TOOLCHAIN_FILE=%~dp0vcpkg\scripts\buildsystems\vcpkg.cmake"
) else (
    echo vcpkg not found; proceeding without vcpkg. If CMake can't find OpenCL, install an OpenCL SDK or set OpenCL paths manually.
)

echo Running CMake configuration...
echo DEBUG: VCPKG_ROOT=%VCPKG_ROOT%
echo DEBUG: CMAKE_TOOLCHAIN_OPTION=%CMAKE_TOOLCHAIN_OPTION%
cmake -S . -B build -DUSE_OPENCL=ON -DCMAKE_BUILD_TYPE=Release %CMAKE_TOOLCHAIN_OPTION%
if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed
    pause
    exit /b 1
)

cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
if exist "build\bin\Release\TinyLLM_OPENCL.exe" (
    echo Build completed. Executable located in build\bin\Release\
    echo.
    echo [How to Run]
    echo   PowerShell: .\build\bin\Release\TinyLLM_OPENCL.exe
    echo   cmd:       build\bin\Release\TinyLLM_OPENCL.exe
) else if exist "build\bin\TinyLLM_OPENCL.exe" (
    echo Build completed. Executable located in build\bin\
    echo.
    echo [How to Run]
    echo   PowerShell/cmd: .\build\bin\TinyLLM_OPENCL.exe
) else (
    echo Build completed, but the executable was not found in expected locations.
    echo Check build\bin\Release\ or build\bin\ for the executable.
)

echo.
pause