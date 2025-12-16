@echo off
set VCPKG_ROOT=C:\vcpkg
echo BEFORE
if not "%VCPKG_ROOT%"=="" (
    echo ENTERED
)
echo AFTER
pause

