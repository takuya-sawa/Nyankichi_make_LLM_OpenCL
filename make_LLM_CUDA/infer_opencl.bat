@echo off
REM ===================================================================
REM TinyLLM OpenCL 版 - 推論（GPU）実行スクリプト
REM ===================================================================

echo.
echo ======================================================
echo     TinyLLM OpenCL 版 - 推論（GPU）実行
echo ======================================================
echo.

if not exist "build\bin\Release\TinyLLM_OPENCL.exe" (
    echo エラー: 実行ファイルが見つかりません
    echo 先に build_opencl.bat を実行してビルドしてください
    pause
    exit /b 1
)

echo [実行] 推論モード開始 (OpenCL)...
echo.

.\build\bin\Release\TinyLLM_OPENCL.exe --opencl infer --verbosity 2

echo.
pause
