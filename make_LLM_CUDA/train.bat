@echo off
REM ===================================================================
REM TinyLLM CUDA 版 - 訓練実行スクリプト
REM ===================================================================

echo.
echo ======================================================
echo     TinyLLM CUDA 版 - 訓練実行
echo        学習用自作LLM GPU最適化版
echo ======================================================
echo.

if not exist "build\bin\Release\TinyLLM_CUDA.exe" (
    echo エラー: 実行ファイルが見つかりません
    echo 先に build_cuda.bat を実行してビルドしてください
    pause
    exit /b 1
)

echo [実行] 訓練モード開始...
echo.

.\build\bin\Release\TinyLLM_CUDA.exe train

echo.
pause
