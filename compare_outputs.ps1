$cpu = Get-Content 'C:\div\gpu_test_cpu.txt' | Select-String 'ステップ|入力:|予測:' | ForEach-Object { $_.Line }
$gpu = Get-Content 'C:\div\gpu_test_gpu.txt' | Select-String 'ステップ|入力:|予測:' | ForEach-Object { $_.Line }
Write-Output '--- CPU ---'
$cpu
Write-Output '--- GPU ---'
$gpu
if ($cpu -eq $gpu) {
    Write-Output 'Exact match'
} else {
    Write-Output 'Differences detected'
    Write-Output '--- Diff ---'
    Compare-Object -ReferenceObject $cpu -DifferenceObject $gpu | ForEach-Object { $_ }
}
