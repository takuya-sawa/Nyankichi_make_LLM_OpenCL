$token = Get-Content -Raw 'C:\div\github_token.txt'
$headers = @{ Authorization = "token $token"; 'User-Agent' = 'tiny' }
try {
    Invoke-RestMethod -Headers $headers -Uri 'https://api.github.com/repos/takuya-sawa/Nyankichi_make_LLM_OpenCL' -Method GET
    Write-Output 'RepoExists'
} catch {
    if ($_.Exception -and $_.Exception.Response -and $_.Exception.Response.StatusCode -eq 404) {
        Write-Output 'RepoMissing'
        $body = @{ name='Nyankichi_make_LLM_OpenCL'; private=$false; description='TinyLLM OpenCL' } | ConvertTo-Json
        Invoke-RestMethod -Headers $headers -Uri 'https://api.github.com/user/repos' -Method POST -Body $body -ContentType 'application/json'
        Write-Output 'RepoCreated'
    } else {
        Write-Error $_
        exit 1
    }
}