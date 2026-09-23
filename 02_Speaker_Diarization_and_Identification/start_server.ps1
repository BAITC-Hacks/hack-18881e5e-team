param([int]$Port = 8082, [int]$Threads = 4, [switch]$Background)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$exe = Join-Path $taskRoot 'runtime\llama-server.exe'
$model = Join-Path $taskRoot 'models\Qwen3-8B-Q4_K_M.gguf'
if (!(Test-Path -LiteralPath $exe) -or !(Test-Path -LiteralPath $model)) {
    throw 'Model/runtime missing. Run setup_model.py first.'
}
$serverArgs = @('-m', ('"' + $model + '"'), '--alias', 'qwen3-8b-text', '--host', '127.0.0.1', '--port', $Port,
    '-c', 4096, '-np', 1, '-t', $Threads, '-tb', $Threads, '-b', 256, '-ub', 128,
    '-ngl', 0, '--jinja', '--reasoning', 'off')
if ($Background) {
    New-Item -ItemType Directory -Force -Path (Join-Path $taskRoot 'logs') | Out-Null
    $proc = Start-Process -FilePath $exe -ArgumentList $serverArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $taskRoot 'logs\server.stdout.log') `
        -RedirectStandardError (Join-Path $taskRoot 'logs\server.stderr.log')
    $proc.Id | Set-Content -LiteralPath (Join-Path $taskRoot 'logs\server.pid')
    Write-Host "Server starting: PID $($proc.Id), http://127.0.0.1:$Port"
} else {
    $serverArgs[1] = $model
    & $exe @serverArgs
}
