param([string]$InputFile)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '..\01_Speech_Transcription_Module\.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { $python = 'python' }
$ready = $false
try { $ready = (Invoke-RestMethod 'http://127.0.0.1:8082/health' -TimeoutSec 3).status -eq 'ok' } catch {}
if (!$ready) {
    & (Join-Path $PSScriptRoot 'start_server.ps1') -Background
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        Start-Sleep -Seconds 2
        try { $ready = (Invoke-RestMethod 'http://127.0.0.1:8082/health' -TimeoutSec 2).status -eq 'ok' } catch {}
        if ($ready) { break }
    }
}
if (!$ready) { throw 'Server did not start; inspect logs/server.stderr.log' }
$models = Invoke-RestMethod 'http://127.0.0.1:8082/v1/models' -TimeoutSec 5
if ('qwen3-8b-text' -notin $models.data.id) { throw 'Port 8082 is occupied by a different model.' }
$scriptPath = Join-Path $PSScriptRoot 'analyze_text.py'
if ($InputFile) { & $python $scriptPath $InputFile } else { & $python $scriptPath }
exit $LASTEXITCODE
