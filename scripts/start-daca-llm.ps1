$ErrorActionPreference = "Stop"

$config = Join-Path $PSScriptRoot "..\configs\attacks\daca.yaml"
& (Join-Path $PSScriptRoot "start-local-llm.ps1") `
    --config $config `
    --section daca_llm `
    @args
exit $LASTEXITCODE
