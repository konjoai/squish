<#
.SYNOPSIS
    Squash AI Model SBOM Attestation — Azure DevOps task runner.

.DESCRIPTION
    Invoked by the SquashAttest@1 pipeline task.  This script:
      1. Reads task inputs from ADO-injected INPUT_* environment variables.
      2. Installs squish[squash] via pip (unless already present).
      3. Builds and runs the squash attest CLI command.
      4. Parses the JSON result and sets pipeline output variables via
         ##vso[task.setvariable] logging commands.
      5. Optionally publishes attestation artifacts to the pipeline run.
      6. Fails the task (exit 1 + ##vso[task.complete result=Failed]) if
         failOnViolation is true and attestation did not pass.

    ADO supplies inputs as environment variables prefixed with INPUT_ and the
    input name uppercased.  Boolean inputs arrive as the string "true"/"false".

.NOTES
    Requires PowerShell Core (pwsh) 7+ for cross-platform support on Linux/macOS
    ADO agents.  Windows agents use Windows PowerShell 5.1+ or PSCore.
    Python 3.9+ with pip must be available on the agent PATH.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Input binding — ADO injects task inputs as INPUT_<UPPERCASED_INPUT_NAME>
# ---------------------------------------------------------------------------

function Get-TaskInput {
    param(
        [string]$Name,
        [string]$Default = '',
        [switch]$Required
    )
    $envKey = "INPUT_$($Name.ToUpperInvariant())"
    $value  = [System.Environment]::GetEnvironmentVariable($envKey)
    if ([string]::IsNullOrWhiteSpace($value)) {
        if ($Required) {
            Write-Host "##vso[task.logissue type=error]Required input '$Name' is missing."
            exit 1
        }
        return $Default
    }
    return $value.Trim()
}

$ModelPath        = Get-TaskInput -Name 'modelPath'        -Required
$Policies         = Get-TaskInput -Name 'policies'         -Default 'enterprise-strict'
$Sign             = (Get-TaskInput -Name 'sign'            -Default 'false') -eq 'true'
$FailOnViolation  = (Get-TaskInput -Name 'failOnViolation' -Default 'true')  -ne 'false'
$OutputDir        = Get-TaskInput -Name 'outputDir'        -Default ''
$SquishVersion    = Get-TaskInput -Name 'squishVersion'    -Default ''
$PythonExe        = Get-TaskInput -Name 'pythonExecutable' -Default 'python3'
$PublishArtifacts = (Get-TaskInput -Name 'publishArtifacts' -Default 'true') -eq 'true'

Write-Host "##[section]Squash — AI Model SBOM Attestation"
Write-Host "  model path : $ModelPath"
Write-Host "  policies   : $Policies"
Write-Host "  sign       : $Sign"
Write-Host "  fail gate  : $FailOnViolation"

# ---------------------------------------------------------------------------
# Install squish[squash]
# ---------------------------------------------------------------------------

$PkgSpec = if ($SquishVersion) { "squish[squash]==$SquishVersion" } else { 'squish[squash]' }
Write-Host ""
Write-Host "##[group]Installing $PkgSpec"
& $PythonExe -m pip install --quiet $PkgSpec
if ($LASTEXITCODE -ne 0) {
    Write-Host "##[endgroup]"
    Write-Host "##vso[task.logissue type=error]pip install failed for '$PkgSpec'. Ensure Python and pip are available on the agent."
    exit 1
}
Write-Host "##[endgroup]"

# ---------------------------------------------------------------------------
# Build squash CLI argument list
# ---------------------------------------------------------------------------

$ResultPath = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "squash-result-$([System.Guid]::NewGuid()).json")

$CliArgs = @('attest', '--model-path', $ModelPath, '--json-result', $ResultPath)

foreach ($Policy in ($Policies -split ',')) {
    $Trimmed = $Policy.Trim()
    if ($Trimmed) { $CliArgs += @('--policy', $Trimmed) }
}

if ($Sign)       { $CliArgs += '--sign' }
if ($OutputDir)  { $CliArgs += @('--output-dir', $OutputDir) }

# ---------------------------------------------------------------------------
# Run squash attest
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "##[group]squash $($CliArgs -join ' ')"
$RunExitCode = 0
try {
    squash @CliArgs
    $RunExitCode = $LASTEXITCODE
} catch {
    Write-Host "##vso[task.logissue type=error]Failed to invoke squash: $_"
    $RunExitCode = 1
} finally {
    Write-Host "##[endgroup]"
}

# ---------------------------------------------------------------------------
# Parse result JSON and set pipeline output variables
# ---------------------------------------------------------------------------

$Passed     = 'false'
$ScanStatus = 'skipped'

if (Test-Path $ResultPath) {
    try {
        $Result    = Get-Content $ResultPath -Raw | ConvertFrom-Json
        $Passed    = if ($Result.passed) { 'true' } else { 'false' }
        $ScanStatus = if ($null -ne $Result.scan_status) { [string]$Result.scan_status } else { 'skipped' }
        $Artifacts  = $Result.artifacts

        # Set pipeline output variables for downstream consumption
        Write-Host "##vso[task.setvariable variable=SQUASH_PASSED;isOutput=true]$Passed"
        Write-Host "##vso[task.setvariable variable=SQUASH_SCAN_STATUS;isOutput=true]$ScanStatus"

        if ($Artifacts) {
            if ($Artifacts.cyclonedx)     { Write-Host "##vso[task.setvariable variable=SQUASH_CYCLONEDX_PATH;isOutput=true]$($Artifacts.cyclonedx)" }
            if ($Artifacts.spdx_json)     { Write-Host "##vso[task.setvariable variable=SQUASH_SPDX_JSON_PATH;isOutput=true]$($Artifacts.spdx_json)" }
            if ($Artifacts.master_record) { Write-Host "##vso[task.setvariable variable=SQUASH_MASTER_RECORD_PATH;isOutput=true]$($Artifacts.master_record)" }
        }

        # Human-readable summary
        Write-Host ""
        Write-Host "Squash result: passed=$Passed  scan=$ScanStatus"
        if ($Result.policy_results) {
            $Result.policy_results.PSObject.Properties | ForEach-Object {
                $PolicyName   = $_.Name
                $PolicyResult = $_.Value
                $Status       = if ($PolicyResult.passed) { 'PASS' } else { 'FAIL' }
                $Errors       = if ($null -ne $PolicyResult.error_count)   { $PolicyResult.error_count }   else { 0 }
                $Warnings     = if ($null -ne $PolicyResult.warning_count) { $PolicyResult.warning_count } else { 0 }
                Write-Host "  [$Status] $PolicyName : $Errors error(s), $Warnings warning(s)"
            }
        }

        # Publish artifacts to pipeline run
        if ($PublishArtifacts -and $Artifacts) {
            foreach ($ArtifactPath in @($Artifacts.cyclonedx, $Artifacts.spdx_json, $Artifacts.master_record)) {
                if ($ArtifactPath -and (Test-Path $ArtifactPath)) {
                    Write-Host "##vso[artifact.upload artifactname=squash-attestation;]$ArtifactPath"
                }
            }
        }

    } catch {
        Write-Host "##vso[task.logissue type=warning]Could not parse Squash result JSON at '$ResultPath': $_"
    }
} else {
    Write-Host "##vso[task.logissue type=warning]squash did not produce a result file at '$ResultPath'."
    $Passed = 'false'
}

# ---------------------------------------------------------------------------
# Final task status
# ---------------------------------------------------------------------------

if ($FailOnViolation -and $Passed -ne 'true') {
    Write-Host "##vso[task.complete result=Failed;]Squash attestation FAILED for '$ModelPath'. Scan: $ScanStatus. See squash-attestation artifact for details."
    exit 1
}

Write-Host "##vso[task.complete result=Succeeded;]Squash attestation passed."
exit 0
