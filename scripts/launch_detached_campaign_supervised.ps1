param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_.@-]+$')]
    [string]$HostAlias,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$LocalLlmApiConfig,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_.-]+$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/')]
    [string]$RemoteSource,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/')]
    [string]$RemoteOutputRoot,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/')]
    [string]$RemoteStatusPath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/')]
    [string]$RemoteRunLog,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/')]
    [string]$RemoteSupervisorLog,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/')]
    [string]$RemotePython,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/')]
    [string]$RecBoleRoot,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_./-]+$')]
    [string]$RunnerRelativePath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_./-]+$')]
    [string]$ContractRelativePath,

    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 31)]
    [int]$CudaVisibleDevice
)

$ErrorActionPreference = "Stop"

function ConvertTo-ShellLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return "'" + $Value.Replace("'", "'`"`"'`"`'") + "'"
}

function Invoke-RemoteCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    $output = & ssh.exe -o BatchMode=yes $HostAlias $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
    return $output
}

$remoteConfig = "/dev/shm/recclaw_${RunId}_llm_runtime.md"
$remoteSupervisor = "/dev/shm/recclaw_${RunId}_supervisor.sh"

$qConfig = ConvertTo-ShellLiteral $remoteConfig
$qSupervisor = ConvertTo-ShellLiteral $remoteSupervisor
$qSource = ConvertTo-ShellLiteral $RemoteSource
$qOutputRoot = ConvertTo-ShellLiteral $RemoteOutputRoot
$qStatus = ConvertTo-ShellLiteral $RemoteStatusPath
$qRunLog = ConvertTo-ShellLiteral $RemoteRunLog
$qSupervisorLog = ConvertTo-ShellLiteral $RemoteSupervisorLog
$qPython = ConvertTo-ShellLiteral $RemotePython
$qRecBole = ConvertTo-ShellLiteral $RecBoleRoot
$qRunner = ConvertTo-ShellLiteral $RunnerRelativePath
$qContract = ConvertTo-ShellLiteral $ContractRelativePath
$qRunId = ConvertTo-ShellLiteral $RunId

$preflight = @"
set -eu
test ! -e $qOutputRoot
test ! -e $qStatus
test ! -e $qConfig
test ! -e $qSupervisor
test -d $qSource
test -x $qPython
test -d $qRecBole
test -d "`$(dirname $qStatus)"
test -d "`$(dirname $qRunLog)"
test -d "`$(dirname $qSupervisorLog)"
test -f $qSource/$qRunner
test -f $qSource/$qContract
"@

Invoke-RemoteCommand $preflight | Out-Null

$supervisorScript = @"
#!/usr/bin/env bash
set -u

config=$qConfig
supervisor_script=$qSupervisor
source_root=$qSource
output_root=$qOutputRoot
status_path=$qStatus
run_log=$qRunLog
python_bin=$qPython
recbole_root=$qRecBole
runner_path=$qRunner
contract_path=$qContract
run_id=$qRunId

write_status() {
  state="`$1"
  exit_code="`$2"
  reason="`$3"
  credential_residue="`$4"
  status_tmp="`$status_path.tmp.`$`$"
  printf '{"run_id":"%s","state":"%s","exit_code":%s,"reason":"%s","credential_residue":%s,"pid":%s}\n' \
    "`$run_id" "`$state" "`$exit_code" "`$reason" "`$credential_residue" "`$`$" > "`$status_tmp"
  mv -f -- "`$status_tmp" "`$status_path"
}

cleanup() {
  rm -f -- "`$config" "`$supervisor_script"
}

terminal_failure() {
  exit_code="`$1"
  reason="`$2"
  cleanup
  residue=false
  if [ -e "`$config" ]; then
    residue=true
  fi
  write_status TERMINAL "`$exit_code" "`$reason" "`$residue"
  exit "`$exit_code"
}

on_signal() {
  terminal_failure 98 SUPERVISOR_SIGNAL
}

trap cleanup EXIT
trap on_signal HUP INT TERM

if [ -e "`$output_root" ]; then
  terminal_failure 96 OUTPUT_ROOT_NOT_FRESH
fi
if [ ! -f "`$config" ]; then
  terminal_failure 95 LLM_CONFIG_MISSING
fi

config_mode=`$(stat -c '%a' "`$config")
if [ "`$config_mode" != "600" ]; then
  terminal_failure 94 LLM_CONFIG_MODE_NOT_600
fi

write_status RUNNING 0 DETACHED_SUPERVISOR_STARTED false

if ! cd "`$source_root"; then
  terminal_failure 93 SOURCE_ROOT_NOT_ACCESSIBLE
fi
set +e
PYTHONPATH="`$source_root:`$source_root/src:`$recbole_root" \
CUDA_VISIBLE_DEVICES=$CudaVisibleDevice \
"`$python_bin" "`$runner_path" \
  --contract "`$source_root/`$contract_path" \
  --llm-api-config "`$config" > "`$run_log" 2>&1
exit_code=`$?
set -e

rm -f -- "`$config"
residue=false
if [ -e "`$config" ]; then
  residue=true
  if [ "`$exit_code" -eq 0 ]; then
    exit_code=97
  fi
fi

write_status TERMINAL "`$exit_code" NATURAL_PROCESS_EXIT "`$residue"
exit "`$exit_code"
"@

$scriptBase64 = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($supervisorScript)
)
$qScriptBase64 = ConvertTo-ShellLiteral $scriptBase64

$configUploaded = $false
$detachedLaunched = $false
try {
    $uploadConfigCommand = "umask 077; cat > $qConfig; chmod 600 $qConfig"
    $uploadArguments = "-o BatchMode=yes $HostAlias `"$uploadConfigCommand`""
    $uploadProcess = Start-Process `
        -FilePath "ssh.exe" `
        -ArgumentList $uploadArguments `
        -RedirectStandardInput $LocalLlmApiConfig `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($uploadProcess.ExitCode -ne 0) {
        throw "LLM config upload failed with exit code $($uploadProcess.ExitCode)"
    }
    $configUploaded = $true

    $installSupervisor = @"
set -eu
printf %s $qScriptBase64 | base64 -d > $qSupervisor
chmod 700 $qSupervisor
"@
    Invoke-RemoteCommand $installSupervisor | Out-Null

    $launchCommand = @"
set -eu
setsid nohup bash $qSupervisor > $qSupervisorLog 2>&1 < /dev/null &
printf '%s\n' "`$!"
"@
    $pidOutput = @(Invoke-RemoteCommand $launchCommand)
    $remotePid = ($pidOutput | Select-Object -Last 1).Trim()
    if ($remotePid -notmatch '^[0-9]+$') {
        throw "Detached launch did not return a numeric PID"
    }
    $detachedLaunched = $true

    Start-Sleep -Seconds 1
    $verifyCommand = @"
set -eu
test -s $qStatus
cat $qStatus
kill -0 $remotePid
"@
    $statusJson = @(Invoke-RemoteCommand $verifyCommand) | Select-Object -First 1
    $status = $statusJson | ConvertFrom-Json
    if ($status.run_id -ne $RunId -or $status.state -ne "RUNNING") {
        throw "Detached supervisor did not reach the RUNNING state"
    }

    [pscustomobject]@{
        host = $HostAlias
        run_id = $RunId
        remote_pid = [int64]$remotePid
        state = $status.state
        status_path = $RemoteStatusPath
        output_root = $RemoteOutputRoot
        credential_mode = 600
        ssh_session_coupled = $false
    } | ConvertTo-Json -Depth 4
}
catch {
    if ($configUploaded -and -not $detachedLaunched) {
        $cleanupCommand = "rm -f -- $qConfig $qSupervisor"
        try {
            Invoke-RemoteCommand $cleanupCommand | Out-Null
        }
        catch {
            Write-Warning "Remote launch cleanup could not be confirmed"
        }
    }
    throw
}
