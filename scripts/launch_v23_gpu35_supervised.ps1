$ErrorActionPreference = "Stop"

$localConfig = "\\wsl.localhost\Ubuntu\root\projects\RecClaw_v2_0_Final_Reference\llm_api.md"
$localStdout = "\\wsl.localhost\Ubuntu\tmp\recclaw_v23_gpu35_ssh.stdout.log"
$localStderr = "\\wsl.localhost\Ubuntu\tmp\recclaw_v23_gpu35_ssh.stderr.log"

if (-not (Test-Path -LiteralPath $localConfig)) {
    throw "Authorized local LLM config is missing"
}

$remoteScript = @'
set -u
config=/dev/shm/recclaw_v23_llm_9225_runtime.md
supervisor_script=/dev/shm/recclaw_v23_supervisor_9225.sh
backend=/NAS2020/Workspaces/DMGroup/tingrangan/recclaw_gpu35_backend_v3_m6i
source=$backend/qualification_v15_v23/source
pilot_root=$backend/pilot_9225_v23
status=$backend/V23_PILOT_SUPERVISOR_STATUS.json
run_log=$backend/V23_PILOT_RUN.log
python=/NAS2020/Workspaces/DMGroup/tingrangan/recclaw_v15_backend_v1/runtime_exact_v2/bin/python
recbole=/NAS2020/Workspaces/DMGroup/tingrangan/recclaw_v15_backend_v1/recbole

cleanup() {
  rm -f "$config" "$supervisor_script"
}

on_signal() {
  cleanup
  printf '{"credential_residue":false,"exit_code":98,"reason":"SUPERVISOR_SIGNAL"}\n' > "$status"
  exit 98
}

trap cleanup EXIT
trap on_signal HUP INT TERM

if [ -e "$pilot_root" ]; then
  printf '{"credential_residue":false,"exit_code":96,"reason":"PILOT_ROOT_NOT_FRESH"}\n' > "$status"
  exit 96
fi

umask 077
cat > "$config"
chmod 600 "$config"

cd "$source" || exit 95
PYTHONPATH="$source:$source/src:$recbole" \
CUDA_VISIBLE_DEVICES=0 \
"$python" scripts/run_v23_pilot.py \
  --contract "$source/docs/research_line/continuous_program/V23_FROZEN_CHAIN_PILOT_CONTRACT.json" \
  --llm-api-config "$config" > "$run_log" 2>&1
exit_code=$?

rm -f "$config"
if [ -e "$config" ]; then
  residue=true
  if [ "$exit_code" -eq 0 ]; then
    exit_code=97
  fi
else
  residue=false
fi

printf '{"credential_residue":%s,"exit_code":%s,"reason":"NATURAL_PROCESS_EXIT"}\n' \
  "$residue" "$exit_code" > "$status"
exit "$exit_code"
'@

$encoded = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($remoteScript)
)
$remoteCommand = (
    "printf %s $encoded | base64 -d > /dev/shm/recclaw_v23_supervisor_9225.sh; " +
    "chmod 700 /dev/shm/recclaw_v23_supervisor_9225.sh; " +
    "bash /dev/shm/recclaw_v23_supervisor_9225.sh; " +
    "rc=`$?; rm -f /dev/shm/recclaw_v23_supervisor_9225.sh; exit `$rc"
)
$arguments = "-o BatchMode=yes gpu35 `"$remoteCommand`""

$process = Start-Process `
    -FilePath "ssh.exe" `
    -ArgumentList $arguments `
    -RedirectStandardInput $localConfig `
    -RedirectStandardOutput $localStdout `
    -RedirectStandardError $localStderr `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

exit $process.ExitCode
