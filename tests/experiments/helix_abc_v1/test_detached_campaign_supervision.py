from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "scripts/launch_detached_campaign_supervised.ps1"


def _launcher_source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_detached_launcher_does_not_bind_campaign_lifetime_to_ssh() -> None:
    source = _launcher_source()

    assert "setsid nohup bash $qSupervisor" in source
    assert "> $qSupervisorLog 2>&1 < /dev/null &" in source
    assert "$remotePid = ($pidOutput | Select-Object -Last 1).Trim()" in source
    assert "kill -0 $remotePid" in source
    assert "ssh_session_coupled = $false" in source


def test_detached_launcher_preserves_fresh_run_and_secret_boundaries() -> None:
    source = _launcher_source()

    assert "test ! -e $qOutputRoot" in source
    assert "test ! -e $qStatus" in source
    assert "test ! -e $qConfig" in source
    assert "test ! -e $qSupervisor" in source
    assert "-RedirectStandardInput $LocalLlmApiConfig" in source
    assert "chmod 600 $qConfig" in source
    assert "config_mode=`$(stat -c '%a' \"`$config\")" in source
    assert 'if [ "`$config_mode" != "600" ]; then' in source
    assert "trap cleanup EXIT" in source
    assert "trap on_signal HUP INT TERM" in source


def test_detached_launcher_status_is_atomic_and_launch_safe() -> None:
    source = _launcher_source()

    assert 'status_tmp="`$status_path.tmp.`$`$"' in source
    assert 'mv -f -- "`$status_tmp" "`$status_path"' in source
    assert "write_status RUNNING 0 DETACHED_SUPERVISOR_STARTED false" in source
    assert "$detachedLaunched = $true" in source
    assert "if ($configUploaded -and -not $detachedLaunched)" in source
    assert '$status.run_id -ne $RunId -or $status.state -ne "RUNNING"' in source
