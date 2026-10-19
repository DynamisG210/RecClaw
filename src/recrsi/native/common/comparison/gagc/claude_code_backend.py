from __future__ import annotations

import errno
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


ValidationCallback = Callable[[], Optional[str]]


@dataclass
class ClaudeAttempt:
    attempt: int
    command: list[str]
    returncode: int | None
    wall_time_secs: float
    timed_out: bool = False
    error: str | None = None
    validation_error: str | None = None
    train_returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    claude_stdout_tail: str = ""
    claude_stderr_tail: str = ""


@dataclass
class ClaudeTrialOutcome:
    success: bool
    wall_time_secs: float
    timed_out: bool = False
    oom: bool = False
    error_message: str | None = None
    train_stdout: str = ""
    train_stderr: str = ""
    attempts: list[ClaudeAttempt] = field(default_factory=list)
    artifact_dir: str = ""
    changed_files: list[str] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "wall_time_secs": self.wall_time_secs,
            "timed_out": self.timed_out,
            "oom": self.oom,
            "error_message": self.error_message,
            "artifact_dir": self.artifact_dir,
            "changed_files": self.changed_files,
            "stdout_tail": self.train_stdout[-3000:],
            "stderr_tail": self.train_stderr[-2000:],
            "attempts": [attempt.__dict__ for attempt in self.attempts],
        }


def claude_backend_enabled() -> bool:
    return os.getenv("GAGC_ENABLE_CLAUDE_BACKEND", "").strip().lower() in {"1", "true", "yes", "on"}


_ENOEXEC_MAX_RETRIES = 5
_ENOEXEC_BACKOFF_SECS = 2.0


def _run_claude_subprocess(
    command: list[str],
    cwd: str,
    env: dict[str, str],
    prompt: str,
    timeout: float,
) -> subprocess.CompletedProcess:
    """Run the claude subprocess, retrying transient ENOEXEC.

    Claude Code's auto-updater rewrites bin/claude.exe in place (its install.cjs
    copies the native binary over the placeholder). A spawn that execve's the
    half-written file gets ENOEXEC (errno 8); the binary is valid again moments
    later, so back off and retry instead of failing the trial. The trial env
    sets DISABLE_AUTOUPDATER to stop trials from triggering such updates; this
    retry also covers updates triggered by other claude processes that share the
    install.
    """
    last_exc: OSError | None = None
    for spawn_attempt in range(_ENOEXEC_MAX_RETRIES):
        try:
            return subprocess.run(
                command,
                cwd=cwd,
                env=env,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            raise
        except subprocess.TimeoutExpired:
            raise
        except OSError as exc:
            last_exc = exc
            if exc.errno != errno.ENOEXEC or spawn_attempt == _ENOEXEC_MAX_RETRIES - 1:
                raise
            time.sleep(_ENOEXEC_BACKOFF_SECS * (spawn_attempt + 1))
    assert last_exc is not None
    raise last_exc  # unreachable: the loop always returns or raises above


def run_claude_code_trial(
    *,
    trial_dir: str,
    train_path: str,
    predict_path: str,
    spec_data: dict[str, Any],
    env: dict[str, str],
    hard_timeout: float,
    benchmark_mode: str,
    validation_callback: ValidationCallback,
) -> ClaudeTrialOutcome:
    """Let Claude Code implement and run one isolated trial workspace.

    The caller remains responsible for parsing benchmark metrics from train_stdout
    and for promoting any winning code. This function never writes outside trial_dir.
    """

    start = time.time()
    artifact_dir = os.path.join(trial_dir, ".gagc_claude")
    os.makedirs(artifact_dir, exist_ok=True)

    max_attempts = _coerce_int(
        spec_data.get("claude_attempts", os.getenv("GAGC_CLAUDE_ATTEMPTS", "3")),
        default=3,
        low=1,
        high=10,
    )
    claude_bin = str(spec_data.get("claude_bin") or os.getenv("GAGC_CLAUDE_BIN") or "claude")
    claude_model = str(spec_data.get("claude_model") or os.getenv("GAGC_CLAUDE_MODEL") or "")
    permission_mode = str(
        spec_data.get("claude_permission_mode")
        or os.getenv("GAGC_CLAUDE_PERMISSION_MODE")
        or "acceptEdits"
    )
    allowed_tools = str(
        spec_data.get("claude_allowed_tools")
        or os.getenv("GAGC_CLAUDE_ALLOWED_TOOLS")
        or "Bash Edit Read Write Glob Grep"
    ).strip()
    allowed_files = _normalise_allowed_files(spec_data, benchmark_mode)
    base_snapshot = _snapshot_workspace(trial_dir)
    feedback = ""
    attempts: list[ClaudeAttempt] = []
    last_error = "claude_code_failed"
    timed_out = False
    oom = False

    for attempt_idx in range(1, max_attempts + 1):
        prompt = _build_prompt(
            trial_dir=trial_dir,
            train_path=train_path,
            predict_path=predict_path,
            spec_data=spec_data,
            benchmark_mode=benchmark_mode,
            allowed_files=allowed_files,
            env=env,
            feedback=feedback,
            attempt_idx=attempt_idx,
            max_attempts=max_attempts,
        )
        command = [
            claude_bin,
            "-p",
            "--permission-mode",
            permission_mode,
            "--output-format",
            "json",
        ]
        if allowed_tools:
            command.extend(["--allowedTools", allowed_tools])
        if claude_model:
            command.extend(["--model", claude_model])

        attempt_start = time.time()
        try:
            proc = _run_claude_subprocess(command, trial_dir, env, prompt, hard_timeout)
            attempt = ClaudeAttempt(
                attempt=attempt_idx,
                command=_redact_command(command),
                returncode=proc.returncode,
                wall_time_secs=time.time() - attempt_start,
                claude_stdout_tail=proc.stdout[-3000:],
                claude_stderr_tail=proc.stderr[-2000:],
            )
        except FileNotFoundError:
            last_error = f"claude binary not found: {claude_bin}"
            return ClaudeTrialOutcome(
                success=False,
                wall_time_secs=time.time() - start,
                error_message=last_error,
                attempts=attempts,
                artifact_dir=artifact_dir,
                changed_files=_changed_files(base_snapshot, _snapshot_workspace(trial_dir)),
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            last_error = "claude_code_timeout"
            attempt = ClaudeAttempt(
                attempt=attempt_idx,
                command=_redact_command(command),
                returncode=None,
                wall_time_secs=time.time() - attempt_start,
                timed_out=True,
                error=last_error,
                claude_stdout_tail=_decode_maybe(exc.stdout)[-3000:],
                claude_stderr_tail=_decode_maybe(exc.stderr)[-2000:],
            )
            attempts.append(attempt)
            break
        except OSError as exc:
            # Persistent exec-format error after the ENOEXEC retries inside
            # _run_claude_subprocess: the claude binary did not recover (e.g. a
            # broken in-place update). Fail the trial cleanly instead of letting
            # the OSError escape to the trial-group caller.
            last_error = f"claude binary not executable (errno {exc.errno}): {claude_bin}"
            return ClaudeTrialOutcome(
                success=False,
                wall_time_secs=time.time() - start,
                error_message=last_error,
                attempts=attempts,
                artifact_dir=artifact_dir,
                changed_files=_changed_files(base_snapshot, _snapshot_workspace(trial_dir)),
            )

        stdout, stderr, train_rc = _read_training_artifacts(artifact_dir)
        attempt.train_returncode = train_rc
        attempt.stdout_tail = stdout[-3000:]
        attempt.stderr_tail = stderr[-2000:]

        validation_error = validation_callback()
        changed_files = _changed_files(base_snapshot, _snapshot_workspace(trial_dir))
        boundary_error = _validate_changed_files(changed_files, allowed_files)
        metric_error = _validate_training_artifacts(
            stdout=stdout,
            stderr=stderr,
            train_returncode=train_rc,
            benchmark_mode=benchmark_mode,
        )
        claude_error = None
        if proc.returncode != 0:
            claude_error = f"claude exited with code {proc.returncode}: {proc.stderr[-1000:]}"
        combined_error = _first_error(claude_error, validation_error, boundary_error, metric_error)
        attempt.validation_error = combined_error
        attempts.append(attempt)

        if "CUDA out of memory" in stderr or "OutOfMemoryError" in stderr:
            oom = True

        _write_attempt_log(artifact_dir, attempts, changed_files)
        if combined_error is None:
            return ClaudeTrialOutcome(
                success=True,
                wall_time_secs=time.time() - start,
                train_stdout=stdout,
                train_stderr=stderr,
                attempts=attempts,
                artifact_dir=artifact_dir,
                changed_files=changed_files,
            )

        last_error = combined_error
        feedback = _build_feedback(combined_error, stdout, stderr, attempt)

    stdout, stderr, _ = _read_training_artifacts(artifact_dir)
    changed_files = _changed_files(base_snapshot, _snapshot_workspace(trial_dir))
    return ClaudeTrialOutcome(
        success=False,
        wall_time_secs=time.time() - start,
        timed_out=timed_out,
        oom=oom,
        error_message=f"claude_code_failed_after_{len(attempts)}_attempts: {last_error}",
        train_stdout=stdout,
        train_stderr=stderr,
        attempts=attempts,
        artifact_dir=artifact_dir,
        changed_files=changed_files,
    )


def _coerce_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(high, max(low, parsed))


def _normalise_allowed_files(spec_data: dict[str, Any], benchmark_mode: str) -> list[str]:
    raw = spec_data.get("allowed_files") or spec_data.get("claude_allowed_files")
    if isinstance(raw, str):
        files = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        files = [str(item).strip() for item in raw if str(item).strip()]
    else:
        files = []
    if files:
        return sorted({_clean_relpath(path) for path in files})
    if benchmark_mode == "kuairec":
        return [
            "train.py",
            "model/decoder.py",
            "model/encoder.py",
            "model/transformer.py",
        ]
    return ["train.py", "predict.py"]


def _build_prompt(
    *,
    trial_dir: str,
    train_path: str,
    predict_path: str,
    spec_data: dict[str, Any],
    benchmark_mode: str,
    allowed_files: list[str],
    env: dict[str, str],
    feedback: str,
    attempt_idx: int,
    max_attempts: int,
) -> str:
    arm = spec_data.get("arm", spec_data.get("dimension", ""))
    hypothesis = spec_data.get("hypothesis") or spec_data.get("code_hint") or ""
    agent_hypothesis = str(spec_data.get("agent_hypothesis") or "").strip()
    memory_context = spec_data.get("gagc_memory_context") if isinstance(spec_data.get("gagc_memory_context"), dict) else {}
    experiment_skill = str(memory_context.get("experiment_skill") or "").strip()
    recent_text_gradients = memory_context.get("recent_text_gradients") or []
    failure_memory = memory_context.get("failure_memory") or []
    skill_notes = str(memory_context.get("skill_notes") or "").strip()
    implementation_prompt = str(spec_data.get("implementation_prompt") or "").strip()
    train_command = str(spec_data.get("claude_train_command") or "").strip()
    if not train_command:
        train_command = _default_train_command(env)
    required_metrics = "XAUC= and MAE=" if benchmark_mode == "kuairec" else "a zero training exit code"
    feedback_block = feedback or "No previous failure; this is the first attempt."
    return f"""
You are implementing one isolated RecHarness trial inside this trial workspace:
{trial_dir}

Attempt {attempt_idx}/{max_attempts}.
Benchmark mode: {benchmark_mode}
Arm: {arm}
Dimension: {spec_data.get('dimension', '')}
Delta: {spec_data.get('delta', 0.0)}
Hypothesis: {hypothesis}
Agent textual-gradient hypothesis: {agent_hypothesis or '(none)'}

RecHarness memory context:
Skill notes: {skill_notes or '(none)'}
Recent text gradients: {recent_text_gradients or '(none)'}
Failure memory / avoids: {failure_memory or '(none)'}
ExperimentSkill:
{experiment_skill or '(none)'}

Additional implementation instructions:
{implementation_prompt or '(none)'}

Strict boundaries:
- Modify ONLY these relative files: {', '.join(allowed_files)}
- Do not use git, do not edit parent directories, and do not edit benchmark/harness/state/promotion logic.
- Keep changes minimal and focused on this trial.
- Preserve the training script entrypoint and metric printing contract.
- Use the selected arm, agent hypothesis, and RecHarness memory to decide the concrete edit.
- Do not change training horizon semantics unless the selected arm explicitly requires it.

After editing, you MUST run the full proxy training command below from the current directory and save artifacts exactly as shown:
mkdir -p .gagc_claude
{train_command} > .gagc_claude/train_stdout.log 2> .gagc_claude/train_stderr.log
echo $? > .gagc_claude/train_returncode.txt

Success requires: Python/contract checks pass, training return code is 0, and stdout contains {required_metrics}.
If the command fails, inspect .gagc_claude/train_stderr.log and .gagc_claude/train_stdout.log, fix the code, and rerun it before finishing.

Failure feedback from previous attempt:
{feedback_block}

Finish with a concise JSON summary of files changed and final metrics/errors. Do not ask questions.
""".strip()


def _default_train_command(env: dict[str, str]) -> str:
    assignments: list[str] = []
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "GAGC_SLOT_ID",
        "GAGC_DATA_DIR",
        "GAGC_TEST_DIR",
        "SASREC_EPOCHS",
        "GRU4REC_EPOCHS",
        "BERT4REC_EPOCHS",
        "NEXTITNET_EPOCHS",
        "HSTU_EPOCHS",
        "GR_TRAIN_DATA",
        "GR_TEST_DATA",
        "GR_NUM_EPOCHS",
    ):
        value = env.get(key)
        if value:
            assignments.append(f"{key}={shlex.quote(str(value))}")
    if (env.get("GR_TRAIN_DATA") or env.get("GR_TEST_DATA")) and "GR_NUM_EPOCHS" not in env:
        assignments.append("GR_NUM_EPOCHS=2")
    python_bin = shlex.quote(sys.executable)
    prefix = " ".join(assignments)
    return f"{prefix} {python_bin} train.py".strip()


def _read_training_artifacts(artifact_dir: str) -> tuple[str, str, int | None]:
    stdout = _read_text(os.path.join(artifact_dir, "train_stdout.log"))
    stderr = _read_text(os.path.join(artifact_dir, "train_stderr.log"))
    rc_raw = _read_text(os.path.join(artifact_dir, "train_returncode.txt")).strip()
    try:
        returncode = int(rc_raw.splitlines()[-1]) if rc_raw else None
    except (ValueError, IndexError):
        returncode = None
    return stdout, stderr, returncode


def _validate_training_artifacts(
    *,
    stdout: str,
    stderr: str,
    train_returncode: int | None,
    benchmark_mode: str,
) -> str | None:
    if train_returncode is None:
        return "missing .gagc_claude/train_returncode.txt; Claude Code did not run the required training command"
    if train_returncode != 0:
        return f"training command failed with exit code {train_returncode}: {stderr[-1000:]}"
    if benchmark_mode == "kuairec" and ("XAUC=" not in stdout or "MAE=" not in stdout):
        return "training stdout is missing required KuaiRec metrics XAUC= and MAE="
    return None


def _snapshot_workspace(trial_dir: str) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for root, dirs, files in os.walk(trial_dir):
        dirs[:] = [d for d in dirs if d not in {".gagc_claude", "__pycache__", ".git"}]
        for filename in files:
            path = os.path.join(root, filename)
            rel = _clean_relpath(os.path.relpath(path, trial_dir))
            if _is_ignored_artifact(rel):
                continue
            try:
                with open(path, "rb") as handle:
                    snapshot[rel] = hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                snapshot[rel] = "<unreadable>"
    return snapshot


def _changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    keys = set(before) | set(after)
    return sorted(path for path in keys if before.get(path) != after.get(path))


def _validate_changed_files(changed_files: list[str], allowed_files: list[str]) -> str | None:
    allowed = {_clean_relpath(path) for path in allowed_files}
    forbidden = [path for path in changed_files if path not in allowed]
    if forbidden:
        return f"Claude Code modified files outside allowed_files: {forbidden}"
    return None


def _is_ignored_artifact(relpath: str) -> bool:
    name = os.path.basename(relpath)
    if name in {"best_model.pth", "train_stdout.log", "train_stderr.log", "train_returncode.txt"}:
        return True
    if name.endswith((".pth", ".pt", ".pkl", ".pyc")):
        return True
    if relpath.startswith("working/") and name == "predict.py":
        return True
    if relpath.startswith(".gagc_claude/") or "/__pycache__/" in f"/{relpath}/":
        return True
    return False


def _clean_relpath(path: str) -> str:
    return path.replace(os.sep, "/").lstrip("./")


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _write_attempt_log(artifact_dir: str, attempts: list[ClaudeAttempt], changed_files: list[str]) -> None:
    payload = {
        "changed_files": changed_files,
        "attempts": [attempt.__dict__ for attempt in attempts],
    }
    try:
        with open(os.path.join(artifact_dir, "attempts.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except OSError:
        pass


def _build_feedback(error: str, stdout: str, stderr: str, attempt: ClaudeAttempt) -> str:
    return "\n".join([
        f"Validation error: {error}",
        f"Claude return code: {attempt.returncode}",
        f"Training return code: {attempt.train_returncode}",
        f"Training stdout tail:\n{stdout[-2000:]}",
        f"Training stderr tail:\n{stderr[-2000:]}",
        f"Claude stderr tail:\n{attempt.claude_stderr_tail[-1000:]}",
    ])


def _first_error(*errors: str | None) -> str | None:
    for error in errors:
        if error:
            return error
    return None


def _decode_maybe(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    if redacted:
        redacted[-1] = redacted[-1][:2000]
    return redacted
