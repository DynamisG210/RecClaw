# Linux host setup

The CPU example needs only Python and the training dependencies. Research runs
also use CUDA, Linux namespaces, Landlock, a memory cgroup, and an unprivileged
worker UID (65534). The allocator reserves 9 GiB of GPU memory per worker and
four CPU cores. All local runs must share the same resource-lock directory.

## Training environment

```bash
python3.10 -m venv /path/to/training-env
source /path/to/training-env/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[train,test]'
python -c 'import sys, torch, recbole; from pathlib import Path; print(sys.executable); print(Path(recbole.__file__).parent.parent); print(torch.__version__, torch.cuda.is_available())'
nvidia-smi --query-gpu=uuid,memory.free --format=csv
python -c 'import os; print(sorted(os.sched_getaffinity(0)))'
mkdir -p /path/to/shared-resource-slots
```

Use the printed interpreter and RecBole location for `worker_python` and
`recbole_root`. Use four available CPU IDs and a GPU UUID. `driver_root` must
contain `nvidia-smi` and the installed NVIDIA libraries, including
`libnvidia-ml.so.1` and `libcuda.so.1`; it may be a directory of symlinks to the
driver installation. On Ubuntu, locate the libraries with
`ldconfig -p | grep -E 'libcuda.so.1|libnvidia-ml.so.1'`.

The worker UID must be able to traverse the project, data, interpreter, and
dependency directories. Use readable shared directories, not a private home
directory. Keep the API credential file outside these worker-readable roots.

## Worker isolation

The trusted controller starts the namespace helper with
`sudo -n /usr/bin/python3 -I <helper> <job.json>`. The helper mounts inputs
read-only, installs the memory limit, isolates networking, and drops to UID
65534 before running candidate code. The watchdog uses the same privileged
interpreter for exact-process termination.

Provision this on a dedicated research host or VM whose controller account is
trusted to administer the machine. An administrator can grant the controller
account passwordless `/usr/bin/python3` execution with `visudo`. This permission
grants administrative code execution; do not grant it to candidate programs or
untrusted users. Candidate workers receive neither sudo access nor credentials.

The host must expose Landlock and allow mount/network/IPC namespaces. Mounts
visible to the helper must have private or slave propagation. The memory
cgroup controller must be writable by the privileged helper. Both memory
cgroup v1 and unified v2 layouts are supported. For v2, enable `memory` in
the parent's `cgroup.subtree_control` during host provisioning. Run the local
preflight after filling `environment.json`:

```bash
recrsi doctor --environment /path/to/project/environment.json
```

The first `baseline --device cuda` exercises the actual isolation, GPU allocation,
training, checkpoint reload, and cost settlement. Its `BOUNDARY.json` and
`ROOT_CALIBRATION.json` retain the execution receipts.

## Model endpoint

Create a private text file, mode 600, outside the repository and worker roots:

```text
base_url: https://YOUR_ENDPOINT/v1
api_key: YOUR_KEY
```

Set `api_config` to its path. The released method profile uses model identities
`gpt-5.6-luna` and `gpt-5.6-terra` through Responses and chat-completions. These
are the configured experiment identities; the endpoint must serve them.
Request/response bytes, actual returned model metadata, token counts, and
reference costs are saved in each run. Reference rates are fixed accounting
inputs: luna input/output 0.2/1.2 and terra 2/12 USD per million tokens.

## Troubleshooting

- Dependency error: run `python -m pip check` in the selected interpreter.
- Namespace or cgroup error: inspect `doctor` output and the baseline worker log; configure the host before launching search.
- Baseline mismatch: retain the measured files together; changing data, source, seed, or training recipe requires a new project and baseline.
- Resource busy: let the current holder finish, then rerun the command. Use the same lock directory across controllers.
- Interrupted search: inspect `PAUSED.json` and the existing ledgers before `run --resume`; completed results stay in the original run.
