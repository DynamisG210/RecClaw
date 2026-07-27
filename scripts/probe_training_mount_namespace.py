#!/usr/bin/env python3
"""Probe the V8 private read-only mount namespace without training."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.training_filesystem import (
    filesystem_mount_audit,
    make_mount_tree_read_only,
    make_mount_writable,
    platform_gpu_device_mounts,
)


def main() -> None:
    device_mounts = tuple(
        Path(path) for path in platform_gpu_device_mounts()
    )
    writable_mounts = (
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/dev/shm"),
        Path("/proc/self"),
        *device_mounts,
    )
    subprocess.run(
        ["/usr/bin/mount", "--make-rprivate", "/"],
        check=True,
    )
    for path in writable_mounts:
        subprocess.run(
            ["/usr/bin/mount", "--bind", str(path), str(path)],
            check=True,
        )
    make_mount_tree_read_only(Path("/"))
    for path in writable_mounts:
        make_mount_writable(path)
    audit = filesystem_mount_audit(writable_mounts)
    print(
        json.dumps(
            {
                "device_mounts": [path.as_posix() for path in device_mounts],
                "mount_audit_status": audit["status"],
                "unexpected_writable_mount_targets": audit[
                    "unexpected_writable_mount_targets"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
