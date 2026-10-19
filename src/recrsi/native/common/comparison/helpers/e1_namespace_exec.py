#!/usr/bin/env python3
"""Trusted same-host OS launcher. No candidate code runs before privilege drop."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_landlock_exec import gpu_device_for_uuid, main as landlock_main

MS_RDONLY, MS_REMOUNT, MS_BIND = 1, 32, 4096
CLONE_NEWNS, CLONE_NEWIPC, CLONE_NEWNET = 0x20000, 0x8000000, 0x40000000


def mount(source, target, flags):
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.mount(source.encode() if source else None, os.fsencode(target),
                        None, ctypes.c_ulong(flags), None)
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def readonly_bind(path):
    # The direct syscall works on GPU41's inherited ext4/NFS mounts where the
    # util-linux remount command failed. Keep the exact paths and source bytes.
    mount(str(path), path, MS_BIND)
    mount(None, path, MS_BIND | MS_REMOUNT | MS_RDONLY)
    if not os.statvfs(path).f_flag & os.ST_RDONLY:
        raise RuntimeError(f'read-only mount was not installed: {path}')


def main():
    if os.geteuid() != 0:
        raise RuntimeError('namespace setup requires the authorized sudo launcher')
    job = json.loads(Path(sys.argv[1]).read_text())
    uid, gid = job['uid'], job['gid']
    if uid <= 0 or gid <= 0:
        raise ValueError('candidate worker must use an unprivileged UID/GID')
    environment = dict(job['environment'])
    for kind in ('net', 'ipc', 'pid'):
        environment[f'E1_HOST_{kind.upper()}_NS'] = str(os.stat(f'/proc/self/ns/{kind}').st_ino)
    gpu_node = gpu_device_for_uuid(environment['CUDA_VISIBLE_DEVICES'])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(CLONE_NEWNS | CLONE_NEWIPC | CLONE_NEWNET):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Protect every readable disk input, including code/runtime on ABI 1 where
    # Landlock alone cannot mediate O_RDONLY|O_TRUNC. Never remount the host view.
    for path in job['readonly']:
        readonly_bind(path)
    mount(job['private_shm'], '/dev/shm', MS_BIND)
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    # sudo may supervise this process. Do not leave a worker behind if its
    # launcher is killed by the native timeout/cancellation path.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
        raise RuntimeError('worker parent-death binding failed')
    # Written before candidate imports, outside its allowlist. The supervisor
    # must distinguish this exec-stable worker PID from sudo's monitor PID.
    pid_path = Path(job['worker_pid_path'])
    temporary_pid = pid_path.with_suffix('.tmp')
    temporary_pid.write_text(str(os.getpid()))
    os.replace(temporary_pid, pid_path)
    os.environ.clear()
    os.environ.update(environment)
    readonly = job['readonly'] + [
        '/proc/self', '/proc/cpuinfo', '/proc/devices', '/proc/driver/nvidia',
        '/proc/sys/vm/mmap_min_addr', '/sys/class/net', '/sys/bus/pci',
        '/sys/devices', '/sys/module/nvidia', '/dev/null', '/dev/zero', '/dev/urandom',
    ]
    writable = job['writable'] + [str(gpu_node), '/dev/nvidiactl',
        '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools', '/dev/shm']
    arguments = ['--require-uid', str(uid), '--workdir', job['cwd']]
    for path in readonly:
        arguments.extend(('--ro', path))
    for path in writable:
        arguments.extend(('--rw', path))
    return landlock_main(arguments + ['--'] + job['command'])


if __name__ == '__main__':
    main()
