"""Privileged signal of one pidfd-identified worker with matching start ticks."""
import ctypes
import os
from pathlib import Path
import signal
import sys


def syscall(number, *args):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    value = libc.syscall(ctypes.c_long(number), *args)
    if value < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return value


def main():
    pid, expected = map(int, sys.argv[1:])
    assert os.uname().machine == 'x86_64' and os.geteuid() == 0 and pid > 1
    try:
        fd = syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    except ProcessLookupError:
        return
    try:
        try:
            ticks = int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
        except FileNotFoundError:
            return
        if ticks != expected:
            raise ValueError('Worker start identity differs; no signal sent')
        try:
            syscall(424, ctypes.c_int(fd), ctypes.c_int(signal.SIGKILL), ctypes.c_void_p(), ctypes.c_uint(0))
        except ProcessLookupError:
            pass
    finally:
        os.close(fd)


if __name__ == '__main__': main()
