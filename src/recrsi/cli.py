"""Public lifecycle for the final, bounded native Research Line."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import runpy
import sys
from .prepare import prepare, read
from .config import initialize, check_config, ADAPTERS


def readonly_inspection(event, args):
    if event == "open":
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            raise PermissionError("Inspection cannot write: " + str(args[0]))
    if event in ("subprocess.Popen", "os.system", "socket.connect", "os.mkdir", "os.remove", "os.rename", "os.chmod", "os.chown"):
        raise PermissionError("Inspection cannot execute this effect: " + event)


def enter(task_dir, *, execute=False, resume=False):
    if sys.platform != "linux":
        raise RuntimeError("The native runtime requires Linux filesystem/process isolation")
    task_dir = Path(task_dir).resolve(strict=True)
    task = read(task_dir/"TASK.json")
    entry = Path(task["entry"]).resolve(strict=True)
    entry.relative_to(task_dir)
    launch = read(task["launch"])
    if execute and (Path(launch["output"])/"STARTED.json").exists() and not resume:
        raise RuntimeError("Existing work requires explicit --resume; completed calls/fits will be reused")
    sys.dont_write_bytecode = True
    if not execute:
        # A missing dependency or an accidental effect fails inspection, without
        # waking a controller, probing the API, claiming resources or training.
        sys.addaudithook(readonly_inspection)
    with ExitStack() as stack:
        if execute:
            import fcntl
            cpus = set(task["cpus"])
            if not cpus <= os.sched_getaffinity(0):
                raise ValueError("Requested CPU allocation is outside the current process allocation")
            slots = Path(task["resource_slots"])
            paths={slots/("CPU%02d_%02d.lock" % (min(cpus),max(cpus))),slots/("CPU%d_%d.lock" % (min(cpus),max(cpus)))}
            for path in slots.glob("CPU*.lock"):
                match = re.fullmatch(r"CPU(\d+)_(\d+)\.lock", path.name)
                if match and cpus.intersection(range(int(match[1]),int(match[2])+1)):
                    paths.add(path)
            for path in sorted(paths):
                handle=stack.enter_context(path.open("a+b"))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            sys.path.insert(0,str(task_dir/"task/root_runtime"))
            from cpu_admission import occupied_cpus
            if cpus & occupied_cpus():
                raise BlockingIOError("Assigned CPU cores have an active worker; retry at its natural boundary")
            os.sched_setaffinity(0,cpus)
            os.environ.update(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",PYTHONDONTWRITEBYTECODE="1")
        sys.argv=[str(entry),"--repeat",str(task["repeat"])]+(["--execute"] if execute else [])
        runpy.run_path(str(entry),run_name="__main__")


def main():
    p=argparse.ArgumentParser(description="RecRSI: mechanism-guided recommendation research")
    sub=p.add_subparsers(dest="command",required=True)
    init=sub.add_parser("init",help="Create separate task and machine configuration templates")
    init.add_argument("--output",type=Path,required=True)
    init.add_argument("--adapter", choices=ADAPTERS, default=ADAPTERS[0])
    check=sub.add_parser("check",help="Validate input identities without API, GPU or worker execution")
    check.add_argument("--config",type=Path,required=True)
    prep=sub.add_parser("prepare",help="Bind frozen task inputs to a new run directory")
    prep.add_argument("--config",type=Path,required=True)
    prep.add_argument("--output",type=Path,required=True)
    inspect=sub.add_parser("inspect",help="Read-only assembly check; no API or worker")
    inspect.add_argument("task_dir",type=Path)
    run=sub.add_parser("run",help="Execute the bounded native research loop")
    run.add_argument("task_dir",type=Path)
    run.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.command=="init":
        print(json.dumps({"initialized":str(initialize(args.output, adapter=args.adapter))}))
    elif args.command=="check":
        print(json.dumps(check_config(args.config)))
    elif args.command=="prepare":
        print(json.dumps({"prepared":str(prepare(args.config,args.output))}))
    else:
        enter(args.task_dir,execute=args.command=="run",resume=getattr(args,"resume",False))

if __name__=="__main__":
    main()
