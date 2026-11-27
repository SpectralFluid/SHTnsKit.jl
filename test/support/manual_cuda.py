#!/usr/bin/env python3
"""Run CUDA parity on a manual server; publish its commit status separately.

Python 3.9+, git and Julia are required for run. Only report --publish uses gh.
The result and log are a portable record, not a signed hardware attestation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


SETUP = "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"
CUDA = '''using CUDA
CUDA.functional() || error("CUDA is not functional; refusing a skipped hardware run")
devices = collect(CUDA.devices())
length(devices) >= 1 || error("CUDA parity requires a visible GPU")
CUDA.versioninfo()
include("test/gpu/cuda/runtests.jl")
'''
MPI_RANK = '''using MPI, CUDA
MPI.Init()
MPI.Comm_size(MPI.COMM_WORLD) == 2 || error("MPI/CUDA parity requires two ranks")
CUDA.functional() || error("CUDA is not functional; refusing a skipped hardware run")
devices = collect(CUDA.devices())
length(devices) >= 2 || error("MPI/CUDA parity requires two visible GPUs on this box")
CUDA.device!(devices[MPI.Comm_rank(MPI.COMM_WORLD) + 1])
CUDA.versioninfo()
MPI.Barrier(MPI.COMM_WORLD)
include("test/gpu/cuda/mpi_runtests.jl")
'''
MPI_CUDA = ("using MPI\nrank_code = " + json.dumps(MPI_RANK) + "\n"
            "run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no "
            "--project=test/gpu/cuda -e $rank_code`)\n")
CONTEXTS = {"cuda": "CUDA / remote manual", "cuda+mpi": "CUDA + MPI / remote manual"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def log_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(command, cwd, env, log):
    log.write("\n$ " + shlex.join(map(str, command)) + "\n")
    log.flush()
    subprocess.run(command, cwd=cwd, env=env, stdout=log,
                   stderr=subprocess.STDOUT, check=True)


def run(args):
    repo = args.repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--end-of-options",
         args.ref + "^{commit}"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected a GitHub-compatible 40-character commit SHA")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if args.devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.devices
    # Resolve an explicitly relative executable before changing to the snapshot.
    julia = str(Path(args.julia).resolve()) if "/" in args.julia else args.julia
    result = {"schema": 1, "commit": commit, "suite": "cuda+mpi" if args.mpi else "cuda",
              "started_at": now(), "completed": False, "stages": [],
              "devices": env.get("CUDA_VISIBLE_DEVICES"), "julia": julia}
    result_path = output / "result.json"
    log_path = output / "cuda-test.log"
    save(result_path, result)
    print(f"Testing {commit}\nLog: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(result) + "\n")
        stage = "setup"
        try:
            with tempfile.TemporaryDirectory(prefix="shtnskit-cuda-") as tmp:
                snapshot = Path(tmp) / "source"
                # Independent objects and index: tests cannot modify the input checkout.
                execute(["git", "clone", "--no-hardlinks", "--no-checkout", "--",
                         str(repo), str(snapshot)], repo, env, log)
                execute(["git", "checkout", "--detach", commit], snapshot, env, log)
                command = [julia, "--startup-file=no", "--project=test/gpu/cuda", "-e"]
                for stage, code in [("setup", SETUP), ("cuda", CUDA)] + (
                        [("mpi_cuda", MPI_CUDA)] if args.mpi else []):
                    print(f"Running {stage} (output in {log_path})", flush=True)
                    execute(command + [code], snapshot, env, log)
                    result["stages"].append({"name": stage, "exit_code": 0})
                    save(result_path, result)
        except (OSError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
            code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
            result["stages"].append({"name": stage, "exit_code": code})
            log.write(f"\n{type(error).__name__}: {error}\n")
    result.update(completed=True, finished_at=now(), log_sha256=log_digest(log_path))
    save(result_path, result)
    state = result_state(result)
    print(f"{state}: {result_path}")
    return 0 if state == "success" else 1


def result_state(result):
    if (type(result.get("schema")) is not int or result["schema"] != 1
            or result.get("completed") is not True
            or result.get("suite") not in CONTEXTS
            or not re.fullmatch(r"[0-9a-f]{40}", str(result.get("commit", "")))):
        raise ValueError("invalid or incomplete manual CUDA result")
    expected = ["setup", "cuda"] + (["mpi_cuda"] if result["suite"] == "cuda+mpi" else [])
    stages = result.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= len(expected):
        raise ValueError("missing or unexpected result stages")
    for index, stage in enumerate(stages):
        if (not isinstance(stage, dict) or stage.get("name") != expected[index]
                or type(stage.get("exit_code")) is not int):
            raise ValueError("invalid result stage")
        if stage["exit_code"] != 0:
            if index != len(stages) - 1:
                raise ValueError("result continued after a failed stage")
            return "error" if index == 0 else "failure"
    if len(stages) != len(expected):
        raise ValueError("result is missing required tests")
    return "success"


def report(args):
    result = json.loads(args.result.read_text(encoding="utf-8"))
    state = result_state(result)
    if result.get("log_sha256") != log_digest(args.result.parent / "cuda-test.log"):
        raise ValueError("log is missing, changed or incompletely transferred")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        raise ValueError("repository must be OWNER/REPO")
    payload = {"state": state, "context": CONTEXTS[result["suite"]],
               "description": f"Manual {result['suite']} parity: {state}"}
    if args.target_url:
        if not args.target_url.startswith(("https://", "http://")):
            raise ValueError("target URL must use https:// or http://")
        payload["target_url"] = args.target_url
    endpoint = f"repos/{args.repository}/statuses/{result['commit']}"
    if not args.publish:
        print(json.dumps({"endpoint": endpoint, "payload": payload}, indent=2))
        return 0
    subprocess.run([args.gh, "api", "--method", "POST", endpoint, "--input", "-"],
                   input=json.dumps(payload), text=True, check=True)
    print(f"Published {state} for {result['commit']}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="test a committed snapshot; no GitHub access")
    runner.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    runner.add_argument("--ref", default="HEAD", help="commit or ref to test (default: HEAD)")
    runner.add_argument("--julia", default="julia", help="Julia executable (default: julia)")
    runner.add_argument("--devices", help="CUDA_VISIBLE_DEVICES; default: inherit environment")
    runner.add_argument("--output", type=Path, required=True, help="new directory for result and log")
    runner.add_argument("--mpi", action="store_true", help="also test two MPI ranks on two visible GPUs")
    runner.set_defaults(function=run)
    reporter = commands.add_parser("report", help="preview a GitHub commit status, or publish it")
    reporter.add_argument("result", type=Path, help="result.json beside its cuda-test.log")
    reporter.add_argument("--repository", default="SpectralFluid/SHTnsKit.jl", help="GitHub OWNER/REPO")
    reporter.add_argument("--target-url", help="optional URL to a separately uploaded log")
    reporter.add_argument("--gh", default="gh", help="GitHub CLI executable")
    reporter.add_argument("--publish", action="store_true", help="post the status using gh authentication")
    reporter.set_defaults(function=report)
    args = parser.parse_args()
    try:
        return args.function(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
