#!/usr/bin/env python3
"""Run the remaining final GreenSA BookSim2 workloads safely and resumably."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "power_model/final_experiments/booksim2_128gb"
CONFIG = ROOT / "configs/systolic_ws_128x128_c4_booksim2_big_model.json"
SIMULATOR = ROOT / "build/bin/Simulator"
COMPARE = ROOT / "power_model/shared_sa_comparison.py"
SUMMARIZE_IDLE = ROOT / "power_model/summarize_sa_region_profile.py"
TRACE = ROOT / "traces/input_32_8.csv"
LAUNCH_LOCK = threading.Lock()

OLD_ENV = (
    "ONNXSIM_REGATE_FUNCTIONAL_PROFILE",
    "ONNXSIM_REGATE_TWO_BANK_PROFILE",
    "ONNXSIM_REGATE_ROW_PRELOAD_PROFILE",
    "ONNXSIM_REGATE_EVENT_DRIVEN_PROFILE",
    "ONNXSIM_REGATE_UPDATED_PROFILE",
    "ONNXSIM_SA_REGION_PROFILE",
    "ONNXSIM_SA_REGION_SIZES",
    "ONNXSIM_SHARED_SA_POLICY",
)

# Deadline-oriented order: finish inexpensive, diverse workloads first so a
# partial run still yields a useful paper table.  Large language workloads are
# deliberately left until the end.
WORKLOADS = (
    ("alexnet", "default", None, "alexnet"),
    ("vgg11", "default", None, "vgg11"),
    ("resnet34", "default", None, "resnet34"),
    ("resnet50", "default", None, "resnet50"),
    ("vgg16", "default", None, "vgg16"),
    ("vgg19", "default", None, "vgg19"),
    ("resnet101", "default", None, "resnet101"),
    ("wide_resnet50_2", "default", None, "wide_resnet50_2"),
    ("gpt2", "default", "power_model/workload_suite/model_lists/gpt2_1.json", None),
    ("opt_125m_full", "language", "power_model/workload_suite/model_lists/opt_125m_12l_full_32_8.json", None),
    ("opt_1_3b", "language", "power_model/workload_suite/model_lists/opt_1_3b_32_8.json", None),
    ("opt_2_7b", "language", "power_model/workload_suite/model_lists/opt_2_7b_32_8.json", None),
    ("opt_6_7b", "language", "power_model/workload_suite/model_lists/opt_6_7b_32_8.json", None),
    ("llama3_8b", "language", "power_model/workload_suite/model_lists/llama3_8b_32_8.json", None),
)


def load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def validate_architecture() -> None:
    config = load_json(CONFIG)
    assert config["num_cores"] == 4
    assert config["dram_type"] == "ramulator2"
    assert config["dram_size"] == 128
    assert config["dram_channels"] == 16
    assert config["icnt_type"] == "booksim2"
    assert config["icnt_freq"] == 1000
    assert config["icnt_injection_ports_per_core"] == 16
    assert config["precision"] == 2
    for core_id in range(4):
        core = config["core_config"][f"core_{core_id}"]
        assert core["core_type"] == "systolic_ws"
        assert core["core_width"] == 128
        assert core["core_height"] == 128
        assert core["spad_size"] == 32768
        assert core["accum_spad_size"] == 4096


def generated_list(name: str) -> Path:
    directory = FINAL_ROOT / "generated_model_lists"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_1.json"
    expected = {"models": [{"name": name, "batch_size": 1}]}
    if path.exists() and load_json(path) != expected:
        raise RuntimeError(f"refusing to overwrite inconsistent {path}")
    path.write_text(json.dumps(expected, indent=2) + "\n")
    return path


def resolve_workloads(selected: set[str] | None):
    resolved = []
    for label, mode, list_name, onnx_name in WORKLOADS:
        if selected is not None and label not in selected:
            continue
        model_list = ROOT / list_name if list_name else generated_list(onnx_name)
        load_json(model_list)
        if onnx_name:
            model_file = ROOT / f"models/{onnx_name}/{onnx_name}.onnx"
            if not model_file.is_file() or model_file.stat().st_size == 0:
                raise RuntimeError(f"missing ONNX model: {model_file}")
        resolved.append((label, mode, model_list))
    if selected is not None:
        found = {item[0] for item in resolved}
        unknown = selected - found
        if unknown:
            raise RuntimeError(f"unknown workload(s): {sorted(unknown)}")
    return resolved


def simulator_count() -> int:
    count = 0
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"build/bin/Simulator" in command:
            count += 1
    return count


def base_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in OLD_ENV:
        env.pop(key, None)
    env["ONNXIM_HOME"] = str(ROOT)
    return env


def simulator_command(mode: str, model_list: Path) -> list[str]:
    command = [
        str(SIMULATOR), "--config", str(CONFIG),
        "--models_list", str(model_list), "--mode", mode,
    ]
    if mode == "language":
        command += ["--trace_file", "input_32_8.csv"]
    return command


def log_valid(path: Path, marker: str) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(errors="replace")
    failures = ("segmentation fault", "floating point exception", "[error]",
                'validation_pass":false')
    return text.count(marker) == 4 and not any(x in text.lower() for x in failures)


def task_complete(directory: Path, task: str) -> bool:
    if task == "idleness":
        return ((directory / "idleness/status.txt").read_text().strip() == "PASS"
                if (directory / "idleness/status.txt").is_file() else False) and \
            log_valid(directory / "idleness/sa_region.log", "SA_REGION_PROFILE ")
    status = directory / f"policies/{task}.status"
    return status.is_file() and status.read_text().strip() == "PASS" and \
        log_valid(directory / f"policies/{task}.log", "SHARED_SA_TIMING ")


def launch_task(label: str, mode: str, model_list: Path, task: str,
                directory: Path):
    env = base_environment()
    if task == "idleness":
        env["ONNXSIM_SA_REGION_PROFILE"] = "1"
        env["ONNXSIM_SA_REGION_SIZES"] = "8,16,32,64,128"
        log_path = directory / "idleness/sa_region.log"
        status_path = directory / "idleness/status.txt"
    else:
        env["ONNXSIM_SHARED_SA_POLICY"] = task
        log_path = directory / f"policies/{task}.log"
        status_path = directory / f"policies/{task}.status"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("w")
    process = subprocess.Popen(
        simulator_command(mode, model_list), cwd=ROOT, env=env,
        stdout=log_stream, stderr=subprocess.STDOUT,
    )
    print(f"started {label}/{task}: pid={process.pid}, log={log_path}", flush=True)
    return process, log_stream, log_path, status_path


def run_group(label: str, mode: str, model_list: Path, max_total: int) -> None:
    directory = FINAL_ROOT / label
    (directory / "policies").mkdir(parents=True, exist_ok=True)
    (directory / "idleness").mkdir(parents=True, exist_ok=True)
    if (directory / "COMPLETE").is_file():
        print(f"skip complete workload: {label}", flush=True)
        return

    tasks = [task for task in ("always_on", "regate", "ours", "idleness")
             if not task_complete(directory, task)]
    running = {}
    failed = False
    while tasks or running:
        while tasks and not failed:
            # Several workload groups may execute concurrently.  Serialize the
            # count-and-launch operation so two groups cannot both observe the
            # same free slot and oversubscribe the global simulator cap.
            with LAUNCH_LOCK:
                if simulator_count() >= max_total:
                    break
                task = tasks.pop(0)
                running[task] = launch_task(
                    label, mode, model_list, task, directory)
        for task, (process, stream, log_path, status_path) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            valid = code == 0 and log_valid(
                log_path, "SA_REGION_PROFILE " if task == "idleness"
                else "SHARED_SA_TIMING ")
            status_path.write_text("PASS\n" if valid else f"FAIL:{code}\n")
            print(f"finished {label}/{task}: exit={code}, valid={valid}", flush=True)
            del running[task]
            if not valid:
                failed = True
        if failed and not running:
            raise RuntimeError(f"{label} failed; inspect {directory}")
        if tasks or running:
            time.sleep(5)

    comparison = directory / "comparison.json"
    subprocess.run([
        sys.executable, str(COMPARE),
        "--always-on-log", str(directory / "policies/always_on.log"),
        "--regate-log", str(directory / "policies/regate.log"),
        "--ours-log", str(directory / "policies/ours.log"),
        "--expected-cores", "4", "--output", str(comparison),
    ], cwd=ROOT, check=True)
    with (directory / "idleness/summary.json").open("w") as output:
        subprocess.run([
            sys.executable, str(SUMMARIZE_IDLE),
            str(directory / "idleness/sa_region.log"),
            "--expected-cores", "4",
        ], cwd=ROOT, stdout=output, check=True)
    compared = load_json(comparison)
    idle = load_json(directory / "idleness/summary.json")
    assert compared["matched_sa_work_validation_pass"] is True
    assert compared["state_partition_validation_pass"] is True
    assert compared["observed_cores"] == 4
    assert idle["validation_pass"] is True
    signature = {
        name: run["sa_events"] for name, run in
        compared["runs_at_representative_beta_0_1"].items()
    }
    assert len(set(signature.values())) == 1, signature
    (directory / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"))
    print(f"validated and completed workload: {label}", flush=True)


def preserve_inputs(workloads) -> None:
    directory = FINAL_ROOT / "experiment_inputs"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        CONFIG, ROOT / "configs/ramulator2_configs/HBM2.yaml",
        ROOT / "configs/booksim2_configs/fly_c64_m16.icnt", TRACE,
    ] + [item[2] for item in workloads]
    manifest = {}
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"missing required input: {path}")
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "remaining_suite_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-total-simulators", type=int, default=4)
    parser.add_argument("--models", nargs="*", help="optional workload labels")
    parser.add_argument(
        "--parallel-workloads", action="store_true",
        help="schedule independent workload groups concurrently under the global simulator cap")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_total_simulators < 1:
        parser.error("--max-total-simulators must be positive")
    validate_architecture()
    for required in (SIMULATOR, COMPARE, SUMMARIZE_IDLE, TRACE):
        if not required.exists():
            raise RuntimeError(f"missing required file: {required}")
    workloads = resolve_workloads(set(args.models) if args.models else None)
    preserve_inputs(workloads)
    print("validated workloads:", ", ".join(item[0] for item in workloads))
    print(f"current simulator processes: {simulator_count()}")
    print(f"global simulator-process cap: {args.max_total_simulators}")
    if args.validate_only:
        return
    if args.parallel_workloads:
        with ThreadPoolExecutor(max_workers=len(workloads)) as executor:
            futures = [executor.submit(
                run_group, *workload, max_total=args.max_total_simulators)
                for workload in workloads]
            for future in futures:
                future.result()
    else:
        for workload in workloads:
            run_group(*workload, max_total=args.max_total_simulators)
    print("all requested remaining workloads completed and validated")


if __name__ == "__main__":
    main()
