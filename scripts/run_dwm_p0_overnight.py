#!/usr/bin/env python3
"""Run frozen D0-C audit and P0 suite, write a report, then shut down."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_command(command, log_handle):
    log_handle.write("$ " + " ".join(command) + "\n")
    log_handle.flush()
    subprocess.run(
        command,
        check=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(
    path,
    status,
    audit,
    gate,
):
    lines = [
        "# DWM P0 Overnight Result",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Decision",
        "",
    ]
    if audit is not None and not audit.get("overall", False):
        lines.extend([
            "**NO-GO at D0-C audit.** The probe dataset did not pass its "
            "frozen data gates, so P0 training was not started.",
        ])
    elif gate is None:
        lines.extend([
            "**No P0 gate was produced.** See the error and logs in this "
            "directory.",
        ])
    elif gate.get("overall", False):
        lines.extend([
            "**GO for P0.** The frozen passive probe ranking gates passed "
            "on the unseen object-configuration test split.",
        ])
    else:
        lines.extend([
            "**NO-GO for P0.** At least one frozen promotion gate failed.",
        ])
    lines.extend(["", "## Status", "", "```json"])
    lines.append(json.dumps(status, indent=2, sort_keys=True))
    lines.extend(["```", ""])
    if audit is not None:
        lines.extend([
            "## D0-C Audit",
            "",
            "```json",
            json.dumps(audit, indent=2, sort_keys=True),
            "```",
            "",
        ])
    if gate is not None:
        lines.extend([
            "## P0 Gate",
            "",
            "```json",
            json.dumps(gate, indent=2, sort_keys=True),
            "```",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_archive(results_dir):
    archive = results_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(results_dir, arcname=results_dir.name)
    return archive


def shutdown():
    attempts = []
    for command in (
        ["bash", "/usr/bin/shutdown"],
        ["shutdown", "-h", "now"],
        ["poweroff"],
        ["halt", "-p"],
    ):
        try:
            result = subprocess.run(command, check=False)
            attempts.append({
                "command": command,
                "returncode": result.returncode,
            })
            if result.returncode == 0:
                break
        except OSError as exc:
            attempts.append({
                "command": command,
                "error": repr(exc),
            })
        time.sleep(2)
    return attempts


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    audit_path = args.results_dir / "d0c_audit.json"
    gate_path = args.results_dir / "p0_gate.json"
    report_path = args.results_dir / "dwm-p0-report-2026-09-22.md"
    status_path = args.results_dir / "status.json"
    log_path = args.results_dir / "overnight.log"
    status = {
        "started_at": utc_now(),
        "finished_at": None,
        "dataset_dir": str(args.dataset_dir),
        "results_dir": str(args.results_dir),
        "phase": "start",
        "error": None,
        "shutdown": [],
    }
    audit = None
    gate = None
    write_json(status_path, status)
    with log_path.open("a", encoding="utf-8") as log:
        try:
            status["phase"] = "d0c_audit"
            write_json(status_path, status)
            repeat_a = args.results_dir / "repeat_a"
            repeat_b = args.results_dir / "repeat_b"
            for repeat_dir in (repeat_a, repeat_b):
                run_command([
                    sys.executable,
                    str(root / "scripts" / "generate_dwm_probe_dataset.py"),
                    "--output-dir",
                    str(repeat_dir),
                    "--resets-per-object",
                    "1",
                    "--object-limit",
                    "1",
                    "--workers",
                    "1",
                ], log)
            run_command([
                sys.executable,
                str(root / "scripts" / "audit_dwm_probe_dataset.py"),
                "--dataset-dir",
                str(args.dataset_dir),
                "--repeat-a",
                str(repeat_a),
                "--repeat-b",
                str(repeat_b),
                "--output",
                str(audit_path),
            ], log)
            audit = load_json(audit_path)
            if not audit.get("overall", False):
                status["phase"] = "d0c_audit_no_go"
                write_report(report_path, status, audit, None)
                return

            status["phase"] = "p0_suite"
            write_json(status_path, status)
            run_command([
                sys.executable,
                str(root / "scripts" / "run_dwm_p0_suite.py"),
                "--dataset-dir",
                str(args.dataset_dir),
                "--output-dir",
                str(args.results_dir / "arms"),
                "--seeds",
                "11",
                "23",
                "37",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--allow-test",
                "--bootstrap",
                str(args.bootstrap),
            ], log)
            source_gate = args.results_dir / "arms" / "p0_gate.json"
            gate = load_json(source_gate)
            write_json(gate_path, gate)
            status["phase"] = (
                "p0_go" if gate.get("overall", False) else "p0_no_go"
            )
        except Exception as exc:
            status["phase"] = "error"
            status["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            log.write(status["error"]["traceback"] + "\n")
            log.flush()
        finally:
            status["finished_at"] = utc_now()
            write_report(report_path, status, audit, gate)
            write_json(status_path, status)
            archive = make_archive(args.results_dir)
            (args.results_dir / "READY_FOR_SHUTDOWN").write_text(
                utc_now() + "\n",
                encoding="utf-8",
            )
            status["archive"] = str(archive)
            write_json(status_path, status)
            subprocess.run(["sync"], check=False)
            status["shutdown"] = shutdown()
            write_json(status_path, status)


if __name__ == "__main__":
    main()
