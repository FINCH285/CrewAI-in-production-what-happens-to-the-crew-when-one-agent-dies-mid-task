from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Checkpoints created by the crew
CHECKPOINT_DIR = Path(".checkpoints")

# Configure the child process as the interrupted test run
env = os.environ.copy()
env["HOTEL_SLEEP_SECONDS"] = "30"
env["CREWAI_TRACING_ENABLED"] = "false"

# Start the CrewAI application as a separate process
proc = subprocess.Popen(
    [sys.executable, "crew_recovery.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
)

assert proc.stdout is not None

# Print the child process output as it arrives
for line in proc.stdout:
    print(line, end="")

    # Terminate the process after the Hotel Researcher
    # enters its tool but before the tool returns
    if "search_hotels sleeping 30s" in line:
        print(
            f"Killing CrewAI process pid={proc.pid}",
            flush=True,
        )
        proc.kill()
        break

proc.wait()

# Find the latest checkpoint written before the process died
checkpoints = sorted(
    CHECKPOINT_DIR.rglob("*.json"),
    key=lambda path: path.stat().st_mtime,
)

if not checkpoints:
    raise RuntimeError(
        "No checkpoint was written before the crash"
    )

latest = checkpoints[-1]

print(f"Restoring checkpoint: {latest}", flush=True)

# Configure the restored run
restore_env = os.environ.copy()
restore_env["HOTEL_SLEEP_SECONDS"] = "1"
restore_env["CREWAI_TRACING_ENABLED"] = "false"

# Start a new process with the latest checkpoint
restored = subprocess.run(
    [
        sys.executable,
        "crew_recovery.py",
        "--restore",
        str(latest),
    ],
    env=restore_env,
)

if restored.returncode != 0:
    raise RuntimeError(
        f"Restore failed with exit code {restored.returncode}"
    )
