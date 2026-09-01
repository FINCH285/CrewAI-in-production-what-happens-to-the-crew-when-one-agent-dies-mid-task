from __future__ import annotations

import os
import subprocess
import sys

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
