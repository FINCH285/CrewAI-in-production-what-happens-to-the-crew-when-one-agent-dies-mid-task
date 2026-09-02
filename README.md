# CrewAI Agent Crash Recovery Test

This folder contains the CrewAI examples used in the article `CrewAI in Production: What Happens to the Crew When One Agent Dies Mid-Task?`.

- `01-no-checkpoint` shows what happens when the CrewAI Hotel Researcher agent is interrupted mid-task and the application is started again without restoring state.
- `02-with-checkpoint` enables CrewAI checkpointing, terminates the process at the same point, then starts a new process and restores the latest completed-task checkpoint.

Each version is complete and runnable on its own.

## Setup

Each test folder contains an `.env.example` file. Copy it to `.env` in the folder you want to run and add your OpenAI API key. Then, from the project root, create and activate a Python virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run Without Checkpointing

```powershell
cd 01-no-checkpoint
python run_crash_test.py
python crew_recovery.py
```

The first command terminates the CrewAI process while the Hotel Researcher is inside `search_hotels()`. The second command starts the application again normally so you can see that the crew begins from the Flight Researcher.

## Run With Checkpointing

Return to the project root, then run:

```powershell
cd 02-with-checkpoint
python run_crash_test.py
```

The script terminates the CrewAI process at the same point, finds the latest checkpoint written after the completed Flight task, starts a new process, and restores that checkpoint. The Flight Researcher is not executed again, while the interrupted Hotel task starts again from the saved task boundary.
