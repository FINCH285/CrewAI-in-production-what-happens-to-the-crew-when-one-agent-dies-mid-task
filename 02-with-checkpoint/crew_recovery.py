from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Type

from crewai import Agent, CheckpointConfig, Crew, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables, including the OpenAI API key
load_dotenv()

# Store checkpoints written during the crew execution
CHECKPOINT_DIR = Path(".checkpoints")

# Control how long the hotel tool remains active
HOTEL_SLEEP_SECONDS = float(
    os.environ.get("HOTEL_SLEEP_SECONDS", "30")
)


def emit(message: str) -> None:
    """Print an event to the terminal"""

    print(message, flush=True)


class FlightArgs(BaseModel):
    # The flight search needs both ends of the route
    origin: str = Field(..., description="Origin city")
    destination: str = Field(
        ...,
        description="Destination city",
    )


class HotelArgs(BaseModel):
    # The hotel search only needs the destination city
    city: str = Field(..., description="Destination city")


class PlanArgs(BaseModel):
    # The planner receives the completed travel research
    summary: str = Field(
        ...,
        description="Flight and hotel results",
    )


class SearchFlightsTool(BaseTool):
    name: str = "Search_flights"
    description: str = "Search flights between two cities."
    args_schema: Type[BaseModel] = FlightArgs
    result_as_answer: bool = True

    def _run(self, origin: str, destination: str) -> str:
        # Record when the tool begins
        emit(
            f">>> TOOL: search_flights("
            f"'{origin}', '{destination}')"
        )

        # Use fixed data so an external API cannot affect the test
        result = (
            f"Found 3 flights from {origin} to "
            f"{destination}, cheapest $412."
        )

        emit(f">>> TOOL COMPLETE: search_flights -> {result}")
        return result


class SearchHotelsTool(BaseTool):
    name: str = "Search_hotels"
    description: str = "Search hotels in a destination city."
    args_schema: Type[BaseModel] = HotelArgs
    result_as_answer: bool = True

    def _run(self, city: str) -> str:
        # Record that execution has entered the hotel tool
        emit(f">>> TOOL: search_hotels('{city}')")

        # Keep the tool active long enough to terminate
        # the CrewAI process before the call completes
        emit(
            f">>> TOOL: search_hotels sleeping "
            f"{HOTEL_SLEEP_SECONDS:g}s"
        )
        time.sleep(HOTEL_SLEEP_SECONDS)

        result = (
            f"Best hotel in {city}: Grand Plaza, "
            "$140/night, 4.5 stars."
        )

        emit(f">>> TOOL COMPLETE: search_hotels -> {result}")
        return result


class FinalizePlanTool(BaseTool):
    name: str = "Finalize_trip_plan"
    description: str = "Finalize a trip using flight and hotel results."
    args_schema: Type[BaseModel] = PlanArgs
    result_as_answer: bool = True

    def _run(self, summary: str) -> str:
        # Reaching this point means the earlier tasks completed
        emit(">>> TOOL: finalize_plan called")

        result = f"Trip plan finalized: {summary}"

        emit(f">>> TOOL COMPLETE: finalize_plan -> {result}")
        return result


flight_researcher = Agent(
    role="Flight Researcher",
    goal="Call the flight search tool exactly once.",
    backstory="You research flights and always use your tool.",
    llm="gpt-4o-mini",
    tools=[SearchFlightsTool()],
    allow_delegation=False,
    verbose=False,
)

hotel_researcher = Agent(
    role="Hotel Researcher",
    goal="Call the hotel search tool exactly once.",
    backstory="You research hotels and always use your tool.",
    llm="gpt-4o-mini",
    tools=[SearchHotelsTool()],
    allow_delegation=False,
    verbose=False,
)

trip_planner = Agent(
    role="Trip Planner",
    goal="Finalize the trip using the completed research.",
    backstory="You produce the final trip plan and always use your tool.",
    llm="gpt-4o-mini",
    tools=[FinalizePlanTool()],
    allow_delegation=False,
    verbose=False,
)


flight_task = Task(
    description=(
        "Find flights from {origin} to {destination}. "
        "You must call your only tool exactly once."
    ),
    expected_output="The flight search result.",
    agent=flight_researcher,
)

hotel_task = Task(
    description=(
        "Find a hotel in {destination}. "
        "You must call your only tool exactly once."
    ),
    expected_output="The hotel search result.",
    agent=hotel_researcher,
    context=[flight_task],
)

planner_task = Task(
    description=(
        "Finalize the trip using the completed flight "
        "and hotel results. You must call your only "
        "tool exactly once."
    ),
    expected_output="The finalized trip plan.",
    agent=trip_planner,
    context=[flight_task, hotel_task],
)


crew = Crew(
    agents=[
        flight_researcher,
        hotel_researcher,
        trip_planner,
    ],
    tasks=[
        flight_task,
        hotel_task,
        planner_task,
    ],
    process=Process.sequential,
    cache=False,
    verbose=False,
    # Save the crew state whenever a task completes
    checkpoint=CheckpointConfig(
        location=str(CHECKPOINT_DIR),
        on_events=["task_completed"],
        max_checkpoints=20,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Accept a checkpoint path when restoring the crew
    parser.add_argument("--restore", type=str, default=None)
    args = parser.parse_args()

    inputs = {
        "origin": "Boston",
        "destination": "Denver",
    }

    if args.restore:
        emit(f">>> RESTORE: {args.restore}")

        # Configure CrewAI to restore the saved run
        restore = CheckpointConfig(
            location=str(CHECKPOINT_DIR),
            restore_from=args.restore,
            on_events=["task_completed"],
            max_checkpoints=20,
        )

        result = crew.kickoff(
            inputs=inputs,
            from_checkpoint=restore,
        )
    else:
        emit(">>> KICKOFF: checkpoint=enabled")
        result = crew.kickoff(inputs=inputs)

    emit(">>> CREW COMPLETE")
    print(result.raw, flush=True)


if __name__ == "__main__":
    main()
