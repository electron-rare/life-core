"""Project management data models."""
from __future__ import annotations
from pydantic import BaseModel, Field


class Gate(BaseModel):
    status: str = "pending"
    date: str | None = None


class TaskCreate(BaseModel):
    name: str
    gate: str
    assignees: list[str] = Field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    depends_on: list[str] = Field(default_factory=list)


class Task(TaskCreate):
    id: str
    status: str = "todo"
    progress: int = 0


class ProjectCreate(BaseModel):
    name: str
    client: str = ""
    repo: str = ""
    hardware: dict = Field(default_factory=lambda: {"pcb_dir": "hardware/pcb", "simulation_dir": "hardware/simulation", "bom_dir": "hardware/bom"})
    firmware: dict = Field(default_factory=lambda: {"framework": "platformio", "src_dir": "firmware/src"})
    agents: list[str] = Field(default_factory=lambda: ["forge", "qa-agent"])


class ProjectUpdate(BaseModel):
    client: str | None = None
    repo: str | None = None
    hardware: dict | None = None
    firmware: dict | None = None
    agents: list[str] | None = None
    gates: dict[str, Gate] | None = None


class TeamMember(BaseModel):
    id: str
    name: str
    type: str
    avatar_url: str | None = None
