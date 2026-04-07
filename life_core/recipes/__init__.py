"""YAML recipe engine for Goose autonomous tasks."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from life_core.goose_client import GooseClient

logger = logging.getLogger("life_core.recipes")
_VAR_PATTERN = re.compile(r"\{(\w+)\}")
RECIPES_DIR = Path(__file__).parent


@dataclass
class RecipeStep:
    name: str
    prompt: str
    timeout: int = 120


@dataclass
class Recipe:
    name: str
    description: str
    steps: list[RecipeStep]


def extract_variables(recipe: Recipe) -> list[str]:
    """Extract {placeholder} variable names from all recipe step prompts."""
    vars_: set[str] = set()
    for step in recipe.steps:
        vars_.update(_VAR_PATTERN.findall(step.prompt))
    return sorted(vars_)


def load_recipe(name: str) -> Recipe:
    """Load a recipe by name from the recipes directory."""
    path = RECIPES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Recipe '{name}' not found at {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    steps = [
        RecipeStep(
            name=s["name"],
            prompt=s["prompt"],
            timeout=s.get("timeout", 120),
        )
        for s in data["steps"]
    ]
    return Recipe(name=data["name"], description=data["description"], steps=steps)


def list_recipes() -> list[Recipe]:
    """List all available recipes."""
    recipes = []
    for path in sorted(RECIPES_DIR.glob("*.yaml")):
        try:
            recipes.append(load_recipe(path.stem))
        except Exception as e:
            logger.warning("Failed to load recipe %s: %s", path.stem, e)
    return recipes


async def run_recipe(
    recipe: Recipe,
    client: "GooseClient",
    working_dir: str = ".",
    variables: dict[str, str] | None = None,
) -> list[dict]:
    """Execute a recipe: create session, run each step, collect results."""
    session = await client.create_session(working_dir=working_dir)
    results = []
    vars_ = variables or {}

    for step in recipe.steps:
        prompt = step.prompt.format(**vars_) if vars_ else step.prompt
        try:
            response = await client.prompt_sync(session.session_id, prompt)
            results.append({
                "step": step.name,
                "status": "ok",
                "response": response,
            })
        except Exception as e:
            logger.error("Recipe step '%s' failed: %s", step.name, e)
            results.append({
                "step": step.name,
                "status": "error",
                "error": str(e),
            })
    return results
