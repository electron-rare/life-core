"""FastAPI router for Goose agent integration."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from life_core.goose_client import GooseClient
from life_core.recipes import list_recipes, load_recipe, run_recipe

logger = logging.getLogger("life_core.goose_api")
router = APIRouter(prefix="/goose", tags=["goose"])

_client: GooseClient | None = None


def _get_client() -> GooseClient:
    global _client
    if _client is None:
        _client = GooseClient()
    return _client


class SessionCreateRequest(BaseModel):
    working_dir: str = "."


class RecipeRunRequest(BaseModel):
    working_dir: str = "."
    variables: dict[str, str] | None = None


@router.get("/health")
async def goose_health():
    """Check goosed agent health."""
    try:
        result = await _get_client().health()
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"goosed unreachable: {e}")


@router.get("/recipes")
async def goose_recipes():
    """List available recipes."""
    recipes = list_recipes()
    return {
        "recipes": [
            {"name": r.name, "description": r.description, "steps": len(r.steps)}
            for r in recipes
        ]
    }


@router.post("/sessions")
async def goose_session_create(req: SessionCreateRequest):
    """Create a new goosed session."""
    try:
        session = await _get_client().create_session(working_dir=req.working_dir)
        return {"session_id": session.session_id, "working_dir": session.working_dir}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to create session: {e}")


@router.post("/recipes/{recipe_name}/run")
async def goose_recipe_run(recipe_name: str, req: RecipeRunRequest):
    """Run a recipe by name."""
    try:
        recipe = load_recipe(recipe_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Recipe '{recipe_name}' not found")
    try:
        results = await run_recipe(
            recipe, _get_client(),
            working_dir=req.working_dir,
            variables=req.variables,
        )
        return {"recipe": recipe_name, "results": results}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Recipe execution failed: {e}")
