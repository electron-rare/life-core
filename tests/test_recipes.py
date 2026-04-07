"""Tests for YAML recipe engine."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from life_core.recipes import Recipe, RecipeStep, load_recipe, list_recipes, run_recipe, extract_variables


def test_recipe_step_dataclass():
    step = RecipeStep(name="check", prompt="run drc", timeout=60)
    assert step.name == "check"
    assert step.prompt == "run drc"
    assert step.timeout == 60


def test_recipe_dataclass():
    r = Recipe(
        name="test-recipe",
        description="A test",
        steps=[RecipeStep(name="s1", prompt="hello", timeout=30)],
    )
    assert r.name == "test-recipe"
    assert len(r.steps) == 1


def test_load_recipe_review_kicad():
    recipe = load_recipe("review-kicad")
    assert recipe.name == "review-kicad"
    assert len(recipe.steps) >= 2
    assert recipe.description


def test_load_recipe_debug_infra():
    recipe = load_recipe("debug-infra")
    assert recipe.name == "debug-infra"
    assert len(recipe.steps) >= 2


def test_load_recipe_index_repos():
    recipe = load_recipe("index-repos")
    assert recipe.name == "index-repos"
    assert len(recipe.steps) >= 1


def test_load_recipe_eval_sft():
    recipe = load_recipe("eval-sft")
    assert recipe.name == "eval-sft"
    assert len(recipe.steps) >= 2


def test_load_recipe_not_found():
    with pytest.raises(FileNotFoundError):
        load_recipe("nonexistent-recipe")


def test_list_recipes():
    recipes = list_recipes()
    names = [r.name for r in recipes]
    assert "review-kicad" in names
    assert "debug-infra" in names
    assert "index-repos" in names
    assert "eval-sft" in names


def test_extract_variables_from_recipe():
    recipe = load_recipe("review-kicad")
    vars_ = extract_variables(recipe)
    assert "project_path" in vars_


def test_extract_variables_no_vars():
    recipe = load_recipe("debug-infra")
    vars_ = extract_variables(recipe)
    assert len(vars_) == 0


@pytest.mark.asyncio
async def test_run_recipe():
    recipe = load_recipe("debug-infra")
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(
        return_value=type("S", (), {"session_id": "test-session"})()
    )
    mock_client.prompt_sync = AsyncMock(return_value="All clear")

    results = await run_recipe(recipe, mock_client, working_dir="/tmp")
    assert len(results) == len(recipe.steps)
    assert all(r["status"] == "ok" for r in results)
    mock_client.create_session.assert_called_once_with(working_dir="/tmp")
