"""Exercise framework for configurable exercise definitions."""

from .exercise import Exercise
from .exercise_registry import ExerciseRegistry
from .exercise_config import ExerciseConfig

__all__ = ["Exercise", "ExerciseRegistry", "ExerciseConfig"]

