"""Registry for managing exercise definitions."""

from typing import Dict, Optional
from pathlib import Path

from .exercise import Exercise
from .exercise_config import ExerciseConfig


class ExerciseRegistry:
    """Registry for exercise definitions."""

    _instance: Optional["ExerciseRegistry"] = None
    _exercises: Dict[str, Exercise] = {}

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(
        self,
        exercise: Exercise,
        overwrite: bool = False,
    ) -> None:
        """
        Register an exercise.

        Args:
            exercise: Exercise instance
            overwrite: If True, overwrite existing exercise with same name
        """
        name = exercise.config.name

        if name in self._exercises and not overwrite:
            raise ValueError(f"Exercise '{name}' already registered. Use overwrite=True to replace.")

        self._exercises[name] = exercise

    def get(self, name: str) -> Optional[Exercise]:
        """
        Get an exercise by name.

        Args:
            name: Exercise name

        Returns:
            Exercise instance or None if not found
        """
        return self._exercises.get(name)

    def list_exercises(self) -> list[str]:
        """
        List all registered exercise names.

        Returns:
            List of exercise names
        """
        return sorted(self._exercises.keys())

    def unregister(self, name: str) -> None:
        """
        Unregister an exercise.

        Args:
            name: Exercise name
        """
        if name in self._exercises:
            del self._exercises[name]

    def clear(self) -> None:
        """Clear all registered exercises."""
        self._exercises.clear()

    def load_exercise(
        self,
        name: str,
        data_root: str = "data",
        config: Optional[ExerciseConfig] = None,
    ) -> Exercise:
        """
        Load an exercise from disk.

        Args:
            name: Exercise name
            data_root: Root directory for data
            config: Optional config (will be loaded from disk if not provided)

        Returns:
            Exercise instance
        """
        if config is None:
            exercise_dir = Path(data_root) / "processed" / name
            config_path = exercise_dir / "config.yaml"
            if config_path.exists():
                config = ExerciseConfig.load(str(config_path))
            else:
                # Create default config
                config = ExerciseConfig(name=name)

        exercise = Exercise(config, data_root=data_root)
        exercise.load()

        self.register(exercise, overwrite=True)

        return exercise

