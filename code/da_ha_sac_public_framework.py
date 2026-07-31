"""
Public DA-HA-SAC framework
==========================

A compact research-code skeleton for coupling a dual-actuator reinforcement-
learning controller with Ansys Fluent.

The exact case geometry, sensor locations, baseline data, reward coefficients,
solver report definitions, training hyperparameters, and trained weights used
in the associated manuscript are intentionally not included before publication.

This file presents the software architecture only. A private JSON configuration
and case-specific implementations of the marked hooks are required to run it.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import ansys.fluent.core as pyfluent
except ImportError:  # Allows the public file to be inspected without Fluent.
    pyfluent = None

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
except ImportError:  # Training requires Stable-Baselines3.
    SAC = None
    DummyVecEnv = SubprocVecEnv = VecMonitor = None


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration loaded from a private JSON file excluded from Git."""

    case_file: Path
    data_file: Path
    workdir: Path
    actuator_boundaries: tuple[str, str]
    sensor_surfaces: tuple[str, ...]
    processor_count: int
    cfd_steps_per_action: int
    max_solver_iterations: int
    episode_length: int
    history_length: int
    actuation_limit: float
    stage_boundaries: tuple[int, int]
    reward_weights: dict[str, tuple[float, ...]]
    sac_parameters: dict[str, Any]
    parallel_workdirs: tuple[str, ...] = ()
    quiet_solver: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Private configuration not found: {source}")

        data = json.loads(source.read_text(encoding="utf-8"))
        required = {
            "case_file",
            "data_file",
            "workdir",
            "actuator_boundaries",
            "sensor_surfaces",
            "processor_count",
            "cfd_steps_per_action",
            "max_solver_iterations",
            "episode_length",
            "history_length",
            "actuation_limit",
            "stage_boundaries",
            "reward_weights",
            "sac_parameters",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"Missing configuration fields: {missing}")

        return cls(
            case_file=Path(data["case_file"]),
            data_file=Path(data["data_file"]),
            workdir=Path(data["workdir"]),
            actuator_boundaries=tuple(data["actuator_boundaries"]),
            sensor_surfaces=tuple(data["sensor_surfaces"]),
            processor_count=int(data["processor_count"]),
            cfd_steps_per_action=int(data["cfd_steps_per_action"]),
            max_solver_iterations=int(data["max_solver_iterations"]),
            episode_length=int(data["episode_length"]),
            history_length=int(data["history_length"]),
            actuation_limit=float(data["actuation_limit"]),
            stage_boundaries=tuple(int(v) for v in data["stage_boundaries"]),
            reward_weights={
                key: tuple(float(v) for v in values)
                for key, values in data["reward_weights"].items()
            },
            sac_parameters=dict(data["sac_parameters"]),
            parallel_workdirs=tuple(data.get("parallel_workdirs", ())),
            quiet_solver=bool(data.get("quiet_solver", True)),
        )


class FluentAdapter:
    """
    Minimal Fluent interface.

    Case-specific report definitions and field-reduction expressions are kept
    private. Replace the marked methods with implementations matching the case.
    """

    def __init__(self, config: ProjectConfig):
        if pyfluent is None:
            raise ImportError("ansys-fluent-core is required to run this framework.")

        self.config = config
        self.session = pyfluent.launch_fluent(
            mode="solver",
            dimension=2,
            ui_mode="no_gui",
            processor_count=config.processor_count,
        )
        self.session.settings.file.read_case(file_name=str(config.case_file))

    @contextmanager
    def quiet_console(self):
        if not self.config.quiet_solver:
            yield
            return

        with open(os.devnull, "w", encoding="utf-8") as null_stream:
            with redirect_stdout(null_stream), redirect_stderr(null_stream):
                yield

    def reset(self) -> None:
        self.session.settings.file.read_data(file_name=str(self.config.data_file))

    def apply_actuation(self, commands: np.ndarray) -> None:
        """Apply two signed actuator commands to the configured boundaries."""
        if commands.shape != (2,):
            raise ValueError(f"Expected two actuator commands, received {commands.shape}")

        for boundary, value in zip(self.config.actuator_boundaries, commands):
            expression = f"{float(value):.6g}[m s^-1]"
            self.session.setup.boundary_conditions.velocity_inlet[
                boundary
            ].momentum.velocity.value = expression

    def advance(self) -> None:
        with self.quiet_console():
            self.session.settings.solution.run_calculation.dual_time_iterate(
                time_step_count=self.config.cfd_steps_per_action,
                max_iter_per_step=self.config.max_solver_iterations,
            )

    def read_sensor_vector(self) -> np.ndarray:
        """
        Return the normalised pressure-observation vector.

        The exact sensor layout, normalisation constants, and Fluent field-data
        calls are intentionally omitted from the public version.
        """
        raise NotImplementedError(
            "Implement read_sensor_vector() using the private sensor definition."
        )

    def read_control_metrics(self) -> dict[str, float]:
        """
        Return normalised control metrics.

        Expected keys:
            separation : separation or reverse-flow indicator
            loss       : aerodynamic-loss indicator
            efficiency : actuation-efficiency indicator

        The exact equations, baselines, delays, and report-file parsing are
        intentionally omitted from the public version.
        """
        raise NotImplementedError(
            "Implement read_control_metrics() using private Fluent reports."
        )

    def close(self) -> None:
        try:
            self.session.exit()
        except Exception:
            pass


class StagedReward:
    """Stage-dependent multi-objective reward using private coefficients."""

    COMPONENTS = ("separation", "loss", "efficiency", "smoothness")

    def __init__(self, config: ProjectConfig):
        self.boundary_1, self.boundary_2 = config.stage_boundaries
        self.weights = config.reward_weights

    def _stage_name(self, step: int) -> str:
        if step < self.boundary_1:
            return "transient"
        if step < self.boundary_2:
            return "transition"
        return "maintenance"

    def evaluate(
        self,
        metrics: dict[str, float],
        action: np.ndarray,
        previous_action: np.ndarray,
        step: int,
    ) -> tuple[float, dict[str, float]]:
        stage = self._stage_name(step)
        weights = self.weights[stage]

        if len(weights) != len(self.COMPONENTS):
            raise ValueError(
                f"Reward stage '{stage}' must provide "
                f"{len(self.COMPONENTS)} coefficients."
            )

        smoothness = -float(np.mean(np.square(action - previous_action)))
        values = np.array(
            [
                float(metrics["separation"]),
                float(metrics["loss"]),
                float(metrics["efficiency"]),
                smoothness,
            ],
            dtype=float,
        )
        reward = float(np.dot(np.asarray(weights, dtype=float), values))

        details = {
            "stage": stage,
            **{name: float(value) for name, value in zip(self.COMPONENTS, values)},
        }
        return reward, details


class CompressorControlEnv(gym.Env):
    """Gymnasium environment for dual-actuator Fluent flow control."""

    metadata = {"render_modes": []}

    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.config = config
        self.adapter = FluentAdapter(config)
        self.reward_model = StagedReward(config)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Observation size is determined after the first sensor read.
        self._sensor_size: Optional[int] = None
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )

        self.history: deque[np.ndarray] = deque(maxlen=config.history_length)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.step_index = 0

    def _physical_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        return np.clip(action, -1.0, 1.0) * self.config.actuation_limit

    def _instantaneous_observation(
        self,
        sensor_vector: np.ndarray,
        metrics: dict[str, float],
    ) -> np.ndarray:
        control_state = np.array(
            [
                self.previous_action[0] / self.config.actuation_limit,
                self.previous_action[1] / self.config.actuation_limit,
                metrics["loss"],
                self.step_index / max(1, self.config.episode_length),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [sensor_vector.astype(np.float32), control_state],
            axis=0,
        )

    def _stack_observation(self, current: np.ndarray) -> np.ndarray:
        if not self.history:
            for _ in range(self.config.history_length):
                self.history.append(current.copy())
        else:
            self.history.appendleft(current.copy())
        return np.concatenate(list(self.history), axis=0).astype(np.float32)

    def _update_observation_space(self, stacked: np.ndarray) -> None:
        if self.observation_space.shape != stacked.shape:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=stacked.shape,
                dtype=np.float32,
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.adapter.reset()
        self.history.clear()
        self.previous_action.fill(0.0)
        self.step_index = 0

        sensors = self.adapter.read_sensor_vector()
        metrics = self.adapter.read_control_metrics()
        current = self._instantaneous_observation(sensors, metrics)
        observation = self._stack_observation(current)
        self._update_observation_space(observation)
        return observation, {}

    def step(self, action):
        physical_action = self._physical_action(action)
        self.adapter.apply_actuation(physical_action)
        self.adapter.advance()

        sensors = self.adapter.read_sensor_vector()
        metrics = self.adapter.read_control_metrics()
        reward, reward_info = self.reward_model.evaluate(
            metrics=metrics,
            action=physical_action,
            previous_action=self.previous_action,
            step=self.step_index,
        )

        self.previous_action = physical_action.astype(np.float32)
        self.step_index += 1

        current = self._instantaneous_observation(sensors, metrics)
        observation = self._stack_observation(current)

        terminated = False
        truncated = self.step_index >= self.config.episode_length
        info = {
            "step": self.step_index,
            "actuator_1": float(physical_action[0]),
            "actuator_2": float(physical_action[1]),
            **metrics,
            **reward_info,
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        self.adapter.close()


class ExperimentManager:
    """Construct environments and train or evaluate a SAC policy."""

    def __init__(self, config: ProjectConfig, output_dir: str | Path = "artifacts"):
        if SAC is None:
            raise ImportError("stable-baselines3 is required for training.")

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_env(self, workdir: Optional[str] = None):
        config = self.config
        if workdir is not None:
            config = ProjectConfig(
                **{
                    **config.__dict__,
                    "workdir": Path(workdir),
                }
            )
        return CompressorControlEnv(config)

    def build_vector_env(self):
        workdirs = self.config.parallel_workdirs
        if not workdirs:
            return VecMonitor(DummyVecEnv([lambda: self._make_env()]))

        factories = [
            (lambda directory=directory: self._make_env(directory))
            for directory in workdirs
        ]
        return VecMonitor(SubprocVecEnv(factories, start_method="spawn"))

    def train(self, total_timesteps: int) -> Path:
        env = self.build_vector_env()
        params = dict(self.config.sac_parameters)

        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=str(self.output_dir / "tensorboard"),
            **params,
        )
        model.learn(total_timesteps=int(total_timesteps))

        model_path = self.output_dir / "da_ha_sac_model"
        model.save(str(model_path))
        env.close()
        return model_path.with_suffix(".zip")

    def evaluate(self, model_path: str | Path) -> float:
        env = DummyVecEnv([lambda: self._make_env()])
        model = SAC.load(str(model_path), env=env)

        observation = env.reset()
        done = np.array([False])
        episode_return = 0.0

        while not bool(done[0]):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, _ = env.step(action)
            episode_return += float(reward[0])

        env.close()
        return episode_return


def main() -> None:
    """
    Example entry point.

    Keep private_config.json outside version control. The file should contain
    case paths, Fluent names, private reward settings, and SAC hyperparameters.
    """
    config = ProjectConfig.from_json("private_config.json")
    manager = ExperimentManager(config)
    model_path = manager.train(total_timesteps=config.sac_parameters.pop("total_timesteps"))
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
