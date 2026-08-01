"""
Public DA-HA-SAC framework
==========================

This file preserves the principal Fluent–reinforcement-learning workflow used
in the project while omitting unpublished case-specific implementation details.

The following items are intentionally not disclosed in this public version:
- exact sensor layout and normalisation;
- reverse-flow-region definition;
- aerodynamic-loss and actuation-efficiency equations;
- reward coefficients and stage-switching parameters;
- solver report definitions;
- trained-model settings and case-specific baselines.

The public code is therefore an architectural reference rather than a complete
reproduction package.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import csv

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import ansys.fluent.core as pyfluent
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Case-independent CFD environment settings."""

    case_id: int
    cas_path: str
    data_path: str
    workdir: str

    actuator_names: tuple[str, str]
    probe_names: tuple[str, ...]

    processor_count: int
    cfd_steps_per_action: int
    max_solver_iterations: int
    episode_length: int
    history_length: int

    observation_dim: int
    action_limit: float

    parallel_workdirs: tuple[str, ...] = ()
    show_gui: str = "no_gui"


@dataclass
class RLConfig:
    """General SAC settings.

    Exact values used in the manuscript are intentionally not included.
    """

    action_dim: int = 2
    learning_rate: float = 0.0
    batch_size: int = 0
    buffer_size: int = 0
    learning_starts: int = 0
    gamma: float = 0.0
    tau: float = 0.0
    train_freq: int = 0
    gradient_steps: int = 0
    ent_coef: str = "auto"
    n_envs: int = 1

    policy_kwargs: dict = field(default_factory=dict)
    tensorboard_log_dir: str = "tb_logs"


# ---------------------------------------------------------------------------
# Data recording
# ---------------------------------------------------------------------------

class HistoryRecorder:
    """Write the principal control variables to a CSV file."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = self.output_path.exists()

    def append(self, row: dict) -> None:
        with self.output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Fluent + Gymnasium environment
# ---------------------------------------------------------------------------

class CompressorEnv(gym.Env):
    """Closed-loop compressor-cascade control environment.

    The main control loop is retained:

        action
          -> actuator boundary update
          -> Fluent time advancement
          -> flow observation
          -> aerodynamic metrics
          -> reward
          -> next history-augmented state

    Sensitive metric and reward implementations are represented by private
    hooks and are not included in this public version.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: EnvConfig,
        rl_config: RLConfig,
        mode: str = "train",
        history_dir: str = "history",
    ):
        super().__init__()

        self.env_config = env_config
        self.rl_config = rl_config
        self.mode = mode

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(rl_config.action_dim,),
            dtype=np.float32,
        )

        stacked_dim = env_config.observation_dim * env_config.history_length
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(stacked_dim,),
            dtype=np.float32,
        )

        self.observation_history: deque[np.ndarray] = deque(
            maxlen=env_config.history_length
        )
        self.previous_action = np.zeros(rl_config.action_dim, dtype=np.float32)
        self.step_index = 0

        history_path = Path(history_dir) / f"case{env_config.case_id}_{mode}.csv"
        self.history = HistoryRecorder(history_path)

        self.session = self._launch_fluent()

    # ------------------------------------------------------------------
    # Fluent interface
    # ------------------------------------------------------------------

    def _launch_fluent(self):
        """Launch Fluent and load the case file."""
        session = pyfluent.launch_fluent(
            mode="solver",
            dimension=2,
            ui_mode=self.env_config.show_gui,
            processor_count=self.env_config.processor_count,
        )
        session.settings.file.read_case(file_name=self.env_config.cas_path)
        return session

    def _reset_flow_solution(self) -> None:
        """Reload the initial flow field at the beginning of an episode."""
        self.session.settings.file.read_data(
            file_name=self.env_config.data_path
        )

    def _map_action_to_physical_command(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        """Map the normalised SAC action to signed actuator commands."""
        action = np.asarray(action, dtype=np.float32).reshape(
            self.rl_config.action_dim
        )
        action = np.clip(action, -1.0, 1.0)
        return action * self.env_config.action_limit

    def _apply_actuator_commands(
        self,
        commands: np.ndarray,
    ) -> None:
        """Apply independent blowing/suction commands to both actuators."""
        if commands.shape != (2,):
            raise ValueError("The public framework assumes two actuators.")

        for boundary_name, command in zip(
            self.env_config.actuator_names,
            commands,
        ):
            expression = f"{float(command):.6g}[m s^-1]"
            boundary = self.session.setup.boundary_conditions.velocity_inlet[
                boundary_name
            ]
            boundary.momentum.velocity.value = expression

    def _advance_cfd(self) -> None:
        """Advance the unsteady CFD solution over one control interval."""
        self.session.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=self.env_config.cfd_steps_per_action,
            max_iter_per_step=self.env_config.max_solver_iterations,
        )

    # ------------------------------------------------------------------
    # Observation and reward hooks
    # ------------------------------------------------------------------

    def _read_pressure_observation(self) -> np.ndarray:
        """Return the normalised pressure-probe vector.

        The exact probe layout, field-data calls, and normalisation constants
        are omitted because they are part of the unpublished implementation.
        """
        raise NotImplementedError(
            "Pressure-observation extraction is not included in the public code."
        )

    def _read_aerodynamic_metrics(self) -> dict[str, float]:
        """Return the principal aerodynamic metrics.

        A private implementation evaluates quantities such as:
        - separation or reverse-flow level;
        - downstream total-pressure loss;
        - actuation efficiency.

        Exact report definitions, delay handling, and baseline values are
        intentionally omitted.
        """
        raise NotImplementedError(
            "Aerodynamic-metric evaluation is not included in the public code."
        )

    def _compute_stage_dependent_reward(
        self,
        metrics: dict[str, float],
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate the stage-dependent multi-objective reward.

        The private implementation changes the emphasis from rapid separation
        suppression during the transient stage to loss reduction and efficient
        flow maintenance during the later stage. Exact equations, weights,
        thresholds, delay compensation, and penalty terms are not disclosed.
        """
        raise NotImplementedError(
            "The manuscript reward formulation is not included before publication."
        )

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_instantaneous_observation(
        self,
        pressure_vector: np.ndarray,
        metrics: dict[str, float],
    ) -> np.ndarray:
        """Combine flow measurements, previous action, and control progress."""
        progress = self.step_index / max(1, self.env_config.episode_length)

        control_features = np.concatenate(
            [
                self.previous_action.astype(np.float32),
                np.array(
                    [
                        float(metrics.get("loss", 0.0)),
                        float(progress),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        observation = np.concatenate(
            [
                pressure_vector.astype(np.float32),
                control_features,
            ]
        )

        if observation.size != self.env_config.observation_dim:
            raise ValueError(
                "Observation size does not match EnvConfig.observation_dim."
            )

        return observation

    def _stack_observation(
        self,
        current_observation: np.ndarray,
        initialise: bool = False,
    ) -> np.ndarray:
        """Construct the history-augmented state used by the SAC agent."""
        if initialise:
            self.observation_history.clear()
            for _ in range(self.env_config.history_length):
                self.observation_history.append(
                    current_observation.copy()
                )
        else:
            self.observation_history.appendleft(
                current_observation.copy()
            )

        return np.concatenate(
            list(self.observation_history),
            axis=0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self._reset_flow_solution()
        self.previous_action.fill(0.0)
        self.step_index = 0

        pressure_vector = self._read_pressure_observation()
        metrics = self._read_aerodynamic_metrics()

        current_observation = self._build_instantaneous_observation(
            pressure_vector,
            metrics,
        )
        observation = self._stack_observation(
            current_observation,
            initialise=True,
        )
        return observation, {}

    def step(self, action):
        # 1. Convert the SAC action to two physical actuator commands.
        physical_action = self._map_action_to_physical_command(action)

        # 2. Apply the commands to Fluent.
        self._apply_actuator_commands(physical_action)

        # 3. Advance the CFD solution over one control interval.
        self._advance_cfd()

        # 4. Read the updated flow observation and control metrics.
        pressure_vector = self._read_pressure_observation()
        metrics = self._read_aerodynamic_metrics()

        # 5. Evaluate the private stage-dependent reward.
        reward, reward_info = self._compute_stage_dependent_reward(
            metrics,
            physical_action,
        )

        # 6. Construct the next history-augmented state.
        self.previous_action = physical_action.astype(np.float32)
        self.step_index += 1

        current_observation = self._build_instantaneous_observation(
            pressure_vector,
            metrics,
        )
        observation = self._stack_observation(current_observation)

        terminated = False
        truncated = self.step_index >= self.env_config.episode_length

        info = {
            "step": self.step_index,
            "actuator_1": float(physical_action[0]),
            "actuator_2": float(physical_action[1]),
            **metrics,
            **reward_info,
        }

        self.history.append(
            {
                "step": self.step_index,
                "reward": float(reward),
                "actuator_1": float(physical_action[0]),
                "actuator_2": float(physical_action[1]),
                "loss": float(metrics.get("loss", np.nan)),
                "efficiency": float(metrics.get("efficiency", np.nan)),
            }
        )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        if hasattr(self, "session") and self.session is not None:
            try:
                self.session.exit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# SAC experiment manager
# ---------------------------------------------------------------------------

class ExperimentManager:
    """Build vectorised environments and manage SAC training/testing."""

    def __init__(
        self,
        env_config: EnvConfig,
        rl_config: RLConfig,
        artifact_dir: str = "artifacts",
    ):
        self.env_config = env_config
        self.rl_config = rl_config

        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _build_single_env(
        self,
        env_config: EnvConfig,
        mode: str,
        history_dir: str,
    ):
        return CompressorEnv(
            env_config=env_config,
            rl_config=self.rl_config,
            mode=mode,
            history_dir=history_dir,
        )

    def _build_vec_env(self, mode: str):
        """Create one or more independent Fluent environments."""
        if self.rl_config.n_envs <= 1:
            return VecMonitor(
                DummyVecEnv(
                    [
                        lambda: self._build_single_env(
                            self.env_config,
                            mode,
                            f"history/{mode}",
                        )
                    ]
                )
            )

        if len(self.env_config.parallel_workdirs) < self.rl_config.n_envs:
            raise ValueError(
                "Insufficient parallel_workdirs for the requested n_envs."
            )

        factories = []

        for rank in range(self.rl_config.n_envs):
            workdir = self.env_config.parallel_workdirs[rank]

            def make_env(
                rank: int = rank,
                workdir: str = workdir,
            ):
                env_config = EnvConfig(
                    **{
                        **self.env_config.__dict__,
                        "workdir": workdir,
                    }
                )
                return self._build_single_env(
                    env_config,
                    mode,
                    f"history/{mode}_env{rank}",
                )

            factories.append(make_env)

        return VecMonitor(
            SubprocVecEnv(
                factories,
                start_method="spawn",
            )
        )

    def _build_model(self, env) -> SAC:
        """Construct the SAC model from the external RL configuration."""
        return SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.rl_config.learning_rate,
            batch_size=self.rl_config.batch_size,
            buffer_size=self.rl_config.buffer_size,
            learning_starts=self.rl_config.learning_starts,
            gamma=self.rl_config.gamma,
            tau=self.rl_config.tau,
            train_freq=(self.rl_config.train_freq, "step"),
            gradient_steps=self.rl_config.gradient_steps,
            ent_coef=self.rl_config.ent_coef,
            policy_kwargs=self.rl_config.policy_kwargs,
            tensorboard_log=str(
                self.artifact_dir / self.rl_config.tensorboard_log_dir
            ),
            verbose=1,
        )

    def train(
        self,
        train_steps: int,
        load_model_path: Optional[str] = None,
    ) -> Path:
        """Train a new SAC policy or continue from an existing checkpoint."""
        env = self._build_vec_env(mode="train")

        if load_model_path:
            model = SAC.load(load_model_path, env=env)
        else:
            model = self._build_model(env)

        model.learn(
            total_timesteps=int(train_steps),
            reset_num_timesteps=load_model_path is None,
        )

        output_path = self.artifact_dir / (
            f"case{self.env_config.case_id}_sac_model"
        )
        model.save(str(output_path))
        env.close()

        return output_path.with_suffix(".zip")

    def test(
        self,
        model_path: str,
        max_steps: Optional[int] = None,
    ) -> float:
        """Evaluate a trained policy in one Fluent environment."""
        env = DummyVecEnv(
            [
                lambda: self._build_single_env(
                    self.env_config,
                    "test",
                    "history/test",
                )
            ]
        )
        model = SAC.load(model_path, env=env)

        observation = env.reset()
        done = np.array([False])
        episode_return = 0.0
        step = 0

        while not bool(done[0]):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )
            observation, reward, done, _ = env.step(action)
            episode_return += float(reward[0])
            step += 1

            if max_steps is not None and step >= max_steps:
                break

        env.close()
        return episode_return
