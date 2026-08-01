"""
Public case runner for the DA-HA-SAC framework.

This file shows how a compressor-cascade case is connected to the reusable
framework. Exact paths, solver settings, sensor names, network settings,
training parameters, and manuscript-specific values are intentionally omitted.
"""

from pathlib import Path

from framework import EnvConfig, RLConfig, ExperimentManager


# ---------------------------------------------------------------------------
# Case configuration
# ---------------------------------------------------------------------------

def build_case_configs() -> tuple[EnvConfig, RLConfig]:
    """Create one public example configuration.

    Replace all placeholder values locally. The unpublished values used in the
    manuscript are not included in this repository.
    """

    env_config = EnvConfig(
        case_id=0,
        cas_path="PATH_TO_CASE_FILE.cas.h5",
        data_path="PATH_TO_INITIAL_DATA.dat.h5",
        workdir="PATH_TO_WORKING_DIRECTORY",

        actuator_names=(
            "ACTUATOR_BOUNDARY_1",
            "ACTUATOR_BOUNDARY_2",
        ),

        probe_names=(
            "PROBE_GROUP_1",
            "PROBE_GROUP_2",
        ),

        processor_count=0,
        cfd_steps_per_action=0,
        max_solver_iterations=0,
        episode_length=0,
        history_length=0,

        observation_dim=0,
        action_limit=0.0,

        parallel_workdirs=(
            "PATH_TO_PARALLEL_ENV_0",
            "PATH_TO_PARALLEL_ENV_1",
        ),

        show_gui="no_gui",
    )

    rl_config = RLConfig(
        action_dim=2,

        learning_rate=0.0,
        batch_size=0,
        buffer_size=0,
        learning_starts=0,
        gamma=0.0,
        tau=0.0,
        train_freq=0,
        gradient_steps=0,
        ent_coef="auto",

        n_envs=1,

        policy_kwargs={
            "net_arch": {
                "pi": [],
                "qf": [],
            }
        },
    )

    return env_config, rl_config


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    env_config, rl_config = build_case_configs()

    manager = ExperimentManager(
        env_config=env_config,
        rl_config=rl_config,
        artifact_dir="artifacts",
    )

    # The exact training length and checkpoint policy are not disclosed.
    train_steps = 0

    model_path = manager.train(
        train_steps=train_steps,
        load_model_path=None,
    )

    print(f"Saved model: {model_path}")

    # Optional deterministic evaluation:
    #
    # episode_return = manager.test(
    #     model_path=str(model_path),
    #     max_steps=env_config.episode_length,
    # )
    # print(f"Episode return: {episode_return}")


if __name__ == "__main__":
    main()
