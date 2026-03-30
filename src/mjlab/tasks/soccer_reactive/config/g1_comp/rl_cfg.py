"""RL configuration for the G1 Comp reactive soccer task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

REACTIVE_RUNNER_CLASS = (
  "mjlab.tasks.soccer_reactive.rl.reactive_soccer_runner:ReactiveSoccerRunner"
)
REACTIVE_ALGO_CLASS = (
  "mjlab.tasks.soccer_reactive.rl.reactive_soccer_ppo:ReactiveSoccerPPO"
)
REACTIVE_MODEL_CLASS = (
  "mjlab.tasks.soccer_reactive.models.history_encoder_actor_critic:ReactiveSoccerActorCritic"
)


def g1_comp_reactive_soccer_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the minimal runner configuration for the reactive soccer task."""
  cfg = RslRlOnPolicyRunnerCfg(
    class_name=REACTIVE_RUNNER_CLASS,
    actor=RslRlModelCfg(
      class_name=REACTIVE_MODEL_CLASS,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      class_name=REACTIVE_MODEL_CLASS,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(class_name=REACTIVE_ALGO_CLASS),
    experiment_name="g1_comp_reactive_soccer",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
  cfg.amp = {
    "motion_root": "data/soccer_amp/motions_unified",
    "discriminator_hidden_dims": (128, 64),
  }
  return cfg
