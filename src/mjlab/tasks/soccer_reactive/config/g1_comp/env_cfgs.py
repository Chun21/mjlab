"""G1 Comp reactive soccer environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.soccer_reactive.paper_env_cfg import make_reactive_soccer_env_cfg


def g1_comp_reactive_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the minimal G1 Comp reactive soccer environment configuration."""
  return make_reactive_soccer_env_cfg(play=play)
