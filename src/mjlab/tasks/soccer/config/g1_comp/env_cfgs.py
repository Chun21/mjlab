"""G1 Comp soccer environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.soccer.soccer_env_cfg import make_soccer_env_cfg


def g1_comp_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 Comp single-robot soccer environment configuration."""
  return make_soccer_env_cfg(play=play)
