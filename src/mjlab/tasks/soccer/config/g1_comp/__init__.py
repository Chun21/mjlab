from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import g1_comp_soccer_env_cfg
from .rl_cfg import g1_comp_soccer_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Soccer-M-G1-Comp",
  env_cfg=g1_comp_soccer_env_cfg(),
  play_env_cfg=g1_comp_soccer_env_cfg(play=True),
  rl_cfg=g1_comp_soccer_ppo_runner_cfg(),
)
