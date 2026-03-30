from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.soccer_reactive.config.g1_comp.env_cfgs import (
  g1_comp_reactive_soccer_env_cfg,
)
from mjlab.tasks.soccer_reactive.config.g1_comp.rl_cfg import (
  g1_comp_reactive_soccer_runner_cfg,
)
from mjlab.tasks.soccer_reactive.rl.reactive_soccer_runner import ReactiveSoccerRunner

register_mjlab_task(
  task_id="Mjlab-Soccer-G1-Comp-ReactivePaper",
  env_cfg=g1_comp_reactive_soccer_env_cfg(),
  play_env_cfg=g1_comp_reactive_soccer_env_cfg(play=True),
  rl_cfg=g1_comp_reactive_soccer_runner_cfg(),
  runner_cls=ReactiveSoccerRunner,
)
