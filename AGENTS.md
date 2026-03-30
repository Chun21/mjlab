# AGENTS.md

## 当前主任务
- 复现论文《Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots》。
- 在 `mjlab` 中实现独立的论文式任务栈：`src/mjlab/tasks/soccer_reactive/`。
- 目标不是只把训练框架跑通，而是要在仿真中训练出**稳定连续**的“找球 → 追球 → 调整朝向 → 射门 → 下一球”行为。

## 论文基线
- 当前工作是在**复现论文**，不是自由发挥做一个相似系统。
- 复现目标论文：`Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots`。
- 论文 PDF 已经拷贝到本机，后续 agent 应优先以本地论文内容为基线理解方法、模块边界和训练目标。
- 本地论文路径：`/home/chunyu/papers/Vision-Driver Reactive Soccer Skills.pdf`。
- 如果实现细节、训练设计、模块命名与论文主链路冲突，应优先检查论文与现有设计文档，再决定是否调整。

## 已确定范围
- 走**方案 1：独立论文式复现**，不要在现有 `tasks/soccer` 上做修补式增强。
- 场景范围：**单机器人、全场、空门、仿真训练与评测闭环**。
- 本轮**不做**真实视觉部署、YOLO、真机 odometry 接口、多机器人对抗。

## 关键约束
- 目标部署对象：`Unitree G1`。
- 频率对齐：
  - 相机：`30 Hz`
  - 里程计：`20 Hz`
  - 控制：`50 Hz`
- 训练预算：`8 × 24G RTX 4090`，希望 `24 小时` 内完成一次主训练与固定评测闭环。

## 实现原则
- 新增独立包：`src/mjlab/tasks/soccer_reactive/`。
- 优先复用底层通用基础设施（env/registry/robot asset/field specs），但不要复用旧 soccer teacher 的任务语义。
- 核心方法必须覆盖：
  - virtual perception
  - 1 秒历史窗口 encoder-decoder
  - dual critic PPO
  - AMP
  - symmetry loss
  - ball-only reset

## 当前文档基线
- 设计文档：`docs/plans/2026-03-30-reactive-soccer-paper-design.md`
- 实现计划：`docs/plans/2026-03-30-reactive-soccer-paper-implementation.md`
- 后续实现、测试、训练与评测，应以这两份文档为主基线。

## Code review
 当实现一个新模块后，开个子agent进行该功能的code review