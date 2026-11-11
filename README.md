<div align="center">

<img src="./assets/image.png" alt="Repository Banner" width="800">

<h1 align="center" >

<img width="25px" src="https://em-content.zobj.net/source/microsoft-teams/337/heart-with-ribbon_1f49d.png"/>
<img width="25px" src="https://em-content.zobj.net/source/microsoft-teams/337/heart-with-ribbon_1f49d.png"/>
<img width="25px" src="https://em-content.zobj.net/source/microsoft-teams/337/heart-with-ribbon_1f49d.png"/>
<b>Cliff Walking with Q-learning</b>
<img width="25px" src="https://em-content.zobj.net/source/microsoft-teams/337/heart-with-ribbon_1f49d.png"/>
<img width="25px" src="https://em-content.zobj.net/source/microsoft-teams/337/heart-with-ribbon_1f49d.png"/>
<img width="25px" src="https://em-content.zobj.net/source/microsoft-teams/337/heart-with-ribbon_1f49d.png"/>

</h1>

[![GitHub](https://img.shields.io/badge/GitHub-vividblueprint-181717?logo=github&logoColor=white)](https://github.com/vividblueprint)
[![Issues](https://img.shields.io/github/issues/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/pulls)
[![Contributors](https://img.shields.io/github/contributors/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/stargazers)
[![Forks](https://img.shields.io/github/forks/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/forks)
[![Watchers](https://img.shields.io/github/watchers/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/watchers)
[![Last Commit](https://img.shields.io/github/last-commit/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/commits)
[![Monthly Commits](https://img.shields.io/github/commit-activity/m/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/commits)
[![Languages](https://img.shields.io/github/languages/count/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl)
[![Code Size](https://img.shields.io/github/languages/code-size/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl)
[![Repo Size](https://img.shields.io/github/repo-size/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl)
[![Total Commits](https://img.shields.io/github/commit-activity/t/vividblueprint/cliffwalking_vanilla_rl?style=flat&label=Total%20Commits)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/commits)
[![Files](https://img.shields.io/github/directory-file-count/vividblueprint/cliffwalking_vanilla_rl?type=file&style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/tree/HEAD)
[![License](https://img.shields.io/github/license/vividblueprint/cliffwalking_vanilla_rl?style=flat)](https://github.com/vividblueprint/cliffwalking_vanilla_rl/blob/HEAD/LICENSE)

</div>

---

## Run

Install dependencies and train:

```bash
uv sync
uv run train.py --episodes 500
```

The agent starts at the bottom-left, reaches the goal at the bottom-right, and receives a large penalty if it steps into the cliff. Training saves `q_table.npy` and `policy.png` under `runs/cliffwalk/`.

Show the learned policy in a window:

```bash
uv run train.py --show
```

Display a random-action episode in the custom CliffWalking environment:

```bash
uv run test_env.py
```

Run the trained greedy policy in the environment window:

```bash
uv run play.py
```

Pass a Q-table path explicitly if it is saved elsewhere:

```bash
uv run play.py runs/cliffwalk/q_table.npy
```

Useful options are `--rows`, `--cols`, `--alpha`, `--gamma`, `--epsilon`, `--seed`, and `--out`. Actions are `0=up`, `1=right`, `2=down`, and `3=left`.

## Compare tabular agents

`src/cliffwalk/agent_multi.py` includes Q-learning, Double Q-learning, Expected SARSA, SARSA, SARSA(lambda), and first-visit Monte Carlo control. Select an agent by name:

```python
from cliffwalk import make_agent

agent = make_agent("double_q", n_states=env.n_states, n_actions=env.n_actions)
```

The training script supports the same algorithms with `--agent`:

```bash
uv run train.py --agent sarsa --episodes 500
uv run train.py --agent monte_carlo --episodes 500
```

For SARSA and SARSA(lambda), choose the next action before updating, then pass it as `next_action` to `learn`. Monte Carlo control updates at episode end; pass `done=True` on the final transition.

## Files

- `src/cliffwalk/env.py` — grid and step rules
- `src/cliffwalk/agent.py` — epsilon-greedy Q-learning
- `src/cliffwalk/agent_multi.py` — additional tabular control algorithms
- `src/cliffwalk/visualize.py` — simple board and policy drawing
- `train.py` — training loop and output files
