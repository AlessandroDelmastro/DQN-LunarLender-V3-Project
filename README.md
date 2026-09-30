# LunarLander-v3: A Comparison of Different DQN Agents' Performance

A Deep Q-Network (DQN) study on Gymnasium's **LunarLander-v3** environment, comparing **soft** (Polyak averaging) and **hard** (periodic copy) target-network update rules under different learning rates and interpolation factors. Each configuration is trained over **5 independent runs**. Results are aggregated with 95% confidence intervals and evaluated in a greedy test phase of 400 episodes per run.

> Project for the **Deep Learning Applications** course. Author: **Alessandro Delmastro**.

---

## Results at a glance

Greedy evaluation (ε = 0), 400 test episodes per run, 5 runs per configuration. An episode counts as a **success** when its score is ≥ 200.

| Experiment      | Target update | α (lr) | τ / update every | ε decay | Mean Success Rate | Global Mean Reward | Avg Std Reward |
|-----------------|:-------------:|:------:|:----------------:|:-------:|:-----------------:|:------------------:|:--------------:|
| **low_lr_hard** | hard          | 5e-4   | every 10 learn steps | 0.999 | **87.1%**      | **236.42**         | **55.28**      |
| baseline_hard   | hard          | 1e-3   | every 10 learn steps | 0.999 | 84.2%          | 227.67             | 64.76          |
| high_tau_soft   | soft          | 1e-3   | τ = 5e-3         | 0.995   | 77.9%             | 211.98             | 71.11          |
| low_lr_soft     | soft          | 5e-4   | τ = 1e-3         | 0.995   | 64.1%             | 194.35             | 74.16          |
| baseline_soft   | soft          | 1e-3   | τ = 1e-3         | 0.995   | 63.3%             | 178.16             | 99.18          |

**Key findings**

- **Hard updates won at test time.** Both hard-update agents reached the highest success rates and the lowest reward variance, even though their training was noisier.
- **Training behaviour.** Hard-update agents showed strong Q-value overestimation (a sharp rise and then a fall between roughly episodes 500 and 1000) and longer episodes. Soft-update agents converged more smoothly, to lower and more realistic Q-values.
- **Soft updates may stabilise too early.** Smoother, faster convergence seems to reduce exploration and lead to suboptimal policies. `high_tau_soft` supports this: its larger τ allows bigger target shifts, and it performed best among the soft agents.
- **A lower learning rate helps hard updates.** `low_lr_hard` makes the Q-function updates less sharp, and it produced the best and most stable policy.

The full discussion, including proposed improvements, is at the bottom of `Testing.ipynb`.

---

## Repository structure

```
.
├── DQN_tools.py        # Network, ReplayMemory, Agent + training, saving, plotting and testing utilities
├── Training.ipynb      # Environment setup, experiment configurations, training loop (5 configs × 5 runs)
├── Testing.ipynb       # Training plots (5×5 grid), aggregated greedy test, discussion of results
└── checkpoints/        # Saved models and metrics (.pth), created automatically during training
```

### `DQN_tools.py`

| Component | Description |
|---|---|
| `Network` | MLP Q-network: `8 → 128 → 128 → 4`, ReLU activations |
| `ReplayMemory` | FIFO experience replay buffer that uses uniform random sampling |
| `Agent` | DQN agent with a local (online) network and a target network. Uses an ε-greedy policy, the Adam optimiser and an MSE loss on the Bellman target, with `soft_update()` / `hard_update()` for the target network |
| `train_agent` | Training loop with ε decay, metric logging, a checkpoint every 100 episodes and early stopping once the environment is solved |
| `save_model` | Saves the network weights, optimiser state, configuration, metrics and a solved/failed flag to a `.pth` file |
| `plot_experiments_grid_universal` | Groups runs by experiment and plots a 5 × N grid (reward, loss, avg Q, ε, episode duration). Each plot shows the mean across runs with a 95% t-Student confidence interval, and shorter runs are padded with their last value |
| `test_all_experiments_aggregated` | Runs a greedy evaluation of every final model and returns a pandas summary table |

---

## Method

### Shared hyperparameters

| Parameter | Value |
|---|---|
| Replay buffer size | 100,000 |
| Batch size | 100 |
| Learning frequency | every 4 environment steps |
| Discount factor γ | 0.99 |
| ε start → end | 1.0 → 0.01 (multiplicative decay per episode) |
| Max steps per episode | 1,000 |
| Max training episodes | 2,000 |
| Runs per configuration | 5 |
| Solved criterion | average score over the last 100 episodes ≥ 200 |

### Target-network update rules

- **Soft**: θ′ ← τ·θ + (1 − τ)·θ′, applied after every learning step.
- **Hard**: θ′ ← θ, applied every `update_every` learning steps.

The hard-update configurations use a slower ε decay (0.999 instead of 0.995). In earlier trials, the faster decay ended exploration too soon, and the hard-update agents did not converge within 2,000 episodes.

### Metrics tracked during training

The following are recorded for every run:

- reward per episode
- loss per learning step
- average max-Q value per episode
- ε
- wall-clock duration of each episode

---

## Getting started

The project was designed to run on **Google Colab** with **Google Drive** as storage. A GPU is used when available.

### Requirements

- Python 3.10+
- `gymnasium[box2d]` (requires `swig`)
- `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`

The first cell of each notebook installs any missing packages automatically.

### 1. Setup

1. Clone or download this repository and upload the folder to your Google Drive.
2. Both notebooks set the project path with this line:
   ```python
   project_path = "/content/drive/MyDrive/DQN project V3 - Copia"
   ```
   Change it (and the `save_path` / `checkpoint_path` variables) to match the folder's location on your Drive.
3. You do not need to run `DQN_tools.py` directly. Both notebooks import it.

### 2. Training (`Training.ipynb`)

Run the cells in order:

1. Install and import the dependencies, mount Drive, and import `DQN_tools`.
2. Create the `LunarLander-v3` environment (state size 8, 4 actions).
3. Set the run parameters: `number_episodes`, `num_runs` and `eval_freq`.
4. Define the `configs` dictionary. This is where you add or edit experiments.
5. Run the main loop. It trains every configuration `num_runs` times and saves the results in `checkpoints/`.

The files are named like this:

```
<experiment>_run<i>_checkpoint_<episode>.pth   # periodic checkpoint (every 100 episodes)
<experiment>_run<i>_solved_<episode>.pth       # final model, environment solved
<experiment>_run<i>_failed_<episode>.pth       # final model, episode limit reached
```

> ⚠️ Full training (5 configurations × 5 runs × up to 2,000 episodes) takes a long time. You can lower `num_runs` or `number_episodes` for a quick test.

### 3. Evaluation (`Testing.ipynb`)

1. Run the setup cells to install packages, mount Drive and import the plotting and testing functions.
2. Plot the training curves:
   ```python
   plot_experiments_grid_universal(checkpoint_dir=checkpoint_path, eval_freq=50)
   ```
   This saves the grid as `Complete experiment grid.png`.
3. Run the greedy test:
   ```python
   test_all_experiments_aggregated(
       checkpoint_dir=checkpoint_path,
       env_name='LunarLander-v3',
       num_test_episodes=400,   # more episodes = longer run time
       success_threshold=200    # official LunarLander "solved" score
   )
   ```
4. Read the discussion section at the end of the notebook.

Both functions skip the periodic `*_checkpoint_*` files and use only the final models (`solved` / `failed`) of each run.

---

## Possible improvements

- **Soft updates**: slow down ε decay to widen the exploration window, and increase τ systematically, as the result of `high_tau_soft` suggests.
- **Hard updates**: try a slightly lower γ to shorten the effective horizon and limit how far the max-operator overestimation propagates. A lower α has already proved beneficial.

---

## Author

**Alessandro Delmastro**, Deep Learning Applications course.
