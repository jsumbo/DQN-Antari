# Project Overview

This project implements a Deep Q-Network (DQN) agent to play the Atari Assault game using Stable Baselines3 and Gymnasium. The goal is to train an agent to play the Assault-v5 Atari environment using reinforcement learning, compare different policy architectures, and evaluate the agent's performance.

## Environment

- Game: Atari Assault (ALE/Assault-v5)
- Learning Algorithm: Deep Q-Network (DQN)
- Policy Networks: Convolutional Neural Network (CnnPolicy) **and** Multilayer Perceptron (MlpPolicy)

## Policy Comparison

Both `CnnPolicy` and `MlpPolicy` are trained and compared in `train.py`. The script automatically evaluates both and saves the best-performing model as `dqn_model.zip` for easy use in `play.py`.

## How to Run

### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Train the Agent
```
python train.py
```
- This will train both CnnPolicy and MlpPolicy agents, compare their performance, and save the best model as `dqn_model.zip`.
- At the end, a summary table of hyperparameters and results will be printed.

### 3. Play with the Trained Agent
```
python play.py --model_path dqn_model.zip
```
- By default, `play.py` loads `dqn_model.zip` and runs the agent in the Assault-v5 environment, rendering the game window.

## Hyperparameter Exploration & Results

Below is a sample of the results table printed by `train.py`:

| Name         | Policy     | LR     | Gamma | Batch | Eps Start | Eps End | Mean Reward | Mean Length |
|--------------|------------|--------|-------|-------|-----------|---------|-------------|-------------|
| CNN_Optimized| CnnPolicy  | 0.0005 | 0.9   | 64    | 0.01      | 0.01    |   401.00    |   1200.00   |
| MLP_HighLR   | MlpPolicy  | 0.005  | 0.95  | 64    | 0.001     | 0.02    |   150.00    |    800.00   |

- The best model (highest mean reward) is saved as `dqn_model.zip`.

## Key Components

### train.py
- Trains both CnnPolicy and MlpPolicy DQN agents.
- Uses Atari wrappers and frame stacking for temporal context.
- Includes callbacks for evaluation and early stopping.
- Logs training data for visualization.
- Prints a summary table of hyperparameter results.

### play.py
- Loads the trained model (default: `dqn_model.zip`).
- Evaluates the agent's performance in the Atari environment.
- Renders gameplay for inspection.
- Model path can be changed with `--model_path` argument.

## Prerequisites

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Results Visualization
- Training logs and reward trends are printed and can be visualized with TensorBoard.
- The agent learns to fire strategically, dodge enemies, and maximize score.

## Contributions

| Name           | Contribution Areas                      |
|----------------|-----------------------------------------|
| Jallah Sumbo   | Play script, Training, Documentation    |
| Festus         | Train Script, Training, Video           |
