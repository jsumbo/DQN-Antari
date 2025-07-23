# -*- coding: utf-8 -*-
"""train.ipynb
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
import gymnasium as gym
import ale_py

# Create the Atari environment
env = "ALE/Assault-v5"

# Function to create and wrap the environment
def make_env(env_id, seed=42):
    env = gym.make(env_id, render_mode='rgb_array')
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    # No VecTransposeImage used, so both train and eval envs are identical
    env.seed(seed)
    return env

# Create training and evaluation environments
train_env = make_env(env)
eval_env = make_env(env)

# Define optimized hyperparameter sets
hyperparams_sets = [
    {
        'policy': 'CnnPolicy',
        'learning_rate': 5e-4,
        'gamma': 0.90,
        'batch_size': 64,
        'buffer_size': 2000,  # Reduced from 5000
        'exploration_fraction': 1.0,
        'exploration_initial_eps': 0.01,
        'exploration_final_eps': 0.01,
        'double_q': True,
        'prioritized_replay': True,
        'name': 'CNN_Optimized'
    },
    {
        'policy': 'MlpPolicy',
        'learning_rate': 5e-3,
        'gamma': 0.95,
        'batch_size': 64,
        'buffer_size': 2000,  # Reduced from 5000
        'exploration_fraction': 1.0,
        'exploration_initial_eps': 0.001,
        'exploration_final_eps': 0.02,
        'double_q': True,
        'prioritized_replay': True,
        'name': 'MLP_HighLR'
    }
]

def model_dqn(params, train_env, eval_env, total_timesteps=15000, eval_freq=15000):
    """Train a DQN model with the given parameters."""
    model = DQN(
        policy=params['policy'],
        env=train_env,
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        batch_size=params['batch_size'],
        buffer_size=params['buffer_size'],
        exploration_fraction=params['exploration_fraction'],
        exploration_initial_eps=params['exploration_initial_eps'],
        exploration_final_eps=params['exploration_final_eps'],
        verbose=1,
        tensorboard_log=f"./tensorboard/{params['name']}/"
    )

    # Early stopping callback to save compute resources
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=20, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_models/{params['name']}/",
        log_path=f"./logs/{params['name']}/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the final model
    model.save(f"dqn_{params['name']}_final")

    return model

# Create directories for saving models and logs
os.makedirs("./best_models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)

# Train models with different hyperparameters
trained_models = []
for params in hyperparams_sets:
    print(f"\nTraining with {params['name']} configuration...")
    train_env = make_env(env)
    eval_env = make_env(env)
    if params['policy'] == 'CnnPolicy':
        train_env = VecTransposeImage(train_env)
        eval_env = VecTransposeImage(eval_env)
    model = model_dqn(params, train_env, eval_env, total_timesteps=15000)
    trained_models.append(model)
    print(f"Completed training for {params['name']}")

def evaluate_model(model, env, n_episodes=10):
    all_rewards = []
    all_lengths = []

    for i in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)

    return np.mean(all_rewards), np.mean(all_lengths)

# Evaluate all trained models
results = []
for i, model in enumerate(trained_models):
    mean_reward, mean_length = evaluate_model(model, eval_env)
    results.append({
        'name': hyperparams_sets[i]['name'],
        'policy': hyperparams_sets[i]['policy'],
        'learning_rate': hyperparams_sets[i]['learning_rate'],
        'gamma': hyperparams_sets[i]['gamma'],
        'batch_size': hyperparams_sets[i]['batch_size'],
        'exploration_initial_eps': hyperparams_sets[i]['exploration_initial_eps'],
        'exploration_final_eps': hyperparams_sets[i]['exploration_final_eps'],
        'mean_reward': mean_reward,
        'mean_length': mean_length
    })
    print(f"{hyperparams_sets[i]['name']}: Mean Reward = {mean_reward:.2f}, Mean Length = {mean_length:.2f}")

# Select the best model based on mean reward
best_result = max(results, key=lambda x: x['mean_reward'])
best_model_index = results.index(best_result)
best_model = trained_models[best_model_index]
print(f"\nBest model: {best_result['name']} (Policy: {best_result['policy']}) with mean reward {best_result['mean_reward']:.2f}")

# Save the best model as dqn_model.zip for easy loading in play.py
best_model.save("dqn_model.zip")
print("Best model saved as dqn_model.zip")

# Print a summary table of results
print("\nHyperparameter Results Table:")
print("| Name         | Policy     | LR     | Gamma | Batch | Eps Start | Eps End | Mean Reward | Mean Length |")
print("|--------------|------------|--------|-------|-------|-----------|---------|-------------|-------------|")
for r in results:
    print(f"| {r['name']:<12} | {r['policy']:<10} | {r['learning_rate']:<6} | {r['gamma']:<5} | {r['batch_size']:<5} | {r['exploration_initial_eps']:<9} | {r['exploration_final_eps']:<7} | {r['mean_reward']:<11.2f} | {r['mean_length']:<11.2f} |")

# Function to load and plot training logs
def plot (log_dir, title):
    # Load the monitoring data
    x, y = [], []
    with open(os.path.join(log_dir, "mon.csv"), 'r') as f:
        next(f)
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                x.append(int(parts[0]))
                y.append(float(parts[1]))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.show()

# Plot results for each configuration
for params in hyperparams_sets:
    log_dir = f"./logs/{params['name']}/"
    if os.path.exists(os.path.join(log_dir, "mon.csv")):
        plot(log_dir, f"Training Progress: {params['name']}")

from IPython.display import display, HTML

# Create a table of hyperparameters and results
table_html = """
<table border="1">
    <tr>
        <th>Configuration</th>
        <th>Policy</th>
        <th>Learning Rate</th>
        <th>Gamma</th>
        <th>Batch Size</th>
        <th>Epsilon Start</th>
        <th>Epsilon End</th>
        <th>Mean Reward</th>
        <th>Mean Length</th>
    </tr>
"""

for result in results:
    params = result['params']
    table_html += f"""
    <tr>
        <td>{result['name']}</td>
        <td>{params['policy']}</td>
        <td>{params['learning_rate']}</td>
        <td>{params['gamma']}</td>
        <td>{params['batch_size']}</td>
        <td>{params['exploration_initial_eps']}</td>
        <td>{params['exploration_final_eps']}</td>
        <td>{result['mean_reward']:.2f}</td>
        <td>{result['mean_length']:.2f}</td>
    </tr>
    """

table_html += "</table>"
display(HTML(table_html))