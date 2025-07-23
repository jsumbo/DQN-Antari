#!/usr/bin/env python3
"""Plays an Atari Game using the presaved model.

Usage:
    python play.py --model_path path/to/model.zip
If --model_path is not provided, defaults to './models/best_model.zip'.
"""
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import ale_py  # Required to register ALE environments
import time
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Play Atari game with a trained DQN model.")
parser.add_argument('--model_path', type=str, default='./models/best_model.zip',
                    help='Path to the trained model zip file (default: ./models/best_model.zip)')
args = parser.parse_args()

# Load the environment
def make_env(env_id):
    env = gym.make(env_id, render_mode='human')  # Changed to 'human'
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env

# Environment and model path
ENV_ID = "ALE/Assault-v5"
MODEL_PATH = args.model_path

# Create environment
env = make_env(ENV_ID)

# Load model with custom objects to handle compatibility issues
try:
    print(f"Loading model from: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        print(f"File exists, size: {os.path.getsize(MODEL_PATH)} bytes")
    else:
        print(f"Model file does not exist: {MODEL_PATH}")
        print("Please provide a valid model path using --model_path.")
        exit(1)
    model = DQN.load(MODEL_PATH, custom_objects={'exploration_schedule': None, 'lr_schedule': None})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if the model file is compatible with your current stable-baselines3 version")
    exit(1)

# Run the agent
for episode in range(5):  # Run 5 episodes
    obs = env.reset()  # Remove unpacking since vectorized env returns different format
    done = False
    episode_reward = 0
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)  # Expect 4 values
        done = done[0]  # Extract scalar for single environment
        episode_reward += reward[0]
        step_count += 1
        time.sleep(0.02)  # Slow down for visibility
    
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")

env.close()