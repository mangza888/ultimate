#!/usr/bin/env python3
# ai/reinforcement_learning.py - Reinforcement Learning Models
# PPO, SAC, A2C, DDPG สำหรับการเทรด

import os
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from gymnasium import spaces
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.sac import SAC, SACConfig
from utils.config_manager import get_config
from utils.logger import get_logger

class TradingEnvironment(gym.Env):
    """Trading Environment สำหรับ RL"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        super().__init__()
        
        self.data = data
        self.config = config
        self.logger = get_logger()
        
        # Environment settings
        self.initial_balance = config.get('initial_balance', 10000)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.max_position = config.get('max_position', 1.0)
        self.sequence_length = config.get('sequence_length', 60)
        
        # Action space: [position_change] (-1 to 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [features + portfolio_state]
        obs_dim = len(self.data.columns) - 1 + 3  # features + [balance, position, portfolio_value]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.sequence_length, obs_dim), 
            dtype=np.float32
        )
        
        # State variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        self.position = 0  # -1 to 1 (short to long)
        self.portfolio_value = self.initial_balance
        self.total_trades = 0
        self.total_profit = 0
        
        # Performance tracking
        self.max_portfolio_value = self.initial_balance
        self.drawdown = 0
        self.max_drawdown = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, self._get_info()
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = self._execute_action(action[0], current_price)
        
        # Update portfolio
        self._update_portfolio(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = (self.current_step >= len(self.data) - 1) or (self.portfolio_value <= 0)
        truncated = done
        
        return self._get_observation(), reward, done, truncated, self._get_info()
    
    def _execute_action(self, action: float, current_price: float) -> float:
        """Execute trading action"""
        # Normalize action to [-1, 1]
        target_position = np.clip(action, -1, 1)
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Execute trade if significant change
        if abs(position_change) > 0.1:
            # Calculate trade size
            trade_size = abs(position_change) * self.initial_balance / current_price
            
            # Transaction cost
            transaction_cost = trade_size * current_price * self.transaction_cost
            
            # Update balance and position
            self.balance -= transaction_cost
            self.position = target_position
            self.total_trades += 1
            
            # Immediate reward based on transaction cost
            reward = -transaction_cost / self.initial_balance
        else:
            reward = 0
        
        return reward
    
    def _update_portfolio(self, current_price: float):
        """Update portfolio value and metrics"""
        # Calculate portfolio value
        position_value = self.position * self.initial_balance
        self.portfolio_value = self.balance + position_value
        
        # Update max portfolio value
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        # Calculate drawdown
        self.drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        if self.drawdown > self.max_drawdown:
            self.max_drawdown = self.drawdown
    
    def _get_observation(self):
        """Get current observation"""
        start_idx = max(0, self.current_step - self.sequence_length)
        end_idx = self.current_step
        
        # Get price features (exclude 'close' column)
        feature_cols = [col for col in self.data.columns if col != 'close']
        features = self.data[feature_cols].iloc[start_idx:end_idx].values
        
        # Pad if necessary
        if len(features) < self.sequence_length:
            padding = np.repeat(features[0:1], self.sequence_length - len(features), axis=0)
            features = np.vstack([padding, features])
        
        # Portfolio state
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.position,
            self.portfolio_value / self.initial_balance
        ])
        
        # Broadcast portfolio state across sequence
        portfolio_features = np.broadcast_to(
            portfolio_state, (self.sequence_length, 3)
        )
        
        # Combine features
        observation = np.hstack([features, portfolio_features]).astype(np.float32)
        
        return observation
    
    def _get_info(self):
        """Get environment info"""
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        return {
            'portfolio_value': self.portfolio_value,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'max_drawdown': self.max_drawdown,
            'position': self.position,
            'balance': self.balance
        }

class RLModelFactory:
    """Factory สำหรับสร้าง RL Models"""
    
    @staticmethod
    def create_model(algorithm: str, env, config: Dict[str, Any]):
        """สร้าง RL model ตาม algorithm"""
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if algorithm.upper() == 'PPO':
            return PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=config.get('learning_rate', 0.0003),
                n_steps=config.get('n_steps', 2048),
                batch_size=config.get('batch_size', 64),
                n_epochs=config.get('n_epochs', 10),
                gamma=config.get('gamma', 0.99),
                gae_lambda=config.get('gae_lambda', 0.95),
                clip_range=config.get('clip_range', 0.2),
                ent_coef=config.get('ent_coef', 0.01),
                vf_coef=config.get('vf_coef', 0.5),
                max_grad_norm=config.get('max_grad_norm', 0.5),
                device=device,
                verbose=1
            )
        
        elif algorithm.upper() == 'SAC':
            return SAC(
                policy='MlpPolicy',
                env=env,
                learning_rate=config.get('learning_rate', 0.0003),
                buffer_size=config.get('buffer_size', 1000000),
                learning_starts=config.get('learning_starts', 100),
                batch_size=config.get('batch_size', 256),
                tau=config.get('tau', 0.005),
                gamma=config.get('gamma', 0.99),
                train_freq=config.get('train_freq', 1),
                gradient_steps=config.get('gradient_steps', 1),
                ent_coef=config.get('ent_coef', 'auto'),
                device=device,
                verbose=1
            )
        
        elif algorithm.upper() == 'A2C':
            return A2C(
                policy='MlpPolicy',
                env=env,
                learning_rate=config.get('learning_rate', 0.0007),
                n_steps=config.get('n_steps', 5),
                gamma=config.get('gamma', 0.99),
                gae_lambda=config.get('gae_lambda', 1.0),
                ent_coef=config.get('ent_coef', 0.01),
                vf_coef=config.get('vf_coef', 0.25),
                max_grad_norm=config.get('max_grad_norm', 0.5),
                device=device,
                verbose=1
            )
        
        elif algorithm.upper() == 'DDPG':
            return DDPG(
                policy='MlpPolicy',
                env=env,
                learning_rate=config.get('learning_rate', 0.001),
                buffer_size=config.get('buffer_size', 1000000),
                learning_starts=config.get('learning_starts', 100),
                batch_size=config.get('batch_size', 100),
                tau=config.get('tau', 0.005),
                gamma=config.get('gamma', 0.99),
                train_freq=config.get('train_freq', 1),
                gradient_steps=config.get('gradient_steps', 1),
                device=device,
                verbose=1
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

class RLTrainer:
    """RL Model Trainer"""
    
    def __init__(self, config_path: str = "config/ai_models.yaml"):
        self.config = get_config(config_path)
        self.logger = get_logger()
        
    def train_model(self, algorithm: str, data: pd.DataFrame, 
                   config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train RL model"""
        
        try:
            self.logger.info(f"Training {algorithm} model...")
            
            # Create environment
            env = TradingEnvironment(data, config)
            env = Monitor(env)
            
            # Create model
            model = RLModelFactory.create_model(algorithm, env, config)
            
            # Training callbacks
            callbacks = self._create_callbacks(env, config)
            
            # Train model
            total_timesteps = config.get('total_timesteps', 100000)
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Evaluate model
            metrics = self._evaluate_model(model, env)
            
            self.logger.success(f"{algorithm} training completed")
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error training {algorithm}: {e}")
            raise
    
    def _create_callbacks(self, env, config):
        """Create training callbacks"""
        callbacks = []
        
        # Evaluation callback
        eval_freq = config.get('eval_freq', 10000)
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"models/rl_best/",
            log_path=f"logs/rl_eval/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Stop training on reward threshold
        target_reward = config.get('target_reward', 0.1)
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=target_reward,
            verbose=1
        )
        callbacks.append(stop_callback)
        
        return callbacks
    
    def _evaluate_model(self, model, env) -> Dict[str, float]:
        """Evaluate trained model"""
        try:
            # Run evaluation episodes
            n_eval_episodes = 10
            episode_rewards = []
            episode_lengths = []
            
            for _ in range(n_eval_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # Calculate metrics
            metrics = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {}

class RayRLTrainer:
    """Ray RLlib Trainer สำหรับ Distributed RL"""
    
    def __init__(self, config_path: str = "config/ai_models.yaml"):
        self.config = get_config(config_path)
        self.logger = get_logger()
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=4)
    
    def train_distributed_model(self, algorithm: str, data: pd.DataFrame,
                              config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train distributed RL model with Ray"""
        
        try:
            self.logger.info(f"Training distributed {algorithm} model...")
            
            # Register environment
            def env_creator(env_config):
                return TradingEnvironment(data, env_config)
            
            from ray.tune.registry import register_env
            register_env("trading_env", env_creator)
            
            # Configure training
            if algorithm.upper() == 'PPO':
                trainer_config = {
                    "env": "trading_env",
                    "env_config": config,
                    "framework": "torch",
                    "num_gpus": 0,
                    "num_workers": 4,
                    "lr": config.get('lr', 0.0003),
                    "train_batch_size": config.get('train_batch_size', 4000),
                    "sgd_minibatch_size": config.get('sgd_minibatch_size', 128),
                    "num_sgd_iter": config.get('num_sgd_iter', 30),
                    "gamma": config.get('gamma', 0.99),
                    "lambda": config.get('lambda', 0.95),
                    "clip_param": config.get('clip_param', 0.2),
                    "entropy_coeff": config.get('entropy_coeff', 0.01),
                }
                
                trainer = ppo.PPOTrainer(config=trainer_config)
                
            elif algorithm.upper() == 'SAC':
                trainer_config = {
                    "env": "trading_env",
                    "env_config": config,
                    "framework": "torch",
                    "num_gpus": 1,
                    "num_workers": 4,
                    "lr": config.get('lr', 0.0003),
                    "replay_buffer_size": config.get('replay_buffer_size', 1000000),
                    "train_batch_size": config.get('train_batch_size', 256),
                    "target_network_update_freq": config.get('target_network_update_freq', 1),
                    "tau": config.get('tau', 0.005),
                    "gamma": config.get('gamma', 0.99),
                }
                
                trainer = sac.SACTrainer(config=trainer_config)
            
            else:
                raise ValueError(f"Unsupported distributed algorithm: {algorithm}")
            
            # Training loop
            num_iterations = config.get('num_iterations', 100)
            best_reward = -float('inf')
            
            for i in range(num_iterations):
                result = trainer.train()
                
                # Log progress
                if i % 10 == 0:
                    mean_reward = result['episode_reward_mean']
                    self.logger.info(f"Iteration {i}: Mean Reward = {mean_reward:.2f}")
                    
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        # Save checkpoint
                        checkpoint = trainer.save()
                        self.logger.info(f"New best model saved: {checkpoint}")
            
            # Final evaluation
            metrics = {
                'final_mean_reward': result['episode_reward_mean'],
                'final_episode_length': result['episode_len_mean'],
                'best_reward': best_reward,
                'total_iterations': num_iterations
            }
            
            self.logger.success(f"Distributed {algorithm} training completed")
            return trainer, metrics
            
        except Exception as e:
            self.logger.error(f"Error training distributed {algorithm}: {e}")
            raise
    
    def __del__(self):
        """Cleanup Ray"""
        if ray.is_initialized():
            ray.shutdown()

class RLModelManager:
    """RL Model Manager"""
    
    def __init__(self, config_path: str = "config/ai_models.yaml"):
        self.config = get_config(config_path)
        self.logger = get_logger()
        self.trainer = RLTrainer(config_path)
        self.distributed_trainer = RayRLTrainer(config_path)
        
    def train_all_rl_models(self, data: pd.DataFrame) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """Train all enabled RL models"""
        
        results = {}
        rl_config = self.config.get('reinforcement_learning', {})
        
        # Train standard RL models
        for algorithm, config in rl_config.items():
            if algorithm in ['ppo', 'sac', 'a2c', 'ddpg'] and config.get('enabled', False):
                try:
                    model, metrics = self.trainer.train_model(
                        algorithm.upper(), data, config
                    )
                    results[algorithm] = (model, metrics)
                    
                    # Log results
                    self.logger.model_performance(
                        algorithm, 
                        metrics.get('mean_reward', 0), 
                        metrics.get('max_reward', 0)
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {algorithm}: {e}")
                    continue
        
        # Train distributed RL models
        distributed_config = self.config.get('distributed_rl', {})
        if distributed_config.get('enabled', False):
            for algorithm, config in distributed_config.get('algorithms', {}).items():
                if config.get('enabled', False):
                    try:
                        model, metrics = self.distributed_trainer.train_distributed_model(
                            algorithm.replace('ray_', '').upper(), data, config
                        )
                        results[algorithm] = (model, metrics)
                        
                        # Log results
                        self.logger.model_performance(
                            algorithm, 
                            metrics.get('final_mean_reward', 0), 
                            metrics.get('best_reward', 0)
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to train distributed {algorithm}: {e}")
                        continue
        
        return results
    
    def get_best_rl_model(self, results: Dict[str, Tuple[Any, Dict[str, float]]]) -> Tuple[str, Any, Dict[str, float]]:
        """Get best performing RL model"""
        
        best_model = None
        best_metrics = None
        best_name = None
        best_score = -float('inf')
        
        for name, (model, metrics) in results.items():
            # Use mean reward as primary metric
            score = metrics.get('mean_reward', metrics.get('final_mean_reward', -float('inf')))
            
            if score > best_score:
                best_score = score
                best_model = model
                best_metrics = metrics
                best_name = name
        
        return best_name, best_model, best_metrics

# Available algorithms
AVAILABLE_ALGORITHMS = ['PPO', 'SAC', 'A2C', 'DDPG']
AVAILABLE_DISTRIBUTED_ALGORITHMS = ['RAY_PPO', 'RAY_SAC']