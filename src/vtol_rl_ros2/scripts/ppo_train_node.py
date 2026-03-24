#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import signal
import csv
import sys
from datetime import datetime

# 导入环境和模型
from envs.vtol_rl_env import VtolRlEnv
from models.actor_critic import ActorCriticLSTM

# --- 训练超参数 ---
LR = 1e-4                
GAMMA = 0.995             
EPS_CLIP = 0.2           
K_EPOCHS = 10            
UPDATE_TIMESTEP = 2048   
MAX_EPISODE_STEPS = 500  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCriticLSTM(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR) 
        self.policy_old = ActorCriticLSTM(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss() 

    def update(self, memory):
        states = torch.stack(memory['states']).to(device).detach()
        actions = torch.stack(memory['actions']).to(device).detach()
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']
        old_logprobs = torch.stack(memory['logprobs']).to(device).detach()

        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        states_seq = states.unsqueeze(0)
        actions_seq = actions.unsqueeze(0)

        for _ in range(K_EPOCHS):
            actor_hidden = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
            critic_hidden = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
            
            logprobs, state_values, dist_entropy = self.policy.evaluate(states_seq, actions_seq, actor_hidden, critic_hidden)
            logprobs = logprobs.squeeze(0)
            state_values = state_values.squeeze(0)

            advantages = returns - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = self.MseLoss(state_values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

def main(args=None):
    rclpy.init(args=args)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 实例化环境
    env = VtolRlEnv()
    logger = env.node.get_logger()
    
    # 注册 Ctrl+C 处理逻辑
    def signal_handler(sig, frame):
        logger.info("\n[PPO Node] Ctrl+C detected! Saving data and shutting down...")
        try:
            csv_file.close()
        except:
            pass
        env.node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Starting PPO Training Node (ROS 2 Native)...")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ppo_agent = PPO(state_dim, action_dim)
    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
    
    data_dir = os.path.join(os.path.expanduser('~'), 'px4_ros2_ws/data/vtol_rl')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_file_path = os.path.join(data_dir, f'training_log_{time_str}.csv')
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Episode', 'Total_Reward', 'Episode_Steps', 'Final_Altitude', 'Final_Speed'])

    time_step = 0
    i_episode = 0
    save_dir = os.path.join(data_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Entering main training loop...")
    
    # 使用 rclpy.ok() 替代 rospy.is_shutdown()
    while rclpy.ok():
        state = env.reset()
        actor_hidden = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
        current_ep_reward = 0
        ep_steps = 0 
        
        for t in range(MAX_EPISODE_STEPS):
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action, action_logprob, actor_hidden = ppo_agent.policy_old.act(state_tensor, actor_hidden)
            
            action_np = action.cpu().numpy()
            next_state, reward, done, _ = env.step(action_np)
            
            ep_steps += 1
            memory['states'].append(state_tensor)
            memory['actions'].append(action)
            memory['logprobs'].append(action_logprob)
            memory['rewards'].append(reward)
            memory['is_terminals'].append(done)
            
            state = next_state
            current_ep_reward += reward
            time_step += 1
            
            if time_step % UPDATE_TIMESTEP == 0:
                logger.info(f"Updating PPO Network at Timestep {time_step}...")
                ppo_agent.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
                
                model_name = f'ppo_vtol_fw_{time_str}_ep{i_episode}.pth'
                torch.save(ppo_agent.policy.state_dict(), os.path.join(save_dir, model_name))
                logger.info("Model saved.")

            if done or not rclpy.ok():
                break
                
        i_episode += 1
        logger.info(f"Episode: {i_episode} \t Reward: {current_ep_reward:.2f}")

        final_speed = state[0] 
        final_altitude = -env.local_pos.z # NED coordinate
        
        csv_writer.writerow([i_episode, current_ep_reward, ep_steps, final_altitude, final_speed])
        csv_file.flush() 

    csv_file.close()

if __name__ == '__main__':
    main()