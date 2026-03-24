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

# 导入 TensorBoard (如果提示没有模块，请终端执行 pip install tensorboard)
from torch.utils.tensorboard import SummaryWriter

from envs.vtol_rl_env import VtolRlEnv
from models.actor_critic import ActorCriticLSTM

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
        
        total_actor_loss = 0
        total_critic_loss = 0

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
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 返回平均 loss 供 TensorBoard 记录
        return total_actor_loss / K_EPOCHS, total_critic_loss / K_EPOCHS

def main(args=None):
    rclpy.init(args=args)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    env = VtolRlEnv()
    logger = env.node.get_logger()
    
    # === [新增] 初始化日志记录路径 ===
    base_dir = os.path.join(os.path.expanduser('~'), 'px4_ros2_ws/data/vtol_rl')
    tb_dir = os.path.join(base_dir, f'tensorboard_logs/{time_str}')
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 1. TensorBoard Writer
    writer = SummaryWriter(log_dir=tb_dir)
    
    # 2. CSV Writer
    csv_file_path = os.path.join(base_dir, f'training_log_{time_str}.csv')
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Episode', 'Total_Reward', 'Episode_Steps', 'Final_Altitude', 'Final_Speed'])
    
    def signal_handler(sig, frame):
        logger.info("\n[PPO Node] Ctrl+C detected! Saving data and shutting down...")
        try:
            writer.close()
            csv_file.close()
        except: pass
        env.node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim)
    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}

    time_step = 0
    i_episode = 0

    logger.info(f"Training started. TensorBoard logs saved to: {tb_dir}")
    
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
            
            # 网络更新
            if time_step % UPDATE_TIMESTEP == 0:
                logger.info(f"Updating PPO Network at Timestep {time_step}...")
                a_loss, c_loss = ppo_agent.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
                
                # 记录网络损失到 TensorBoard
                writer.add_scalar('Loss/Actor', a_loss, time_step)
                writer.add_scalar('Loss/Critic', c_loss, time_step)
                
                model_name = f'ppo_vtol_fw_{time_str}.pth'
                torch.save(ppo_agent.policy.state_dict(), os.path.join(save_dir, model_name))

            if done or not rclpy.ok():
                break
                
        i_episode += 1
        logger.info(f"Episode: {i_episode} \t Reward: {current_ep_reward:.2f} \t Steps: {ep_steps}")

        final_speed = state[0] 
        final_altitude = -env.local_pos.z 
        
        # 记录每轮数据到 CSV 和 TensorBoard
        csv_writer.writerow([i_episode, current_ep_reward, ep_steps, final_altitude, final_speed])
        csv_file.flush() 
        
        writer.add_scalar('Episode/Total_Reward', current_ep_reward, i_episode)
        writer.add_scalar('Episode/Steps_Survived', ep_steps, i_episode)
        writer.add_scalar('Episode/Final_Altitude', final_altitude, i_episode)
        writer.add_scalar('Episode/Final_Speed', final_speed, i_episode)

    csv_file.close()
    writer.close()

if __name__ == '__main__':
    main()