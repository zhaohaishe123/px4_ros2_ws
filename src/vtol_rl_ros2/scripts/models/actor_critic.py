#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorCriticLSTM(nn.Module):
    """
    高度优化版架构：为解决原 LSTM 在跨 Episode 时的状态污染问题，
    底层架构替换为鲁棒性更强且收敛更快的 MLP（多层感知机）。
    同时保留了原有的类名和隐藏状态传参接口，以完全兼容 PPO 训练节点。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256): 
        super(ActorCriticLSTM, self).__init__()
        
        # === Actor 网络 (策略网络) ===
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # 限制输出物理极值在 [-1, 1] 之间
        )
        
        # 可训练的标准差参数，独立于状态特征之外 (PPO 连续控制标准做法)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) 

        # === Critic 网络 (价值网络) ===
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 权重正交初始化 (大幅提升 PPO 在连续控制任务中的初始稳定性)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def act(self, state, actor_hidden):
        """用于在 ROS 环境中单步生成动作"""
        # 兼容 PPO 脚本中对 state 的展平处理
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        action_mean = self.actor(state)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample() 
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        action_clamped = torch.clamp(action, -1.0, 1.0)
        
        # 原样返回 actor_hidden (占位符) 以兼容外部脚本
        return action_clamped.squeeze(0).detach(), action_logprob.detach(), actor_hidden

    def evaluate(self, state_seq, action_seq, actor_hidden, critic_hidden):
        """用于 PPO 算法更新计算梯度和优势函数"""
        # PPO 传入的 state_seq 形状可能是 (1, batch_size, state_dim)
        # 我们将其展平为 (batch_size, state_dim) 喂给 MLP
        state_flat = state_seq.squeeze(0)
        action_flat = action_seq.squeeze(0)

        # Actor 评估
        action_mean = self.actor(state_flat)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action_flat).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        # Critic 评估
        state_values = self.critic(state_flat).squeeze(-1)
        
        # 将结果重新增加一维以兼容 PPO 脚本中预期的拆包格式 (squeeze)
        return action_logprobs.unsqueeze(0), state_values.unsqueeze(0), dist_entropy.unsqueeze(0)