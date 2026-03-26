#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorCriticTCN(nn.Module):
    """
    时间卷积网络 (TCN) 架构：
    使用 Conv1d 提取时间序列特征，适用于处理带有惯性延迟和传感器噪声的 VTOL 飞行任务。
    """
    def __init__(self, state_dim, action_dim, seq_len=8, hidden_dim=256): 
        super(ActorCriticTCN, self).__init__()
        self.seq_len = seq_len
        self.state_dim = state_dim
        
        # ================= [新增] 状态归一化层 =================
        # 它可以自动将各种巨大尺度的物理量拉回到安全范围，完美解决梯度消失
        self.state_norm = nn.LayerNorm(state_dim)
        # =======================================================

        # === Actor TCN 特征提取器 ===
        # 输入: (batch, state_dim, seq_len)
        self.actor_tcn = nn.Sequential(
            nn.Conv1d(in_channels=state_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        tcn_out_dim = 128 * seq_len

        # === Actor MLP 动作输出 ===
        self.actor_mlp = nn.Sequential(
            nn.Linear(tcn_out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # 限制输出物理极值在 [-1, 1] 之间
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) 

        # === Critic TCN 价值网络 ===
        self.critic_tcn = nn.Sequential(
            nn.Conv1d(in_channels=state_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.critic_mlp = nn.Sequential(
            nn.Linear(tcn_out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 权重正交初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward_actor(self, state_seq):
        # [新增] 先进行归一化
        normed_state = self.state_norm(state_seq)

        # 转换维度: (batch, seq_len, state_dim) -> (batch, state_dim, seq_len)
        x = state_seq.permute(0, 2, 1) 
        features = self.actor_tcn(x)
        return self.actor_mlp(features)

    def forward_critic(self, state_seq):
        # [新增] 先进行归一化
        normed_state = self.state_norm(state_seq)
        
        x = state_seq.permute(0, 2, 1)
        features = self.critic_tcn(x)
        return self.critic_mlp(features)

    def act(self, state_seq, hidden=None):
        # state_seq 此时是一个二维张量: (seq_len, state_dim)
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(0) # 扩充 batch 维度
            
        action_mean = self.forward_actor(state_seq)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample() 
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        action_clamped = torch.clamp(action, -1.0, 1.0)
        
        return action_clamped.squeeze(0).detach(), action_logprob.detach(), hidden

    def evaluate(self, state_seq, action_seq, actor_hidden=None, critic_hidden=None):
        # 兼容 ppo_train_node 中历史遗留的 unsqueeze(0) 嵌套
        if state_seq.dim() == 4:
            state_seq = state_seq.squeeze(0)
        if action_seq.dim() == 3:
            action_seq = action_seq.squeeze(0)

        action_mean = self.forward_actor(state_seq)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action_seq).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        state_values = self.forward_critic(state_seq).squeeze(-1)
        
        return action_logprobs.unsqueeze(0), state_values.unsqueeze(0), dist_entropy.unsqueeze(0)