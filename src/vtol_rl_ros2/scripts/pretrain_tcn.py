#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 导入你写好的 TCN 网络
from models.actor_critic import ActorCriticTCN

def create_sequences(states, actions, seq_len=8):
    """
    为 TCN 网络构建滑动窗口序列数据。
    输入 states: (N, 17), actions: (N, 4)
    输出 X: (N-seq_len+1, 8, 17), Y: (N-seq_len+1, 4)
    """
    X, Y = [], []
    for i in range(len(states) - seq_len + 1):
        X.append(states[i : i + seq_len])
        # 标签是当前窗口最后一帧对应的专家动作
        Y.append(actions[i + seq_len - 1]) 
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def main():
    print("="*50)
    print("  BEHAVIORAL CLONING (PRE-TRAINING) STARTED  ")
    print("="*50)

    # ================= 1. 加载数据 =================
    data_dir = os.path.expanduser('~/px4_ros2_ws/data/vtol_rl/expert_data')
    states_path = os.path.join(data_dir, 'expert_states.npy')
    actions_path = os.path.join(data_dir, 'expert_actions.npy')

    if not os.path.exists(states_path) or not os.path.exists(actions_path):
        print(f"[ERROR] 找不到数据文件！请确保文件存在于 {data_dir}")
        return

    print("Loading expert data...")
    raw_states = np.load(states_path)
    raw_actions = np.load(actions_path)
    
    print(f"Raw States: {raw_states.shape}, Raw Actions: {raw_actions.shape}")

    # ================= 2. 构建时序数据集 =================
    SEQ_LEN = 8
    X_np, Y_np = create_sequences(raw_states, raw_actions, seq_len=SEQ_LEN)
    print(f"Sequenced X: {X_np.shape}, Sequenced Y: {Y_np.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 转换为 Tensor
    X_tensor = torch.FloatTensor(X_np).to(device)
    Y_tensor = torch.FloatTensor(Y_np).to(device)

    dataset = TensorDataset(X_tensor, Y_tensor)
    # 使用较大的 Batch Size 加速训练
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True) 

    # ================= 3. 初始化 TCN 网络 =================
    state_dim = 17
    action_dim = 4
    policy = ActorCriticTCN(state_dim, action_dim, seq_len=SEQ_LEN).to(device)
    
    # 我们只预训练 Actor (动作输出层)，Critic (价值评估层) 留给 PPO 自己去学
    optimizer = optim.Adam(policy.actor_mlp.parameters(), lr=1e-3)
    # 添加 TCN 特征提取器的参数
    optimizer.add_param_group({'params': policy.actor_tcn.parameters(), 'lr': 1e-3})
    
    loss_fn = nn.MSELoss()

    # ================= 4. 开始监督学习 =================
    EPOCHS = 100
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # TCN 的 forward_actor 会输出连续的 action
            predicted_actions = policy.forward_actor(batch_x)
            
            # 计算均方误差 (MSE)
            loss = loss_fn(predicted_actions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | MSE Loss: {avg_loss:.6f}")

    # ================= 5. 保存预训练权重 =================
    save_dir = os.path.expanduser('~/px4_ros2_ws/data/vtol_rl/saved_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pretrained_tcn.pth')
    
    torch.save(policy.state_dict(), save_path)
    print(f"\n[SUCCESS] Pre-trained model saved to: {save_path}")
    print("="*50)

if __name__ == '__main__':
    main()