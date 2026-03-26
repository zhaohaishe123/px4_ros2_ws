#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import numpy as np
import time
import math
from envs.vtol_rl_env import VtolRlEnv

def get_expert_action(obs, current_alt, roll, pitch):
    """
    专家 PID 控制器：计算完美的物理动作，并将其反向映射为 RL 网络的 [-1, 1] 动作
    """
    speed = obs[0] * 10.0 + 20.0
    
    # 1. PID 计算期望的物理控制量
    alt_err = 50.0 - current_alt
    speed_err = 20.0 - speed
    
    target_pitch_rate = np.clip(alt_err * 0.15, -0.4, 0.4) 
    target_roll_rate = np.clip(-roll * 2.0, -0.6, 0.6)
    target_throttle = np.clip(0.6 + (speed_err * 0.05) + (alt_err * 0.01), 0.3, 0.8)
    
    # 2. 反向映射 (Inverse Mapping)：将物理量转换为网络应该输出的 [-1, 1] 动作
    # 对应 vtol_rl_env.py 中 step() 函数的物理映射范围
    
    # 油门: [0.3, 0.8] -> [-1.0, 1.0]
    action_0 = np.interp(target_throttle, [0.3, 0.8], [-1.0, 1.0])
    # 俯仰: 映射系数 0.5
    action_1 = target_pitch_rate / 0.5
    # 横滚: 映射系数 0.8
    action_2 = target_roll_rate / 0.8
    # 偏航: 暂时不控，设为 0
    action_3 = 0.0
    
    action = np.array([action_0, action_1, action_2, action_3], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


def main():
    rclpy.init()
    
    print("="*50)
    print("  EXPERT DATA COLLECTION STARTED  ")
    print("="*50)
    
    env = VtolRlEnv() # 初始化环境 (内部会自动等待 READY 信号)
    
    expert_states = []
    expert_actions = []
    
    TARGET_STEPS = 20000  # 收集 2万 步数据 (大约相当于 40 个完美 Episode)
    steps_collected = 0
    
    obs = env.reset()
    
    while rclpy.ok() and steps_collected < TARGET_STEPS:
        # 获取真实姿态
        w, x, y, z = env.att.q[0], env.att.q[1], env.att.q[2], env.att.q[3]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(2.0 * (w * y - z * x))
        current_alt = -env.local_pos.z
        
        # 计算专家动作
        expert_action = get_expert_action(obs, current_alt, roll, pitch)
        
        # 保存数据
        expert_states.append(obs)
        expert_actions.append(expert_action)
        
        # 执行动作
        next_obs, _, done, _ = env.step(expert_action)
        
        steps_collected += 1
        
        if steps_collected % 500 == 0:
            print(f"[Collecting] Progress: {steps_collected}/{TARGET_STEPS} | Alt: {current_alt:.1f}m | Speed: {obs[0]*10+20:.1f}m/s")
            
        if done:
            obs = env.reset()
        else:
            obs = next_obs

    # ================= 保存数据集 =================
    print("\n[INFO] Saving Dataset to disk...")
    states_np = np.array(expert_states, dtype=np.float32)
    actions_np = np.array(expert_actions, dtype=np.float32)
    
    # 保存到 data 目录下
    import os
    save_dir = os.path.expanduser('~/px4_ros2_ws/data/vtol_rl/expert_data')
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'expert_states.npy'), states_np)
    np.save(os.path.join(save_dir, 'expert_actions.npy'), actions_np)
    
    print(f"[SUCCESS] Saved {len(states_np)} transitions to {save_dir}")
    print("="*50)
    
    env.node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()