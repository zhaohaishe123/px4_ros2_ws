#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time

from std_msgs.msg import Float32MultiArray, Bool
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, VehicleAngularVelocity, SensorCombined, VehicleStatus

def euler_from_quaternion(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(2.0 * (w * y - z * x))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw

class VtolRlEnv(gym.Env):
    def __init__(self):
        super(VtolRlEnv, self).__init__()
        
        # 动作: [Throttle, PitchRate, RollRate, YawRate] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # 状态: 17维 (完全复刻论文)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.node = rclpy.create_node('vtol_rl_env_core')
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.cmd_pub = self.node.create_publisher(Float32MultiArray, '/rl/actuator_cmds', 10)

        self.local_pos, self.att, self.ang_vel, self.imu, self.status = VehicleLocalPosition(), VehicleAttitude(), VehicleAngularVelocity(), SensorCombined(), VehicleStatus()

        self.node.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', lambda msg: setattr(self, 'local_pos', msg), qos_profile)
        self.node.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', lambda msg: setattr(self, 'att', msg), qos_profile)
        self.node.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity', lambda msg: setattr(self, 'ang_vel', msg), qos_profile)
        self.node.create_subscription(SensorCombined, '/fmu/out/sensor_combined', lambda msg: setattr(self, 'imu', msg), qos_profile)
        self.node.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', lambda msg: setattr(self, 'status', msg), qos_profile)

        self.target_speed = 20.0
        self.target_roll = 0.0
        self.target_pitch = 0.05  # NED坐标系中，正pitch为抬头
        self.target_altitude = 50.0 
        
        self.prev_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.prev_pqr = np.zeros(3)

        # 等待系统就绪机制
        self.node.get_logger().info("Waiting for PX4 DDS connection and Offboard take-off...")
        while rclpy.ok():
            self._spin_once()
            # 必须满足三个条件：1.收到状态 2.处于解锁状态 3.高度大于10米(说明C++节点已经把它带飞起来了)
            if hasattr(self, 'status') and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                if hasattr(self, 'local_pos') and -self.local_pos.z > 10.0:
                    self.node.get_logger().info("PX4 is Airborne and Ready! Starting RL...")
                    break
            time.sleep(0.5)

        # 专门用于接收 C++ 节点状态位的变量
        self.is_offboard_ready = False
        self.node.create_subscription(Bool, '/rl/training_ready', self._ready_cb, qos_profile)

        self.target_speed = 20.0
        self.target_roll = 0.0
        self.target_pitch = 0.05  
        self.target_altitude = 50.0 
        
        self.prev_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.prev_pqr = np.zeros(3)

        # 基于状态位的就绪等待机制
        self.node.get_logger().info("Waiting for C++ Offboard Node to complete take-off and transition...")
        while rclpy.ok():
            self._spin_once()
            
            # 只要收到 C++ 节点的 True 信号，直接跳出死等，开始强化学习
            if self.is_offboard_ready:
                self.node.get_logger().info("Received READY signal from C++ node. Starting RL Training!")
                break
                
            time.sleep(0.5)


    # 状态位回调函数
    def _ready_cb(self, msg):
        self.is_offboard_ready = msg.data

    def _spin_once(self): 
        rclpy.spin_once(self.node, timeout_sec=0.01)

    def step(self, action):
        # ================= 1. 获取当前真实的姿态角 =================
        w, x, y, z = self.att.q[0], self.att.q[1], self.att.q[2], self.att.q[3]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(2.0 * (w * y - z * x))

        # ================= 2. 保守的“限舵”操作 (事前限制) =================
        # 即使 C++ 会做物理映射，我们也必须在这里将 RL 的输出 [-1, 1] 压制在更小的安全范围内。
        # 绝不能让网络在初期有机会输出满舵导致飞机不可逆翻滚。
        safe_action = np.copy(action)
        safe_action[0] = np.clip(action[0], -0.5, 0.5)  # 限制油门剧烈波动
        safe_action[1] = np.clip(action[1], -0.4, 0.4)  # 限制俯仰幅度 (防止猛扎机头)
        safe_action[2] = np.clip(action[2], -0.5, 0.5)  # 限制横滚幅度 (防止翻滚)
        safe_action[3] = np.clip(action[3], -0.2, 0.2)  # 限制偏航幅度

        # ================= 3. 强硬的姿态电子围栏 (事后接管) =================
        # 收紧安全边界：超过 23度(Pitch) 或 34度(Roll) 直接剥夺 RL 控制权
        SAFE_PITCH = 0.4  
        SAFE_ROLL  = 0.6  

        # 俯仰越界：强制拉回
        if pitch > SAFE_PITCH:     
            safe_action[1] = -0.8  # 强制压机头
        elif pitch < -SAFE_PITCH:
            safe_action[1] = 0.8   # 强制拉机头
            safe_action[0] = 0.8   # 拉机头时强制增加油门，防止失速坠落

        # 横滚越界：防止倾翻的核心
        if roll > SAFE_ROLL:       
            safe_action[2] = -1.0  # 直接打满反舵：强制强力左滚转改平
        elif roll < -SAFE_ROLL:  
            safe_action[2] = 1.0   # 直接打满反舵：强制强力右滚转改平

        # ================= 4. 动作平滑与下发 =================
        # 使用经过安全过滤的 safe_action 进行平滑计算和下发
        tau = 0.4
        smoothed_action = (1.0 - tau) * self.prev_action + tau * safe_action
        smoothed_action = np.clip(smoothed_action, -1.0, 1.0)

        # 发送给 C++ 节点
        msg = Float32MultiArray()
        msg.data = smoothed_action.tolist()
        self.cmd_pub.publish(msg)

        time.sleep(0.05) # 20Hz
        self._spin_once()
        
        obs = self._get_obs()
        
        # ================= 5. 奖励评估 =================
        # 【关键】：计算奖励时，必须传入网络最初始、最真实的 `action`，而不是 `smoothed_action`！
        # 这样网络才会因为输出满舵而在 _compute_reward_and_done 的平滑性惩罚中被严重扣分，
        # 从而学到“我乱打杆虽然没死（因为有保护），但是分数很惨”。
        reward, done = self._compute_reward_and_done(obs, action)

        self.prev_action = np.copy(smoothed_action)
        self.prev_pqr = obs[4:7]

        return obs, reward, done, {}
    
    def reset(self):
        """优化版超限恢复逻辑：防死锁与失速改出"""
        self.prev_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.prev_pqr = np.zeros(3)

        self._spin_once()
        obs = self._get_obs()
        current_alt = -self.local_pos.z 
        speed = obs[0] * 10.0 + 20.0 # 还原实际速度

        # 如果已经很完美，直接开始
        if 48.0 < current_alt < 52.0 and 17.0 < speed < 22.0:
            return obs

        self.node.get_logger().info(f"Out of bounds! Alt: {current_alt:.1f}m, Speed: {speed:.1f}m/s. Initiating active recovery...")
        
        start_time = time.time()
        
        while rclpy.ok():
            self._spin_once()
            
            # ================= [终极修复：不死节点机制] =================
            if time.time() - start_time > 30.0 or -self.local_pos.z < 5.0:
                self.node.get_logger().warn("⚠️ Recovery failed (Crashed)! Waiting for C++ Auto-Respawn...")
                
                # 1. 强行将就绪标志位置为 False，切断控制权
                self.is_offboard_ready = False
                
                # 2. 挂起 Python 节点，死等 C++ 重新起飞并发布 READY
                while rclpy.ok():
                    self._spin_once()
                    if self.is_offboard_ready:
                        self.node.get_logger().info("PX4 is Airborne again! Resuming recovery...")
                        break
                    time.sleep(0.5)
                
                # 3. 飞机重新起飞后，重置超时计时器，继续执行恢复逼近逻辑
                start_time = time.time()
                continue
            # =========================================================

            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                time.sleep(1.0)
                continue
            
            check_alt = -self.local_pos.z
            check_speed = np.sqrt(self.local_pos.vx**2 + self.local_pos.vy**2 + self.local_pos.vz**2)
            roll, pitch, yaw = euler_from_quaternion(self.att.q)

            if 48.0 < check_alt < 52.0 and 18.0 < check_speed < 22.0 and abs(roll) < 0.15:
                self.node.get_logger().info("Recovery successful! Starting new episode...")
                break

            # --- 智能分段 P 控制器 ---
            # [新增] 失速改出逻辑：如果速度太低，不要管高度，强行推机头加速！
            if check_speed < 12.0:
                cmd_throttle = 1.0       # 满油门
                cmd_pitch_rate = -0.5    # 强制压机头俯冲增速
                cmd_roll_rate = np.clip(-roll * 2.0, -1.0, 1.0) # 尽量改平横滚
            else:
                # 速度正常后，再去追高度
                alt_err = 50.0 - check_alt
                speed_err = 20.0 - check_speed
                
                cmd_pitch_rate = np.clip(alt_err * 0.1, -0.6, 0.6) 
                cmd_roll_rate = np.clip(-roll * 2.0, -1.0, 1.0)
                cmd_throttle = np.clip(0.6 + (speed_err * 0.1) + (alt_err * 0.02), 0.1, 1.0)
            
            rec_action = np.array([cmd_throttle, cmd_pitch_rate, cmd_roll_rate, 0.0], dtype=np.float32)
            msg = Float32MultiArray()
            msg.data = rec_action.tolist()
            self.cmd_pub.publish(msg)
            
            time.sleep(0.05)
            
        return self._get_obs()

    def _get_obs(self):
        roll, pitch, yaw = euler_from_quaternion(self.att.q)
        delta_yaw = np.arctan2(np.sin(yaw - self.prev_yaw), np.cos(yaw - self.prev_yaw))
        self.prev_yaw = yaw

        p, q_rate, r = self.ang_vel.xyz[0], self.ang_vel.xyz[1], self.ang_vel.xyz[2]
        vx, vy, vz = self.local_pos.vx, self.local_pos.vy, self.local_pos.vz
        Va = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # 旋转矩阵 (四元数转旋转矩阵)
        q_w, q_x, q_y, q_z = self.att.q[0], self.att.q[1], self.att.q[2], self.att.q[3]
        rot_mat = np.array([
            [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_w*q_z), 2*(q_x*q_z + q_w*q_y)],
            [2*(q_x*q_y + q_w*q_z), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_w*q_x)],
            [2*(q_x*q_z - q_w*q_y), 2*(q_y*q_z + q_w*q_x), 1 - 2*(q_x**2 + q_y**2)]
        ])
        body_vel = np.dot(rot_mat.T, np.array([vx, vy, vz]))
        u, v, w = body_vel[0], body_vel[1], body_vel[2]

        ax, ay, az = self.imu.accelerometer_m_s2[0], self.imu.accelerometer_m_s2[1], self.imu.accelerometer_m_s2[2]
        current_alt = -self.local_pos.z
        alt_error = self.target_altitude - current_alt

        # 完全复刻 ROS1 中近似归一化的组装
        obs = np.array([
            (Va - 20.0)/10.0, roll/1.57, pitch/1.0, delta_yaw/3.14,
            p/2.0, q_rate/2.0, r/2.0,
            (u-20.0)/10.0, v/5.0, w/5.0,
            ax/10.0, ay/10.0, (az-9.8)/10.0, # 抵消重力
            (self.target_speed-20.0)/10.0, self.target_roll, self.target_pitch,
            alt_error/20.0
        ], dtype=np.float32)

        return obs
    def _compute_reward_and_done(self, obs, action):
        """优化版奖励函数：根治死亡俯冲，彻底阻断自杀策略"""
        # 解包状态
        Va = obs[0] * 10.0 + 20.0
        roll = obs[1] * 1.57
        pitch = obs[2] * 1.0
        delta_yaw = obs[3] * 3.14
        p, q_rate, r = obs[4] * 2.0, obs[5] * 2.0, obs[6] * 2.0
        
        # 垂直速度 (NED坐标系：vz > 0 表示正在下降)
        vz = self.local_pos.vz 
        current_alt = -self.local_pos.z
        
        # 计算误差
        Va_e = self.target_speed - Va
        pitch_e = self.target_pitch - pitch
        roll_e = self.target_roll - roll
        alt_e = self.target_altitude - current_alt

        # 动作变化率 (惩罚抖动)
        action_rate = action - self.prev_action 
        
        # ================= [死亡俯冲专项惩罚] =================
        dive_penalty = 0.0
        if vz > 3.0 and pitch < -0.2: 
            dive_penalty = 20.0 * vz 
        
        # --- 1. 基础飞行成本 (Longitudinal) ---
        long_cost = (2.0 * abs(Va_e) + 15.0 * abs(pitch_e) + 3.0 * abs(alt_e) + 2.0 * abs(vz))
        
        # --- 2. 基础飞行成本 (Lateral) ---
        lat_cost = (15.0 * abs(roll_e) + 10.0 * abs(delta_yaw))

        # --- 3. 稳定性与平滑性惩罚 ---
        smooth_penalty = 5.0 * np.sum(np.square(action_rate)) 
        oscillation_penalty = 0.5 * (abs(p) + abs(q_rate) + abs(r))

        # --- 4. 计算总奖励 ---
        step_reward = 40.0 - (long_cost + lat_cost + smooth_penalty + oscillation_penalty + dive_penalty)
        
        # ================= [核心修复：截断负分底线] =================
        # 允许出现负分来惩罚颠簸，但单步最多扣 -20 分。
        # 这样即使活满 500 步，最差总分也就是 -10000 分左右。
        reward = np.clip(step_reward, -20.0, 50.0)

        # --- 5. 终止条件与惩罚 ---
        done = False
        
        # ================= [核心修复：深渊级坠毁惩罚] =================
        # 坠毁惩罚必须远大于最差存活总分（-10000），让它永远不敢自杀
        terminal_penalty = -30000.0 

        # 维持之前修改的 30.0 米生命线
        if current_alt < 30.0 or current_alt > 120.0: 
            done = True
            reward += terminal_penalty
        elif abs(roll) > 1.2 or abs(pitch) > 0.8: 
            done = True
            reward += terminal_penalty
        elif Va < 10.0: 
            done = True
            reward += terminal_penalty

        return float(reward), done