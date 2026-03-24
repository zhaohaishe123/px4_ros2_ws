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

    def _spin_once(self): rclpy.spin_once(self.node, timeout_sec=0.01)

    def step(self, action):
        # ================= [新增] 飞行姿态安全保护罩 (Attitude Safety Guard) =================
        # 1. 获取当前真实的 Roll 和 Pitch 姿态角
        w, x, y, z = self.att.q[0], self.att.q[1], self.att.q[2], self.att.q[3]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(2.0 * (w * y - z * x))

        # 2. 定义安全边界 (弧度)
        SAFE_PITCH = 0.5  # 约 28 度 (防止严重抬机头/大角度俯冲)
        SAFE_ROLL  = 0.8  # 约 45 度 (防止过度倾斜导致掉高甚至翻滚)

        # 3. 拦截危险指令：如果已经在危险边缘，且 RL 动作还要继续加剧倾斜，则强行切断该方向指令
        if (pitch > SAFE_PITCH and action[1] > 0.0) or (pitch < -SAFE_PITCH and action[1] < 0.0):
            action[1] = 0.0
        if (roll > SAFE_ROLL and action[2] > 0.0) or (roll < -SAFE_ROLL and action[2] < 0.0):
            action[2] = 0.0

        # 4. 强力改平辅助：如果由于惯性越过了极限安全线，脱离 RL 控制，直接施加反向恢复力矩
        if pitch > SAFE_PITCH + 0.1:
            action[1] = -0.6  # 强制压机头
        elif pitch < -(SAFE_PITCH + 0.1):
            action[1] = 0.6   # 强制拉机头

        if roll > SAFE_ROLL + 0.1:
            action[2] = -0.8  # 强制反向横滚改平
        elif roll < -(SAFE_ROLL + 0.1):
            action[2] = 0.8
        # =================================================================================

        # 动作平滑 (融入了安全修正后的 action)
        tau = 0.4
        smoothed_action = (1.0 - tau) * self.prev_action + tau * action
        smoothed_action = np.clip(smoothed_action, -1.0, 1.0)

        # 发送给 C++ 节点 (C++中会映射为具体的物理边界)
        msg = Float32MultiArray()
        msg.data = smoothed_action.tolist()
        self.cmd_pub.publish(msg)

        time.sleep(0.05) # 20Hz
        self._spin_once()
        
        obs = self._get_obs()
        reward, done = self._compute_reward_and_done(obs, smoothed_action)

        self.prev_action = np.copy(smoothed_action)
        self.prev_pqr = obs[4:7]

        return obs, reward, done, {}

    def reset(self):
        """超限恢复逻辑：使用角速度P控制器"""
        self.prev_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.prev_pqr = np.zeros(3)

        self._spin_once()
        obs = self._get_obs()
        current_alt = -self.local_pos.z # NED转为正高度
        speed = obs[0]

        if 48.0 < current_alt < 52.0 and 17.0 < speed < 22.0:
            return obs

        self.node.get_logger().info(f"Out of bounds! Alt: {current_alt:.1f}m, Speed: {speed:.1f}m/s. Initiating active Body Rate recovery...")
        
        while rclpy.ok():
            self._spin_once()
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                time.sleep(1.0)
                continue
            
            check_alt = -self.local_pos.z
            check_speed = np.sqrt(self.local_pos.vx**2 + self.local_pos.vy**2 + self.local_pos.vz**2)
            roll, pitch, yaw = euler_from_quaternion(self.att.q)

            if 48.5 < check_alt < 51.5 and 18.5 < check_speed < 21.5 and abs(roll) < 0.1:
                self.node.get_logger().info("Recovery successful! Starting new episode...")
                break

            # --- 角速度 P 控制器 ---
            alt_err = 50.0 - check_alt
            speed_err = 20.0 - check_speed
            
            # 1. Pitch Rate (NED: 正值抬头)
            cmd_pitch_rate = np.clip(alt_err * 0.1, -0.6, 0.6) 
            # 2. Roll Rate (反向打杆以改平)
            cmd_roll_rate = np.clip(-roll * 2.0, -1.0, 1.0)
            # 3. Throttle (映射到 [-1, 1])
            cmd_throttle = np.clip(0.0 + (speed_err * 0.15) + (alt_err * 0.02), -0.5, 1.0)
            
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
        """优化版奖励函数：强化平滑性约束与震荡抑制"""
        # 解包状态 (按照 _get_obs 的组装顺序)
        # Va[0], roll[1], pitch[2], delta_yaw[3], p[4], q[5], r[6]
        Va = obs[0] * 10.0 + 20.0
        roll = obs[1] * 1.57
        pitch = obs[2] * 1.0
        delta_yaw = obs[3] * 3.14
        p, q_rate, r = obs[4] * 2.0, obs[5] * 2.0, obs[6] * 2.0
        
        # 垂直速度 (从 local_pos 获取，用于抑制震荡)
        vz = self.local_pos.vz 
        current_alt = -self.local_pos.z
        
        # 计算误差
        Va_e = self.target_speed - Va
        pitch_e = self.target_pitch - pitch
        roll_e = self.target_roll - roll
        alt_e = self.target_altitude - current_alt

        # 动作变化率 (惩罚抖动)
        action_rate = action - self.prev_action 

        # --- 1. 基础飞行成本 (Longitudinal) ---
        # 增加对 alt_e 的权重，并引入 vz 惩罚以抑制高度震荡
        long_cost = (3.0 * abs(Va_e) + 10.0 * abs(pitch_e) + 2.0 * abs(alt_e) + 1.5 * abs(vz))
        
        # --- 2. 基础飞行成本 (Lateral) ---
        lat_cost = (10.0 * abs(roll_e) + 15.0 * abs(delta_yaw))

        # --- 3. 稳定性与平滑性惩罚 (关键修改) ---
        # 去掉原有的 clip，或者大幅提高上限，让网络“感受到”乱打杆的痛苦
        smooth_penalty = 5.0 * np.sum(np.square(action_rate)) 
        
        # 惩罚过大的角速度绝对值 (强制平稳)
        oscillation_penalty = 0.5 * (abs(p) + abs(q_rate) + abs(r))

        # --- 4. 计算总奖励 ---
        # 基础生存奖励设定为 35，减去成本
        step_reward = 35.0 - (long_cost + lat_cost + smooth_penalty + oscillation_penalty)
        
        # 保证每步至少有微小正奖励，鼓励存活
        reward = max(0.0, step_reward)

        # --- 5. 终止条件与惩罚 ---
        done = False
        terminal_penalty = -2000.0 # 提高坠毁惩罚幅度

        if current_alt < 20.0 or current_alt > 120.0: # 稍微放宽高度边界方便探索
            done = True
            reward += terminal_penalty
        elif abs(roll) > 1.2 or abs(pitch) > 0.8: 
            done = True
            reward += terminal_penalty
        elif Va < 10.0: # 进一步防止失速
            done = True
            reward += terminal_penalty

        return float(reward), done