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

# 导入 ROS 2 消息与 PX4 原生消息
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
        
        # 动作空间 [Throttle, Pitch_Rate, Roll_Rate, Yaw_Rate] ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # 状态空间 17 维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.node = rclpy.create_node('vtol_rl_env_core')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.cmd_pub = self.node.create_publisher(Float32MultiArray, '/rl/actuator_cmds', 10)

        # 状态变量缓存
        self.local_pos = VehicleLocalPosition()
        self.att = VehicleAttitude()
        self.ang_vel = VehicleAngularVelocity()
        self.imu = SensorCombined()
        self.status = VehicleStatus()

        # 订阅者
        self.node.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self._pos_cb, qos_profile)
        self.node.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._att_cb, qos_profile)
        self.node.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity', self._ang_vel_cb, qos_profile)
        self.node.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self._imu_cb, qos_profile)
        self.node.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self._status_cb, qos_profile)

        self.target_speed = 20.0
        self.target_roll = 0.0
        self.target_pitch = -0.05
        self.target_altitude = 50.0 
        
        self.prev_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.prev_pqr = np.zeros(3)

    def _pos_cb(self, msg): self.local_pos = msg
    def _att_cb(self, msg): self.att = msg
    def _ang_vel_cb(self, msg): self.ang_vel = msg
    def _imu_cb(self, msg): self.imu = msg
    def _status_cb(self, msg): self.status = msg

    def _spin_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)

    def step(self, action):
        tau = 0.4
        smoothed_action = (1.0 - tau) * self.prev_action + tau * action
        smoothed_action = np.clip(smoothed_action, -1.0, 1.0)

        msg = Float32MultiArray()
        msg.data = smoothed_action.tolist()
        self.cmd_pub.publish(msg)

        time.sleep(0.05) # 模拟 20Hz
        self._spin_once()
        
        obs = self._get_obs()
        reward, done = self._compute_reward_and_done(obs, smoothed_action)

        self.prev_action = np.copy(smoothed_action)
        self.prev_pqr = obs[4:7]

        return obs, reward, done, {}

    def reset(self):
        self.prev_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.prev_pqr = np.zeros(3)

        self._spin_once()
        obs = self._get_obs()
        current_alt = -self.local_pos.z 
        speed = obs[0]

        if 48.0 < current_alt < 52.0 and 17.0 < speed < 22.0:
            return obs

        self.node.get_logger().info("Initiating active recovery (Using Body Rate Control)...")
        
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

            # --- 角速度 P 控制器恢复逻辑 ---
            alt_err = 50.0 - check_alt
            speed_err = 20.0 - check_speed
            
            # 1. 速度转推力 [-1, 1] 映射
            cmd_throttle = np.clip(0.0 + (speed_err * 0.1) + (alt_err * 0.02), -1.0, 1.0)
            # 2. 高度差转 Pitch Rate (q)
            cmd_pitch_rate = np.clip(alt_err * 0.1, -1.0, 1.0) 
            # 3. 倾斜角转 Roll Rate (p) 以改平飞机
            cmd_roll_rate = np.clip(-roll * 2.0, -1.0, 1.0)
            
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
        u, v, w = vx, vy, vz 

        ax, ay, az = self.imu.accelerometer_m_s2[0], self.imu.accelerometer_m_s2[1], self.imu.accelerometer_m_s2[2]
        alt_error = self.target_altitude - (-self.local_pos.z)

        obs = np.array([Va, roll, pitch, delta_yaw, p, q_rate, r, u, v, w, ax, ay, az, 
                        self.target_speed, self.target_roll, self.target_pitch, alt_error], dtype=np.float32)
        return obs

    def _compute_reward_and_done(self, obs, action):
        Va, roll, pitch, delta_yaw = obs[0], obs[1], obs[2], obs[3]
        p, q_rate, r = obs[4], obs[5], obs[6]
        current_alt = -self.local_pos.z
        
        Va_e = self.target_speed - Va
        pitch_e = self.target_pitch - pitch
        roll_e = self.target_roll - roll
        alt_e = self.target_altitude - current_alt

        action_rate = action - self.prev_action 

        long_cost = (1.0 * abs(Va_e) + 5.0 * abs(pitch_e) + 0.5 * abs(q_rate) + 1.0 * abs(alt_e))
        long_cost += np.clip(0.5 * abs(action_rate[0]), 0.0, 0.5) 
        long_cost += np.clip(1.0 * abs(action_rate[1]), 0.0, 1.0) 

        lat_cost = (8.0 * abs(roll_e) + 0.5 * abs(p) + 0.5 * abs(r) + 15.0 * abs(delta_yaw))
        lat_cost += np.clip(1.0 * abs(action_rate[2]), 0.0, 1.0) 
        lat_cost += np.clip(1.0 * abs(action_rate[3]), 0.0, 1.0) 

        step_reward = 30.0 - (long_cost + lat_cost)
        reward = max(5.0, step_reward)

        done = False
        terminal_penalty = -1500.0

        if current_alt < 30.0 or current_alt > 100.0:
            done = True
            reward += terminal_penalty
        elif abs(roll) > 1.57 or abs(pitch) > 1.0: 
            done = True
            reward += terminal_penalty
        elif Va < 12.0:
            done = True
            reward += terminal_penalty

        return float(reward), done