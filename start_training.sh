#!/bin/bash

# --- 配置区 ---
# 请确保以下路径指向你机器上的实际位置
WS_PATH=~/px4_ros2_ws
PX4_PATH=~/PX4-Autopilot  # 修改为你下载 PX4 源码的路径
PKG_NAME=vtol_rl_ros2
# --------------

echo "正在启动 PX4 仿真及 VTOL 强化学习全套环境..."

# 1. 启动 PX4 SITL Gazebo (Standard VTOL)
gnome-terminal --window --title="1. PX4 SITL Gazebo" -- bash -c "
echo '正在编译并启动 PX4 仿真...';
cd $PX4_PATH;
make px4_sitl_default gazebo-classic_standard_vtol;
exec bash"

# PX4 和 Gazebo 启动较慢，需要预留足够的加载时间
echo "等待仿真环境加载 (15秒)..."
sleep 7

# 2. 启动 Micro XRCE-DDS Agent
gnome-terminal --window --title="2. MicroXRCEAgent" -- bash -c "
echo '启动 Micro XRCE-DDS Agent...';
MicroXRCEAgent udp4 -p 8888;
exec bash"

# 等待 Agent 与 PX4 建立连接
sleep 3

# 3. 启动 C++ 底层执行机构管理节点
# 该节点负责处理起飞、过渡模态切换
gnome-terminal --window --title="3. C++ Offboard Manager" -- bash -c "
echo '启动 C++ 管理节点...';
cd $WS_PATH && source install/setup.bash;
ros2 run $PKG_NAME offboard_actuator_manager;
exec bash"

# 等待无人机起飞并达到训练高度
sleep 5

# 4. 启动 Python 强化学习训练节点
# 该节点运行 PPO 训练循环并记录日志
gnome-terminal --window --title="4. Python PPO Trainer" -- bash -c "
echo '启动 Python 训练节点...';
cd $WS_PATH && source install/setup.bash;
export PYTHONUNBUFFERED=1;
ros2 run $PKG_NAME ppo_train_node.py;
exec bash"

echo "所有任务已分发到独立终端窗口。"
