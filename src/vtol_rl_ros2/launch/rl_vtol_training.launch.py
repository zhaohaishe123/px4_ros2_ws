import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # 1. 启动 Micro XRCE-DDS Agent (负责 PX4 与 ROS 2 的通信)
        # 注意：这相当于在终端运行 MicroXRCEAgent udp4 -p 8888
        ExecuteProcess(
            cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
            output='screen'
        ),

        # 2. 启动 C++ 底层执行机构管理节点
        Node(
            package='vtol_rl_ros2',
            executable='offboard_actuator_manager',
            name='offboard_actuator_manager_node',
            output='screen',
            emulate_tty=True # 保证终端彩色输出正常显示
        ),

        # 3. 启动 Python 强化学习训练节点
        Node(
            package='vtol_rl_ros2',
            executable='ppo_train_node.py', 
            name='ppo_train_node',
            output='screen',
            emulate_tty=True,
            # [新增] 强制 Python 实时输出，不进行缓冲
            additional_env={'PYTHONUNBUFFERED': '1'} 
        )
    ])