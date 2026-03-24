cmake_minimum_required(VERSION 3.8)
project(vtol_rl_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# 编译 C++ 节点
add_executable(offboard_actuator_manager src/offboard_actuator_manager.cpp)
ament_target_dependencies(offboard_actuator_manager rclcpp px4_msgs std_msgs geometry_msgs sensor_msgs)

# 1. 安装 C++ 可执行文件
install(TARGETS
  offboard_actuator_manager
  DESTINATION lib/${PROJECT_NAME}
)

# 2. 安装 Python 脚本目录
# 这将把整个 scripts 文件夹安装到包的运行路径下，使 ros2 run 能够执行其中的 python 文件
install(DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

# 安装 Launch 文件 (如果有)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
ament_package()