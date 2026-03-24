cmake_minimum_required(VERSION 3.5)
project(uav_data_collector)

# 统一设置 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# 添加可执行文件
add_executable(data_logger src/data_logger.cpp)

# ROS2 依赖处理
ament_target_dependencies(data_logger
  rclcpp
  geometry_msgs
  mavros_msgs
  tf2
  tf2_geometry_msgs
)

# 链接 C++17 文件系统库 (去掉 PRIVATE 以匹配 ament 的模式)
target_link_libraries(data_logger stdc++fs)

# 安装
install(TARGETS
  data_logger
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()