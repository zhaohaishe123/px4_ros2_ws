#include <rclcpp/rclcpp.hpp>
#include <stdint.h>
#include <cmath>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstdlib>
// PX4 消息
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_thrust_setpoint.hpp>
#include <px4_msgs/msg/vehicle_torque_setpoint.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vtol_vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <std_msgs/msg/bool.hpp>
// ROS 消息
#include <std_msgs/msg/float32_multi_array.hpp>

using namespace std::chrono_literals;

enum class FlightState {
    INIT,
    TAKEOFF,
    ACCELERATE_TRANSITION,
    RL_TRAINING_READY,
    EMERGENCY_RTL
};

class OffboardActuatorManager : public rclcpp::Node {
public:
    // 析构函数，节点结束时安全关闭 CSV 文件
    ~OffboardActuatorManager() {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
    }
    OffboardActuatorManager() : Node("offboard_actuator_manager") {
        // --- 发布者 ---
        offboard_control_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_pub_   = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        rates_pub_ = this->create_publisher<px4_msgs::msg::VehicleRatesSetpoint>("/fmu/in/vehicle_rates_setpoint", 10);
        //thrust_pub_ = this->create_publisher<px4_msgs::msg::VehicleThrustSetpoint>("/fmu/in/vehicle_thrust_setpoint", 10);
        //torque_pub_ = this->create_publisher<px4_msgs::msg::VehicleTorqueSetpoint>("/fmu/in/vehicle_torque_setpoint", 10);
        vehicle_command_pub_       = this->create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", 10);
        // [新增] 初始化状态位发布者
        ready_pub_ = this->create_publisher<std_msgs::msg::Bool>("/rl/training_ready", 10);

        // --- 使用 SensorDataQoS 订阅 PX4 状态 ---
        rclcpp::QoS qos_profile = rclcpp::SensorDataQoS();

        vehicle_status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
            "/fmu/out/vehicle_status", qos_profile, [this](const px4_msgs::msg::VehicleStatus::SharedPtr msg) { status_ = *msg; });
        
        local_position_sub_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", qos_profile, [this](const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg) { local_pos_ = *msg; });
            
        vtol_status_sub_ = this->create_subscription<px4_msgs::msg::VtolVehicleStatus>(
            "/fmu/out/vtol_vehicle_status", qos_profile, [this](const px4_msgs::msg::VtolVehicleStatus::SharedPtr msg) { vtol_status_ = *msg; });

        // RL 指令订阅
        rl_cmd_subscriber_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/rl/actuator_cmds", 10, std::bind(&OffboardActuatorManager::rl_cmd_callback, this, std::placeholders::_1));

        // --- 定时器 ---
        // 20Hz 控制循环
        timer_ = this->create_wall_timer(10ms, std::bind(&OffboardActuatorManager::timer_callback, this));
        
        // [新增] 1Hz 状态打印循环
        logging_timer_ = this->create_wall_timer(1000ms, std::bind(&OffboardActuatorManager::logging_timer_callback, this));
        
        last_rl_cmd_time_ = this->get_clock()->now();
        RCLCPP_INFO(this->get_logger(), "VTOL RL Manager (ROS 2) Started.");

        // ==========================================
        // [新增] 初始化 CSV 日志文件
        // ==========================================
        // 确保目录存在
        system("mkdir -p ~/px4_ros2_ws/data/vtol_rl/offboard_logs");
        
        std::string home_dir = getenv("HOME");
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << home_dir << "/px4_ros2_ws/data/vtol_rl/offboard_logs/offboard_cmds_" 
           << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".csv";
        
        csv_file_.open(ss.str());
        if (csv_file_.is_open()) {
            // 写入 CSV 表头
            csv_file_ << "Timestamp,State,VTOL_Type,Altitude_m,Speed_ms,Cmd_Throttle,Cmd_Elevator,Cmd_Aileron,Cmd_Rudder\n";
            RCLCPP_INFO(this->get_logger(), "CSV Logger initialized at: %s", ss.str().c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file for logging!");
        }
    }

private:
    px4_msgs::msg::VehicleStatus status_{};
    px4_msgs::msg::VehicleLocalPosition local_pos_{};
    px4_msgs::msg::VtolVehicleStatus vtol_status_{};
    
    FlightState flight_state_ = FlightState::INIT;
    rclcpp::Time last_request_time_ = this->get_clock()->now();
    rclcpp::Time last_rl_cmd_time_;


    bool rl_cmd_received_ = false;
    bool transition_cmd_sent_ = false;
    
    float rl_throttle_ = 0.6f, rl_elevator_ = 0.0f, rl_aileron_ = 0.0f, rl_rudder_ = 0.0f;

    void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0, float param3 = 0.0, float param4 = 0.0) {
        px4_msgs::msg::VehicleCommand msg{};
        msg.param1 = param1; msg.param2 = param2; msg.param3 = param3; msg.param4 = param4;
        msg.command = command;
        msg.target_system = 1; msg.target_component = 1; msg.source_system = 1; msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_command_pub_->publish(msg);
    }

    double get_speed() {
        return std::sqrt(local_pos_.vx * local_pos_.vx + local_pos_.vy * local_pos_.vy + local_pos_.vz * local_pos_.vz);
    }

    double get_distance(float x, float y) {
        float dx = local_pos_.x - x; float dy = local_pos_.y - y;
        return std::sqrt(dx * dx + dy * dy);
    }

    // ==========================================
    // [新增] 1Hz 日志打印回调函数
    // ==========================================
    void logging_timer_callback() {
        std::string state_str = "UNKNOWN";
        switch (flight_state_) {
            case FlightState::INIT: state_str = "INIT"; break;
            case FlightState::TAKEOFF: state_str = "TAKEOFF"; break;
            case FlightState::ACCELERATE_TRANSITION: state_str = "TRANSITION"; break;
            case FlightState::RL_TRAINING_READY: state_str = "RL_TRAINING"; break;
            case FlightState::EMERGENCY_RTL: state_str = "EMERG_RTL"; break;
        }

        std::string vtol_str = (status_.vehicle_type == px4_msgs::msg::VehicleStatus::VEHICLE_TYPE_FIXED_WING) ? "FW" : "MC";   
        // NED坐标系中Z向下为正，为了直观显示，我们打印 -local_pos_.z 作为高度
        RCLCPP_INFO(this->get_logger(), 
            "\033[1;36m[STATUS] State: %-12s | VTOL: %s | Alt: %5.1fm | Speed: %4.1fm/s | RL_Cmds[T:%.2f E:%.2f A:%.2f R:%.2f]\033[0m",
            state_str.c_str(), vtol_str.c_str(), 
            -local_pos_.z, get_speed(),
            rl_throttle_, rl_elevator_, rl_aileron_, rl_rudder_);
    }

    // ==========================================
    // 20Hz 控制主逻辑 (与之前相同)
    // ==========================================
    void timer_callback() {
        uint64_t timestamp_us = this->get_clock()->now().nanoseconds() / 1000;

        if (flight_state_ != FlightState::INIT && status_.arming_state != px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED) {
            RCLCPP_WARN(this->get_logger(), "Vehicle disarmed! Auto-Respawn triggering...");
            flight_state_ = FlightState::INIT;
            transition_cmd_sent_ = false; rl_cmd_received_ = false;
            last_request_time_ = this->get_clock()->now();
            return;
        }

        px4_msgs::msg::OffboardControlMode ocm{};
        ocm.timestamp = timestamp_us;
        if (flight_state_ == FlightState::RL_TRAINING_READY) {
            ocm.position = false;
            ocm.velocity = false;
            ocm.attitude = false;
            ocm.body_rate = true;  // <--- 开启角速度直控
            ocm.actuator = false;         
        } else {
            ocm.position = true;
            ocm.body_rate = false;
        }
        offboard_control_mode_pub_->publish(ocm);

        px4_msgs::msg::TrajectorySetpoint sp{};
        sp.timestamp = timestamp_us; sp.yaw = 0.0f; 
        float flight_altitude_ned = -50.0f; 

        switch (flight_state_) {
            case FlightState::INIT:
                sp.position = {0.0f, 0.0f, flight_altitude_ned};
                trajectory_setpoint_pub_->publish(sp);
                if ((this->get_clock()->now() - last_request_time_).seconds() > 5.0) {
                    if (status_.nav_state != px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD) {
                        publish_vehicle_command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
                    } else if (status_.arming_state != px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED) {
                        publish_vehicle_command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);
                    } else {
                        flight_state_ = FlightState::TAKEOFF;
                    }
                    last_request_time_ = this->get_clock()->now();
                }
                break;

            case FlightState::TAKEOFF:
                sp.position = {0.0f, 0.0f, flight_altitude_ned};
                trajectory_setpoint_pub_->publish(sp);
                if (std::abs(local_pos_.z - flight_altitude_ned) < 2.0f) {
                    flight_state_ = FlightState::ACCELERATE_TRANSITION;
                }
                break;

            case FlightState::ACCELERATE_TRANSITION:
                sp.position = {200.0f, 0.0f, flight_altitude_ned}; 
                trajectory_setpoint_pub_->publish(sp);
                if (status_.vehicle_type == px4_msgs::msg::VehicleStatus::VEHICLE_TYPE_FIXED_WING) {
                    transition_cmd_sent_ = false; rl_cmd_received_ = false;
                    flight_state_ = FlightState::RL_TRAINING_READY;
                } else {
                    if (!transition_cmd_sent_ && get_distance(0.0f, 0.0f) > 100.0 && get_speed() > 7.0 && std::abs(local_pos_.z - flight_altitude_ned) < 10.0) {
                        publish_vehicle_command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_VTOL_TRANSITION, 4.0f);
                        transition_cmd_sent_ = true;
                    }
                }
                break;

            case FlightState::RL_TRAINING_READY:
                {
                    px4_msgs::msg::VehicleRatesSetpoint rates_msg{};
                    rates_msg.timestamp = timestamp_us;

                    if (!rl_cmd_received_) {
                        // 如果还没收到 Python 的指令，维持安全的初始滑翔姿态
                        rates_msg.roll = 0.0f;   
                        rates_msg.pitch = 0.05f; 
                        rates_msg.yaw = 0.0f;    
                        rates_msg.thrust_body[0] = 0.65f; 
                    } else {
                        // 【核心】：完全交由强化学习神经网络控制！
                        rates_msg.roll = rl_aileron_;    // RL 算出的 Roll 角速度
                        rates_msg.pitch = rl_elevator_;  // RL 算出的 Pitch 角速度
                        rates_msg.yaw = rl_rudder_;      // RL 算出的 Yaw 角速度
                        rates_msg.thrust_body[0] = rl_throttle_; // RL 算出的 推力
                    }
                    
                    rates_msg.thrust_body[1] = 0.0f;
                    rates_msg.thrust_body[2] = 0.0f; 

                    rates_pub_->publish(rates_msg);
                }
                break;

            case FlightState::EMERGENCY_RTL:
                if (status_.nav_state != px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_AUTO_RTL) {
                    publish_vehicle_command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
                }
                break;
        }
        // ==========================================
        // [新增] 向外广播 RL 是否可以接管的状态位
        // ==========================================
        std_msgs::msg::Bool ready_msg;
        // 只有当 C++ 节点的状态机正式进入 RL_TRAINING_READY 阶段时，才输出 True
        ready_msg.data = (flight_state_ == FlightState::RL_TRAINING_READY);
        ready_pub_->publish(ready_msg);
    }

    void rl_cmd_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        if (msg->data.size() < 4) return;
        
        for (float val : msg->data) {
            if (std::isnan(val)) return; 
        }

        rl_cmd_received_ = true;
        last_rl_cmd_time_ = this->get_clock()->now();
        
        // 动作空间映射 (Action Mapping)
        // 0: Throttle -> 映射到 [0.1, 1.0] (保证最小空速)
        rl_throttle_ = 0.1f + ((msg->data[0] + 1.0f) / 2.0f) * 0.9f; 
        
        // 1: Pitch Rate -> 映射到 [-1.0, 1.0] rad/s (约57度/秒)
        rl_elevator_ = msg->data[1] * 0.4f;  
        
        // 2: Roll Rate -> 映射到 [-1.5, 1.5] rad/s (约85度/秒，滚转可以快一点)
        rl_aileron_  = msg->data[2] * 0.6f;  
        
        // 3: Yaw Rate -> 映射到 [-0.5, 0.5] rad/s (方向舵微调)
        rl_rudder_   = msg->data[3] * 0.3f;  

        // ==========================================
        // [新增] 每次收到强化学习网络指令时，写入 CSV
        // ==========================================
        if (csv_file_.is_open()) {
            std::string state_str = "UNKNOWN";
            switch (flight_state_) {
                case FlightState::INIT: state_str = "INIT"; break;
                case FlightState::TAKEOFF: state_str = "TAKEOFF"; break;
                case FlightState::ACCELERATE_TRANSITION: state_str = "TRANSITION"; break;
                case FlightState::RL_TRAINING_READY: state_str = "RL_TRAINING"; break;
                case FlightState::EMERGENCY_RTL: state_str = "EMERG_RTL"; break;
            }
            std::string vtol_str = (status_.vehicle_type == px4_msgs::msg::VehicleStatus::VEHICLE_TYPE_FIXED_WING) ? "FW" : "MC";   

            // 写入一行数据 (保留3位小数)
            csv_file_ << std::fixed << std::setprecision(3)
                      << this->get_clock()->now().seconds() << ","
                      << state_str << ","
                      << vtol_str << ","
                      << -local_pos_.z << ","
                      << get_speed() << ","
                      << rl_throttle_ << ","
                      << rl_elevator_ << ","
                      << rl_aileron_ << ","
                      << rl_rudder_ << "\n";
        }
    
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr logging_timer_; // 新增定时器
    
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_control_mode_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_command_pub_;

    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr vehicle_status_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_position_sub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr rates_pub_;
    //rclcpp::Publisher<px4_msgs::msg::VehicleThrustSetpoint>::SharedPtr thrust_pub_;
    //rclcpp::Publisher<px4_msgs::msg::VehicleTorqueSetpoint>::SharedPtr torque_pub_;
    rclcpp::Subscription<px4_msgs::msg::VtolVehicleStatus>::SharedPtr vtol_status_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr rl_cmd_subscriber_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ready_pub_; // [新增] 状态位发布者
    // [新增] CSV 文件流对象
    std::ofstream csv_file_;

};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardActuatorManager>());
    rclcpp::shutdown();
    return 0;
}