#include <rclcpp/rclcpp.hpp>
#include <stdint.h>
#include <cmath>
#include <string>

// PX4 消息
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/actuator_motors.hpp>
#include <px4_msgs/msg/actuator_servos.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vtol_vehicle_status.hpp>

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
    OffboardActuatorManager() : Node("offboard_actuator_manager") {
        // --- 发布者 ---
        offboard_control_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_pub_   = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        actuator_motors_pub_       = this->create_publisher<px4_msgs::msg::ActuatorMotors>("/fmu/in/actuator_motors", 10);
        actuator_servos_pub_       = this->create_publisher<px4_msgs::msg::ActuatorServos>("/fmu/in/actuator_servos", 10);
        vehicle_command_pub_       = this->create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", 10);

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
            ocm.actuator = true;  // PX4 1.14 使用 actuator
        } else {
            ocm.position = true;
            ocm.actuator = false;
        }
        offboard_control_mode_pub_->publish(ocm);

        px4_msgs::msg::TrajectorySetpoint sp{};
        sp.timestamp = timestamp_us; sp.yaw = 0.0f; 
        float flight_altitude_ned = -30.0f; 

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
                    px4_msgs::msg::ActuatorMotors motor_msg{};
                    px4_msgs::msg::ActuatorServos servo_msg{};
                    motor_msg.timestamp = timestamp_us; servo_msg.timestamp = timestamp_us;
                    for (int i = 0; i < 12; i++) motor_msg.control[i] = 0.0f;
                    for (int i = 0; i < 8; i++) servo_msg.control[i] = 0.0f;

                    if (!rl_cmd_received_) {
                        motor_msg.control[4] = 0.6f; 
                        servo_msg.control[0] = 0.0f; servo_msg.control[1] = 0.0f; servo_msg.control[2] = 0.0f; 
                    } else {
                        if ((this->get_clock()->now() - last_rl_cmd_time_).seconds() > 5.0) {
                            flight_state_ = FlightState::EMERGENCY_RTL;
                        } else {
                            motor_msg.control[4] = rl_throttle_;
                            servo_msg.control[0] = rl_elevator_;
                            servo_msg.control[1] = rl_aileron_;
                            servo_msg.control[2] = rl_rudder_;
                        }
                    }
                    actuator_motors_pub_->publish(motor_msg);
                    actuator_servos_pub_->publish(servo_msg);
                }
                break;

            case FlightState::EMERGENCY_RTL:
                if (status_.nav_state != px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_AUTO_RTL) {
                    publish_vehicle_command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
                }
                break;
        }
    }

    void rl_cmd_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    if (msg->data.size() < 4) return;
    
    // 如果收到的数据里有任何一个是 NaN，直接丢弃，不更新指令
    for (float val : msg->data) {
        if (std::isnan(val)) return; 
    }

    rl_cmd_received_ = true;
    last_rl_cmd_time_ = this->get_clock()->now();
    
    // 映射与赋值
    rl_throttle_ = (msg->data[0] + 1.0f) / 2.0f; 
    rl_elevator_ = msg->data[1];
    rl_aileron_  = msg->data[2];
    rl_rudder_   = msg->data[3];
}

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr logging_timer_; // 新增定时器
    
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_control_mode_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::ActuatorMotors>::SharedPtr actuator_motors_pub_;
    rclcpp::Publisher<px4_msgs::msg::ActuatorServos>::SharedPtr actuator_servos_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_command_pub_;

    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr vehicle_status_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_position_sub_;
    rclcpp::Subscription<px4_msgs::msg::VtolVehicleStatus>::SharedPtr vtol_status_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr rl_cmd_subscriber_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardActuatorManager>());
    rclcpp::shutdown();
    return 0;
}