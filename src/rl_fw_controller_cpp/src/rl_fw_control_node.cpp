#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vtol_vehicle_status.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/actuator_motors.hpp>
#include <px4_msgs/msg/actuator_servos.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>

#include <cmath>

using namespace std::chrono_literals;

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"

enum FlightState {
    WAIT_FOR_CONNECTION,
    INIT,
    TAKEOFF,
    FORWARD_ACCEL,    // <--- 新增：前飞加速阶段
    TRANSITION_TO_FW,
    FW_CRUISE
};

class RLDirectControlNode : public rclcpp::Node {
public:
    RLDirectControlNode() : Node("rl_direct_control_node"), flight_state_(WAIT_FOR_CONNECTION) {
        
        rclcpp::QoS qos_profile = rclcpp::SensorDataQoS();

        status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
            "/fmu/out/vehicle_status", qos_profile, [this](const px4_msgs::msg::VehicleStatus::SharedPtr msg) { 
                current_status_ = *msg; 
                has_status_ = true; 
            });
            
        local_pos_sub_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", qos_profile, [this](const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg) { 
                current_local_pos_ = *msg; 
            });
            
        vtol_status_sub_ = this->create_subscription<px4_msgs::msg::VtolVehicleStatus>(
            "/fmu/out/vtol_vehicle_status", qos_profile, [this](const px4_msgs::msg::VtolVehicleStatus::SharedPtr msg) { 
                current_vtol_status_ = *msg; 
                is_FW_ = (msg->vehicle_vtol_state == px4_msgs::msg::VtolVehicleStatus::VEHICLE_VTOL_STATE_FW);
            });

        offboard_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        vehicle_cmd_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", 10);
        trajectory_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        actuator_motors_pub_ = this->create_publisher<px4_msgs::msg::ActuatorMotors>("/fmu/in/actuator_motors", 10);
        actuator_servos_pub_ = this->create_publisher<px4_msgs::msg::ActuatorServos>("/fmu/in/actuator_servos", 10);

        takeoff_alt_ = -30.0; 

        timer_ = this->create_wall_timer(20ms, std::bind(&RLDirectControlNode::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), CYAN "RL 节点启动，包含前飞加速修正逻辑..." RESET);
    }

private:
    void timer_callback() {
        
        if (!has_status_) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                YELLOW "等待 PX4 /fmu/out/vehicle_status 数据接入..." RESET);
            return;
        }

        if (flight_state_ == WAIT_FOR_CONNECTION) {
            RCLCPP_INFO(this->get_logger(), GREEN "链路已完全打通！进入初始化起飞程序。" RESET);
            flight_state_ = INIT;
            last_req_ = this->get_clock()->now();
        }

        auto now = this->get_clock()->now();
        publish_offboard_control_mode();

        switch (flight_state_) {
            case INIT:
            {
                publish_trajectory_setpoint(0.0, 0.0, takeoff_alt_);

                if (current_status_.nav_state != px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD) {
                    if ((now - last_req_).seconds() > 2.0) {
                        this->publish_vehicle_command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
                        RCLCPP_INFO(this->get_logger(), "正在请求进入 OFFBOARD 模式...");
                        last_req_ = now;
                    }
                } 
                else if (current_status_.arming_state != px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED) {
                    if ((now - last_req_).seconds() > 2.0) {
                        this->publish_vehicle_command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);
                        RCLCPP_INFO(this->get_logger(), "OFFBOARD 成功，正在请求解锁 (ARM)...");
                        last_req_ = now;
                    }
                } 
                else {
                    RCLCPP_INFO(this->get_logger(), GREEN "已解锁! 旋翼起飞中..." RESET);
                    flight_state_ = TAKEOFF;
                }
                break;
            }

            case TAKEOFF:
            {
                publish_trajectory_setpoint(0.0, 0.0, takeoff_alt_);
                
                // 到达起飞高度后，切入前飞加速阶段
                if (current_local_pos_.z <= takeoff_alt_ + 1.0) {
                    RCLCPP_INFO(this->get_logger(), GREEN "到达起飞高度. 开始前飞加速 (Forward Accel)..." RESET);
                    flight_state_ = FORWARD_ACCEL;
                    last_req_ = now;
                }
                break;
            }

            case FORWARD_ACCEL:
            {
                // <--- 核心修改 1：设定点在当前位置正前方（北向X轴）50米处，引导多旋翼前倾加速
                publish_trajectory_setpoint(current_local_pos_.x + 50.0, current_local_pos_.y, takeoff_alt_);
                
                // 计算当前地速
                float speed = std::sqrt(current_local_pos_.vx * current_local_pos_.vx + 
                                        current_local_pos_.vy * current_local_pos_.vy);
                
                // 持续加速 4 秒，或者速度超过 6.0 m/s 后执行转换
                if (speed > 6.0 || (now - last_req_).seconds() > 4.0) {
                    RCLCPP_INFO(this->get_logger(), GREEN "已具备初始前飞速度 (%.1f m/s). 开始执行 VTOL 转换 (Transition)." RESET, speed);
                    flight_state_ = TRANSITION_TO_FW;
                    last_req_ = now;
                }
                break;
            }

            case TRANSITION_TO_FW:
            {
                if (!is_FW_) {
                    if ((now - last_req_).seconds() > 3.0) {
                        RCLCPP_INFO(this->get_logger(), YELLOW "正在发送 VTOL 转换指令 (Target: FW)..." RESET);
                        this->publish_vehicle_command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_VTOL_TRANSITION, 4.0);
                        last_req_ = now;
                    }
                    // <--- 核心修改 2：转换期间必须继续保持向前的设定点，绝不能设回(0,0)，否则多旋翼会试图后仰刹车！
                    publish_trajectory_setpoint(current_local_pos_.x + 50.0, current_local_pos_.y, takeoff_alt_);
                } else {
                    RCLCPP_INFO(this->get_logger(), GREEN "检测到已完全进入 FW 模式！切入底层 Actuator 控制。" RESET);
                    flight_state_ = FW_CRUISE;
                }
                break;
            }

            case FW_CRUISE:
            {
                // Offboard 模式下切换到 actuator = true
                float delta_t = 0.7; // 推力电机 70%
                float delta_e = 0.0; 
                float delta_a = 0.0; 
                float delta_r = 0.0; 

                publish_direct_actuators(delta_t, delta_e, delta_a, delta_r);

                if (step_counter_ % 50 == 0) {
                    RCLCPP_INFO(this->get_logger(), CYAN "[FW 巡航] 高度: %.1f m | 油门:%.1f, 升降:%.2f" RESET, 
                                -current_local_pos_.z, delta_t, delta_e);
                }
                break;
            }
            default: break;
        }

        step_counter_++;
    }

    void publish_offboard_control_mode() {
        px4_msgs::msg::OffboardControlMode msg{};
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        
        if (flight_state_ == FW_CRUISE) {
            msg.position = false;
            msg.actuator = true;  // PX4 1.14 规范
        } else {
            msg.position = true;         
            msg.actuator = false;
        }
        
        msg.velocity = false;
        msg.acceleration = false;
        msg.attitude = false;
        msg.body_rate = false;
        offboard_mode_pub_->publish(msg);
    }

    void publish_trajectory_setpoint(float x, float y, float z) {
        px4_msgs::msg::TrajectorySetpoint msg{};
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        msg.position = {x, y, z};
        msg.yaw = 0.0; 
        msg.velocity = {NAN, NAN, NAN};
        msg.acceleration = {NAN, NAN, NAN};
        msg.jerk = {NAN, NAN, NAN};
        trajectory_pub_->publish(msg);
    }

    void publish_direct_actuators(float delta_t, float delta_e, float delta_a, float delta_r) {
        uint64_t timestamp = this->get_clock()->now().nanoseconds() / 1000;

        px4_msgs::msg::ActuatorMotors motor_msg{};
        motor_msg.timestamp = timestamp;
        std::fill(std::begin(motor_msg.control), std::end(motor_msg.control), NAN);
        motor_msg.control[4] = delta_t; 
        actuator_motors_pub_->publish(motor_msg);

        px4_msgs::msg::ActuatorServos servo_msg{};
        servo_msg.timestamp = timestamp;
        std::fill(std::begin(servo_msg.control), std::end(servo_msg.control), NAN);
        servo_msg.control[0] = delta_a;
        servo_msg.control[1] = -delta_a; 
        servo_msg.control[2] = delta_e;
        servo_msg.control[3] = delta_r;
        actuator_servos_pub_->publish(servo_msg);
    }

    void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) {
        px4_msgs::msg::VehicleCommand msg{};
        msg.param1 = param1;
        msg.param2 = param2;
        msg.command = command;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_cmd_pub_->publish(msg);
    }

    FlightState flight_state_;
    bool has_status_ = false;
    bool is_FW_ = false;
    float takeoff_alt_;
    uint64_t step_counter_ = 0;
    rclcpp::Time last_req_;

    px4_msgs::msg::VehicleStatus current_status_;
    px4_msgs::msg::VehicleLocalPosition current_local_pos_;
    px4_msgs::msg::VtolVehicleStatus current_vtol_status_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_mode_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_cmd_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_pub_;
    rclcpp::Publisher<px4_msgs::msg::ActuatorMotors>::SharedPtr actuator_motors_pub_;
    rclcpp::Publisher<px4_msgs::msg::ActuatorServos>::SharedPtr actuator_servos_pub_;

    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr status_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_pos_sub_;
    rclcpp::Subscription<px4_msgs::msg::VtolVehicleStatus>::SharedPtr vtol_status_sub_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RLDirectControlNode>());
    rclcpp::shutdown();
    return 0;
}