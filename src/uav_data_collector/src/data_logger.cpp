#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <mavros_msgs/msg/attitude_target.hpp>
#include <mavros_msgs/msg/rc_out.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem> // 用于创建文件夹 (C++17)
#include <string>

using namespace std::chrono_literals;
namespace fs = std::filesystem;

class DataLogger : public rclcpp::Node {
public:
    DataLogger() : Node("uav_data_logger") {
        // 1. 初始化物理量
        r = p = y = pr = qr = rr = rd = pd = yd = 0.0;
        for(int i=0; i<4; i++) act[i] = 0.0f;

        // 2. 创建保存路径
        // 获取当前工作目录下的 data 文件夹
        std::string dir_path = "src/uav_data_collector/data";
        if (!fs::exists(dir_path)) {
            fs::create_directories(dir_path);
        }

        // 3. 生成带时间戳的文件名
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << dir_path << "/flight_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << ".csv";
        std::string filename = ss.str();

        // 4. 打开文件并写入表头
        csv_file.open(filename, std::ios::out | std::ios::trunc);
        csv_file << "time,r,p,y,p_rate,q_rate,r_rate,r_d,p_d,y_d,ail,ele,rud,thr" << std::endl;

        RCLCPP_INFO(this->get_logger(), "数据将保存至: %s", filename.c_str());

        // 5. 订阅配置
        auto qos = rclcpp::SensorDataQoS();
        
        sub_pose = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mavros/local_position/pose", qos, [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                tf2::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
                tf2::Matrix3x3(q).getRPY(r, p, y);
            });

        sub_vel = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            "/mavros/local_position/velocity_body", qos, [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
                pr = msg->twist.angular.x; qr = msg->twist.angular.y; rr = msg->twist.angular.z;
            });

        sub_target = this->create_subscription<mavros_msgs::msg::AttitudeTarget>(
            "/mavros/setpoint_raw/target_attitude", qos, [this](const mavros_msgs::msg::AttitudeTarget::SharedPtr msg) {
                tf2::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
                tf2::Matrix3x3(q).getRPY(rd, pd, yd);
            });

        sub_rc_out = this->create_subscription<mavros_msgs::msg::RCOut>(
            "/mavros/rc/out", qos, [this](const mavros_msgs::msg::RCOut::SharedPtr msg) {
                if (msg->channels.size() >= 8) {
                    this->act[0] = (static_cast<float>(msg->channels[4]) - 1500.0f) / 500.0f;
                    this->act[1] = (static_cast<float>(msg->channels[5]) - 1500.0f) / 500.0f;
                    this->act[2] = (static_cast<float>(msg->channels[7]) - 1500.0f) / 500.0f;
                    this->act[3] = (static_cast<float>(msg->channels[6]) - 1000.0f) / 1000.0f;
                }
            });

        // 50Hz 定时器写入
        timer_ = this->create_wall_timer(20ms, std::bind(&DataLogger::save_to_csv, this));
    }

    ~DataLogger() {
        if (csv_file.is_open()) csv_file.close();
    }

private:
    void save_to_csv() {
        double t = this->get_clock()->now().seconds();
        // 只有当姿态角不为0（确保有数据流）才保存
        if (std::abs(r) > 1e-5 || std::abs(p) > 1e-5) {
            csv_file << std::fixed << std::setprecision(6) 
                     << t << "," << r << "," << p << "," << y << "," 
                     << pr << "," << qr << "," << rr << ","
                     << rd << "," << pd << "," << yd << ","
                     << act[0] << "," << act[1] << "," << act[2] << "," << act[3] << std::endl;
        }
    }

    std::ofstream csv_file;
    double r, p, y, pr, qr, rr, rd, pd, yd;
    float act[4];
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_vel;
    rclcpp::Subscription<mavros_msgs::msg::AttitudeTarget>::SharedPtr sub_target;
    rclcpp::Subscription<mavros_msgs::msg::RCOut>::SharedPtr sub_rc_out;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DataLogger>());
    rclcpp::shutdown();
    return 0;
}