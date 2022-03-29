#include "ros/ros.h"
#include <math.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include <std_msgs/UInt32MultiArray.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/frustum_culling.h>
#include <pcl/filters/filter_indices.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcloud_projection/FilteredPointCloud.h>

tf2_ros::Buffer tf_buffer;
Eigen::Matrix4f camera_pose;
sensor_msgs::CameraInfo latest_cam_info;
ros::Publisher publisher;
ros::Publisher pc_publisher;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
std_msgs::UInt32MultiArray indicesArray;

Eigen::Matrix4f odom2matrix4f(const geometry_msgs::PoseStamped pose)
{
    auto orientation = pose.pose.orientation;
    auto position = pose.pose.position;

    Eigen::Quaterniond quat;
    quat.w() = orientation.w;
    quat.x() = orientation.x;
    quat.y() = orientation.y;
    quat.z() = orientation.z;

    Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
    isometry.linear() = quat.toRotationMatrix();
    isometry.translation() = Eigen::Vector3d(position.x, position.y, position.z);

    Eigen::Matrix4f cam2robot;
    // Swap the axes according to the robot
    cam2robot << 0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    return isometry.matrix().cast<float>() * cam2robot;
}

void processAndPublish()
{
    if (cloud->empty())
    {
        std::cout << "Map is not available yet. Skipping..." << std::endl;
    }
    std::cout << "Performing frustum culling" << std::endl;
    pcl::FrustumCulling<pcl::PointXYZRGB> fc(true);
    fc.setInputCloud(cloud);
    fc.setCameraPose(camera_pose);
    fc.setHorizontalFOV(80);
    fc.setVerticalFOV(60);
    fc.setNearPlaneDistance(0);
    fc.setFarPlaneDistance(15);

    pcl::PointCloud<pcl::PointXYZRGB> target;
    fc.filter(target);
    pcl::Indices indices;
    fc.filter(indices);
    if (target.empty())
    {
        std::cout << "Frustum is empty. Is there something wrong in the camera pose settings?" << std::endl;
        return;
    }
    indicesArray.data.clear();
    for (const auto i : indices)
    {
        indicesArray.data.push_back(i);
    }
    std::cout << "Publishing frustum cloud and point indices" << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB> transformed_target;
    pcloud_projection::FilteredPointCloud out_msg;

    auto transform = tf_buffer.lookupTransform("openni_rgb_optical_frame", "map", ros::Time(0));
    pcl_ros::transformPointCloud(target, transformed_target, transform.transform);
    out_msg.indices = indicesArray;
    pcl::toROSMsg(transformed_target, out_msg.pc);
    out_msg.pc.header.frame_id = "openni_rgb_optical_frame";
    publisher.publish(out_msg);
    pc_publisher.publish(out_msg.pc);
}

void onOdomReceived(const nav_msgs::Odometry &odom_ptr)
{
    // const auto &transform = tf_buffer.lookupTransform("map", "odom", ros::Time(0));
    try
    {
        std::cout << "Odom received" << std::endl;
        auto pose = geometry_msgs::PoseStamped();
        pose.pose = odom_ptr.pose.pose;
        pose.header = odom_ptr.header;
        const auto t = tf_buffer.lookupTransform("map", pose.header.frame_id, ros::Time(0));
        pose.header.stamp = t.header.stamp;
        const auto transformed_odom = tf_buffer.transform(pose, "map");
        camera_pose = odom2matrix4f(transformed_odom);
        processAndPublish();
    } catch (tf2::ConnectivityException& e){
        std::cout << "Skipping update because no TF can be found: " << e.what() << std::endl; 
    } catch (std::exception& e){
        std::cout << "An unexpected error occured, skipping this update: " << e.what() << std::endl;
    }
}

void onCamInfoReceived(sensor_msgs::CameraInfo::ConstPtr cam_info)
{
    latest_cam_info = *(cam_info.get());
}

void onMapReceived(const sensor_msgs::PointCloud2ConstPtr cloud_ptr)
{
    std::cout << "Map received" << std::endl;
    pcl::fromROSMsg(*cloud_ptr.get(), *cloud);
    processAndPublish();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcloud_projection_node");
    ros::NodeHandle nh("~");
    tf2_ros::TransformListener tf_listener(tf_buffer);
    auto map_subscriber = nh.subscribe("/rtabmap/cloud_map", 1, onMapReceived);
    auto odom_subscriber = nh.subscribe("/rtabmap/odom", 1, onOdomReceived);
    auto cam_info_subscriber = nh.subscribe("/camera/rgb/camera_info", 100, onCamInfoReceived);
    publisher = nh.advertise<pcloud_projection::FilteredPointCloud>("/culling/pc_with_indices", 1);
    pc_publisher = nh.advertise<sensor_msgs::PointCloud2>("/culling/pc", 1);
    std::cout << "Point cloud culling started" << std::endl;
    ros::spin();
}