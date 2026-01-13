#!/usr/bin/env python3
"""
Complete launch file for Jetson including motor control and networking
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_jetson_motor = get_package_share_directory('jetson_motor_control')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    host_ip = LaunchConfiguration('host_ip', default='192.168.1.5')  # Host PC IP
    jetson_ip = LaunchConfiguration('jetson_ip', default='192.168.1.100')  # Jetson IP
    
    # Configuration file
    config_file = os.path.join(pkg_jetson_motor, 'config', 'jetson_motor_params.yaml')
    
    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    declare_host_ip_cmd = DeclareLaunchArgument(
        'host_ip',
        default_value='192.168.1.5',
        description='IP address of the host PC'
    )
    
    declare_jetson_ip_cmd = DeclareLaunchArgument(
        'jetson_ip',
        default_value='192.168.1.100',
        description='IP address of the Jetson'
    )
    
    # Set ROS2 environment variables for networking
    set_domain_id = SetEnvironmentVariable('ROS_DOMAIN_ID', '42')
    set_localhost_only = SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0')

    
    # Restart ROS2 daemon to ensure clean networking
    restart_daemon = ExecuteProcess(
        cmd=['ros2', 'daemon', 'stop'],
        shell=False,
        output='screen'
    )
    
    # Jetson motor controller node
    motor_controller_node = Node(
        package='jetson_motor_control',
        executable='motor_controller',
        name='jetson_motor_controller',
        output='screen',
        parameters=[config_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('odom', '/odom'),
            ('cmd_vel', '/cmd_vel'),
            ('joint_states', '/joint_states'),
        ],
        # Ensure proper environment
        additional_env={'ROS_DOMAIN_ID': '42', 'ROS_LOCALHOST_ONLY': '0'}
    )
    
    # Static transform for base_footprint (commonly used by Nav2)
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'base_link', '--child-frame-id', 'base_footprint'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        additional_env={'ROS_DOMAIN_ID': '42', 'ROS_LOCALHOST_ONLY': '0'}
    )

    
    return LaunchDescription([
        # Environment setup
        set_domain_id,
        set_localhost_only, 
        
        # Launch arguments
        declare_use_sim_time_cmd,
        declare_host_ip_cmd,
        declare_jetson_ip_cmd,
        
        # Restart daemon first
        restart_daemon,
        
        # Main nodes
        motor_controller_node,
        static_transform_publisher,
    ])