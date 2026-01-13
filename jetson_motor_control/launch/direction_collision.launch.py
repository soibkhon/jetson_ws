#!/usr/bin/env python3
"""
Launch file for direction-based collision avoidance
Simple approach using joystick x,y values
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_jetson_motor_control = get_package_share_directory('jetson_motor_control')
    
    # Configuration file
    config_file = os.path.join(pkg_jetson_motor_control, 'config', 'direction_collision_config.yaml')

    # Launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    # Direction-based collision avoidance node
    collision_node = Node(
        package='jetson_motor_control',
        executable='direction_based_collision',
        name='direction_collision_node',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )
    
    # Your original motor controller (unchanged)
    motor_controller = Node(
        package='jetson_motor_control',
        executable='jetson_motor_control',
        name='jetson_motor_controller', 
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        collision_node,
        motor_controller,
    ])