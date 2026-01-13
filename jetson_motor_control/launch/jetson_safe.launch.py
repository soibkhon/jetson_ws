#!/usr/bin/env python3
"""
wheelchair_motor_with_collision.launch.py - Jetson-side motor control with collision avoidance
Launches motor controller, collision avoidance, and velocity scaling nodes
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    # 1. Collision Avoidance Node
    collision_avoidance_node = Node(
        package='jetson_motor_control',  # Replace with your package name
        executable='collision_avoidance',
        name='collision_avoidance',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'safety_distance': 0.3,
            'warning_distance': 1.0,
            'front_sector_angle': 90.0,  # degrees
            'side_sector_angle': 120.0,   # degrees
            'laser_scan_topic': '/scan',
            'filtered_scan_topic': '/scan_filtered',
            'robot_width': 0.7,          # wheelchair width
            'robot_length': 1.0,        # wheelchair length
            'filter_margin': 0.1,        # extra margin for filtering
            'min_valid_range': 0.15,     # minimum valid range
            'publish_filtered_scan': True,
            'debug_mode': False,
        }]
    )
    
    # 2. Velocity Scaler Node
    velocity_scaler_node = Node(
        package='jetson_motor_control',  # Replace with your package name
        executable='velocity_scaler_node',
        name='velocity_scaler',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'base_linear_speed_multiplier': 2.0,   # Make it faster
            'base_angular_speed_multiplier': 2.0,  # Make turning faster
            'safety_distance': 0.3,
            'warning_distance': 1.0,
            'min_speed_factor': 0.1,
            'emergency_stop_factor': 0.0,
            'joystick_speed_multiplier': 1.5,
            'input_cmd_vel_topic': 'cmd_vel_nav',   # From Nav2
            'output_cmd_vel_topic': 'cmd_vel',      # To motor controller
            'joystick_cmd_vel_topic': 'cmd_vel_joy',
        }]
    )
    
    # 3. Jetson Motor Controller (Simplified)
    jetson_motor_node = Node(
        package='jetson_motor_control',  # Your motor control package
        executable='jetson_motor_control_simple',
        name='jetson_motor_controller',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            # Motor configuration
            'can_interface': 'socketcan',
            'can_channel': 'can1',
            'motor_left_id': 1,
            'motor_right_id': 2,
            'wheel_diameter': 0.243,
            'wheel_base': 0.598,
            'encoder_resolution': 4096,
            'max_linear_velocity': 1.0,      # Increased base speed
            'max_angular_velocity': 0.8,     # Increased base turning
            'max_joystick_speed': 120,       # Increased joystick speed
            
            # GPIO configuration
            'gpio_pin_speed_up': 29,
            'gpio_pin_speed_down': 31,
            'gpio_pin_brake': 33,
            'hold_time_enable_motors': 3.0,
            'hold_time_disable_motors': 2.0,
            
            # Control parameters
            'publish_rate': 50.0,
            'joystick_publish_rate': 20.0,
            'joystick_deadzone': 0.2,
            'speed_scale_increment': 0.1,
            'cmd_vel_timeout': 0.5,
            'button_debounce_time': 0.3,
            
            # Frame configuration
            'odom_frame_id': 'odom',
            'base_frame_id': 'base_link',
            'publish_tf': True,
        }],
        remappings=[
            # Receives processed cmd_vel from velocity_scaler
            ('cmd_vel', 'cmd_vel'),
        ]
    )
    
    # 4. Optional: Laser scan relay (if needed to remap topics)
    scan_relay_node = Node(
        package='topic_tools',
        executable='relay',
        name='scan_relay',
        arguments=['/livox/lidar', '/scan'],
        condition=None,  # Add condition if needed
        remappings=[]
    )

    return LaunchDescription([
        # Launch arguments
        declare_use_sim_time_cmd,
        
        # Core nodes
        collision_avoidance_node,
        velocity_scaler_node,
        jetson_motor_node,
        
        # Optional nodes
        # scan_relay_node,  # Uncomment if you need to relay scan topics
    ])