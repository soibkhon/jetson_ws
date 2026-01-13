import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # Get the package directory
    pkg_path = os.path.join(get_package_share_directory('wheelchair_description'))
    controller_config = os.path.join(pkg_path, 'launch', 'controller.yaml')
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Controller manager
    controller_manager_cmd = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[controller_config, {'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Position controllers
    revolute_20_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['revolute_20_position_controller'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    revolute_21_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['revolute_21_position_controller'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    revolute_22_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['revolute_22_position_controller'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    revolute_23_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['revolute_23_position_controller'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    revolute_24_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['revolute_24_position_controller'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Robot state publisher
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        remappings=[('/joint_states', '/wheelchair/joint_states')],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    
    # Add nodes
    ld.add_action(controller_manager_cmd)
    ld.add_action(joint_state_broadcaster_spawner)
    ld.add_action(revolute_20_controller_spawner)
    ld.add_action(revolute_21_controller_spawner)
    ld.add_action(revolute_22_controller_spawner)
    ld.add_action(revolute_23_controller_spawner)
    ld.add_action(revolute_24_controller_spawner)
    ld.add_action(robot_state_publisher_cmd)
    
    return ld