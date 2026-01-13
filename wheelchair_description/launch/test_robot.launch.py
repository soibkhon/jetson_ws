import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    
    # Get the package directory
    pkg_path = os.path.join(get_package_share_directory('wheelchair_description'))
    xacro_file = os.path.join(pkg_path, 'urdf', 'wheelchair.xacro')
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Process the XACRO file and publish as parameter
    robot_description_content = Command(['xacro ', xacro_file])
    
    # Robot state publisher
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )
    
    # Joint state publisher (without GUI for testing)
    joint_state_publisher_cmd = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Echo robot_description to verify it's published
    echo_robot_description = ExecuteProcess(
        cmd=['ros2', 'param', 'get', '/robot_state_publisher', 'robot_description'],
        output='screen'
    )
    
    # Launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    
    # Add nodes
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(joint_state_publisher_cmd)
    
    return ld