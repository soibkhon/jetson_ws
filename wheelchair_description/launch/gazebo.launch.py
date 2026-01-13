import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # Get the package directory
    pkg_path = os.path.join(get_package_share_directory('wheelchair_description'))
    xacro_file = os.path.join(pkg_path, 'urdf', 'wheelchair.xacro')
    
    # Get Gazebo launch file
    gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world')
    
    # Robot state publisher
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', xacro_file]),
            'use_sim_time': use_sim_time
        }]
    )
    
    # Gazebo launch
    gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'pause': 'true',
            'use_sim_time': use_sim_time,
            'gui': 'true',
            'headless': 'false',
            'debug': 'false'
        }.items()
    )
    
    # Spawn robot
    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'wheelchair'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='',
        description='SDF world file'
    )
    
    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)
    
    # Add nodes
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(gazebo_cmd)
    ld.add_action(spawn_entity_cmd)
    
    return ld