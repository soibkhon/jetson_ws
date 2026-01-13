import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import launch
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Get package directory
    pkg_jetson_motor = get_package_share_directory('jetson_motor_control')
    realsense_share_dir = get_package_share_directory('realsense2_camera')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    config_file = LaunchConfiguration('config_file')
    
    # Paths
    default_config_file = os.path.join(pkg_jetson_motor, 'config', 'jetson_motor_params.yaml')
    
    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Full path to the configuration file'
    )
    
    # Jetson motor controller node
    motor_controller_node = Node(
        package='jetson_motor_control',
        executable='lidar_control',
        name='lidar_control',
        output='screen',
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(realsense_share_dir, 'launch', 'rs_launch.py')),
        launch_arguments={
                'camera_name': 'camera',
                'camera_namespace': 'camera',
                'pointcloud.enable': 'true',
                'pointcloud.allow_no_texture_points': 'true',
                'pointcloud.ordered_pc': 'true',
                'align_depth.enable': 'true',
                'rgb_camera.profile': '640x480x30',
                'depth_module.profile': '640x480x30'
            }.items()
    )
    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_config_file_cmd,
        motor_controller_node,
        realsense_launch,
    ])
