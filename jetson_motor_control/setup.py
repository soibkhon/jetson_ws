from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'jetson_motor_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Motor control package for Jetson Orin Nano with GPIO joystick',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_controller = jetson_motor_control.jetson_motor_controller:main',
            'collision_avoid = jetson_motor_control.jetson_collision_avoid:main',
            'velocity_scaler_node = jetson_motor_control.velocity_scaler_node:main',
            'jetson_motor_control_simple = jetson_motor_control.jetson_motor_control_simple:main',
            'collision_avoidance = jetson_motor_control.collision_avoidance_node:main',
            'direction_based_collision = jetson_motor_control.direction_based_collision:main',
            'joystick_collision_filter = jetson_motor_control.joystick_collision_filter:main',
            'keep_safe_node = jetson_motor_control.keep_safe_node:main',
            'complementary_node = jetson_motor_control.complementary_node:main',
            'lidar_control = jetson_motor_control.lidar_control:main',
        ],
    },
)