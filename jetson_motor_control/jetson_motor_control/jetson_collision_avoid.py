#!/usr/bin/env python3
"""
Jetson Motor Control Node with Collision Avoidance
ROS2 node for differential drive robot control with GPIO interface and laser-based collision avoidance
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Joy, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
import tf_transformations
import math
import time
import threading
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import Jetson.GPIO as GPIO
from jetson_motor_control.l2db_motor_driver import L2DBMotorDriver
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class JetsonMotorController(Node):
    
    def __init__(self):
        super().__init__('jetson_motor_controller')
        
        self._declare_params()
        self._get_params()
        self._init_collision_avoidance()  # Initialize collision avoidance FIRST
        self._init_hardware()
        self._init_state()
        self._setup_ros_interfaces()
        
        self.init_thread = threading.Thread(target=self._initialize_motors, daemon=True)
        self.init_thread.start()
        
        self.get_logger().info("Jetson Motor Controller with Collision Avoidance initialized")
    
    def _declare_params(self):
        """Declare ROS2 parameters"""
        # Motor configuration
        self.declare_parameter('can_interface', 'socketcan')
        self.declare_parameter('can_channel', 'can1')
        self.declare_parameter('motor_left_id', 1)
        self.declare_parameter('motor_right_id', 2)
        self.declare_parameter('wheel_diameter', 0.243)
        self.declare_parameter('wheel_base', 0.598)
        self.declare_parameter('encoder_resolution', 4096)
        self.declare_parameter('max_linear_velocity', 0.83)
        self.declare_parameter('max_angular_velocity', 0.5)
        self.declare_parameter('max_joystick_speed', 100)
        
        # GPIO configuration
        self.declare_parameter('gpio_pin_speed_up', 29)
        self.declare_parameter('gpio_pin_speed_down', 31)
        self.declare_parameter('gpio_pin_brake', 33)
        self.declare_parameter('hold_time_enable_motors', 3.0)
        self.declare_parameter('hold_time_disable_motors', 2.0)
        
        # Control parameters
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('joystick_publish_rate', 20.0)
        self.declare_parameter('joystick_deadzone', 0.2)
        self.declare_parameter('speed_scale_increment', 0.1)
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('button_debounce_time', 0.3)
        
        # Collision avoidance parameters
        self.declare_parameter('collision_avoidance_enabled', True)
        self.declare_parameter('safety_distance', 0.4)
        self.declare_parameter('warning_distance', 1)
        self.declare_parameter('max_deceleration_factor', 0.4)
        self.declare_parameter('use_laser_scan', True)
        self.declare_parameter('front_sector_angle', 60)
        self.declare_parameter('side_sector_angle', 60)
        self.declare_parameter('laser_scan_topic', '/scan')
        
        # Frame configuration
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('publish_tf', True)
        
        # Covariance matrices
        self.declare_parameter('pose_covariance_diagonal', [0.01, 0.01, 0.001, 0.001, 0.001, 0.09])
        self.declare_parameter('twist_covariance_diagonal', [0.002, 0.001, 0.001, 0.001, 0.001, 0.02])
        
    def _get_params(self):
        """Get parameter values and calculate derived constants"""
        # Get all parameter values
        self.can_interface = self.get_parameter('can_interface').value
        self.can_channel = self.get_parameter('can_channel').value
        self.motor_left_id = self.get_parameter('motor_left_id').value
        self.motor_right_id = self.get_parameter('motor_right_id').value
        self.wheel_diameter = self.get_parameter('wheel_diameter').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.encoder_resolution = self.get_parameter('encoder_resolution').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.max_joystick_speed = self.get_parameter('max_joystick_speed').value
        self.hold_time_enable = self.get_parameter('hold_time_enable_motors').value
        self.hold_time_disable = self.get_parameter('hold_time_disable_motors').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.joystick_publish_rate = self.get_parameter('joystick_publish_rate').value
        self.joystick_deadzone = self.get_parameter('joystick_deadzone').value
        self.speed_scale_increment = self.get_parameter('speed_scale_increment').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.button_debounce_time = self.get_parameter('button_debounce_time').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.pose_covariance_diagonal = self.get_parameter('pose_covariance_diagonal').value
        self.twist_covariance_diagonal = self.get_parameter('twist_covariance_diagonal').value
        
        # GPIO pin mapping
        self.gpio_pins = {
            'speed_up': self.get_parameter('gpio_pin_speed_up').value,
            'speed_down': self.get_parameter('gpio_pin_speed_down').value,
            'brake': self.get_parameter('gpio_pin_brake').value
        }
        
        # Derived constants
        self.wheel_radius = self.wheel_diameter / 2.0
        self.wheel_circumference = math.pi * self.wheel_diameter
        self.meters_per_count = self.wheel_circumference / self.encoder_resolution
    
    def _init_collision_avoidance(self):
        """Initialize collision avoidance system - called early to set all attributes"""
        # Set default values first to ensure attributes exist
        self.collision_avoidance_enabled = True
        self.use_laser_scan = True
        self.laser_scan_topic = '/scan'
        self.safety_distance = 0.3
        self.warning_distance = 1.0
        self.max_deceleration_factor = 0.4
        self.front_sector_angle = 60
        self.side_sector_angle = 55

        # Override with actual parameter values if available
        try:
            self.collision_avoidance_enabled = self.get_parameter('collision_avoidance_enabled').value
            self.safety_distance = self.get_parameter('safety_distance').value
            self.warning_distance = self.get_parameter('warning_distance').value
            self.max_deceleration_factor = self.get_parameter('max_deceleration_factor').value
            self.use_laser_scan = self.get_parameter('use_laser_scan').value
            self.front_sector_angle = self.get_parameter('front_sector_angle').value
            self.side_sector_angle = self.get_parameter('side_sector_angle').value
            self.laser_scan_topic = self.get_parameter('laser_scan_topic').value
        except Exception as e:
            self.get_logger().warn(f"Could not load some collision avoidance parameters, using defaults: {e}")
        
        # Collision detection state
        self.last_scan = None
        self.collision_detected = False
        self.collision_direction = None  # 'front', 'left', 'right'
        self.min_obstacle_distance = float('inf')
        
        self.get_logger().info(f"Collision avoidance initialized (enabled: {self.collision_avoidance_enabled})")
        if self.use_laser_scan:
            self.get_logger().info(f"Using laser scan from topic: {self.laser_scan_topic}")
            self.get_logger().info(f"Front sector: ±{self.front_sector_angle/2}°, Side sector: ±{self.side_sector_angle}°")
        self.get_logger().info(f"Safety distance: {self.safety_distance}m, Warning distance: {self.warning_distance}m")
    
    def _init_hardware(self):
        """Initialize all hardware components"""
        self._init_gpio()
        self._init_motors()
        self._init_analog_joystick()
    
    def _init_gpio(self):
        """Initialize GPIO pins"""
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)
        
        for pin in self.gpio_pins.values():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        self.button_states = {name: False for name in self.gpio_pins.keys()}
        self.button_hold_times = {name: None for name in self.gpio_pins.keys()}
        self.last_button_press = {}
        
        self.get_logger().info(f"GPIO initialized on pins: {list(self.gpio_pins.values())}")
    
    def _init_motors(self):
        """Initialize motor drivers"""
        self.motor_left = L2DBMotorDriver(self.can_interface, self.can_channel, self.motor_left_id)
        self.motor_right = L2DBMotorDriver(self.can_interface, self.can_channel, self.motor_right_id)
    
    def _init_analog_joystick(self):
        """Initialize analog joystick via I2C/ADS1115"""
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(self.i2c)
            self.ads.gain = 1
            
            self.x_channel = AnalogIn(self.ads, ADS.P0)
            self.y_channel = AnalogIn(self.ads, ADS.P1)
            
            self.joystick_cal = {
                'x_min': 0.485, 'x_max': 2.883, 'x_center': 1.684,
                'y_min': 0.476, 'y_max': 2.887, 'y_center': 1.681
            }
            
            self.analog_available = True
            self.get_logger().info("Analog joystick initialized")
            
        except Exception as e:
            self.analog_available = False
            self.get_logger().warn(f"Analog joystick unavailable: {e}")
    
    def _init_state(self):
        """Initialize robot state variables"""
        # Pose and velocity
        self.x = self.y = self.theta = 0.0
        self.vx = self.vth = 0.0
        
        # Control state
        self.motors_enabled = False
        self.emergency_stop = False
        self.braking_active = False
        self.joystick_control_active = False
        self.current_speed_scale = 0.5
        self.control_mode = 'nav2'
        
        # Command tracking
        self.target_linear_vel = self.target_angular_vel = 0.0
        self.last_cmd_vel_time = self.get_clock().now()
        
        # Encoder state
        self.last_left_encoder = self.last_right_encoder = 0
        self.first_reading = True
        
        # Wheel state
        self.left_wheel_pos = self.right_wheel_pos = 0.0
        self.left_wheel_vel = self.right_wheel_vel = 0.0
        
        self.last_time = self.get_clock().now()
    
    def _setup_ros_interfaces(self):
        """Setup ROS2 publishers, subscribers, services, and timers"""
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.joy_pub = self.create_publisher(Joy, 'joy', 10)
        self.status_pub = self.create_publisher(String, 'motor_status', 10)
        
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_callback, 10)
        
        # Collision avoidance subscribers
        if self.collision_avoidance_enabled and self.use_laser_scan:
            self.get_logger().info(f"Setting up laser scan subscription to {self.laser_scan_topic}...")
            
            # Set QoS profile to match the scan publisher (usually BEST_EFFORT for sensor data)
            
            
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,  # Match typical sensor QoS
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            
            self.scan_sub = self.create_subscription(
                LaserScan, 
                self.laser_scan_topic, 
                self._scan_callback, 
                qos_profile  # Use the compatible QoS profile
            )
            self.get_logger().info("Laser scan subscriber created with BEST_EFFORT QoS - waiting for scan data...")
            
            # Create a timer to check scan connection
            self.scan_check_timer = self.create_timer(5.0, self._check_scan_connection)
        
        # Services
        self.emergency_stop_srv = self.create_service(Trigger, 'emergency_stop', self._emergency_stop_service)
        self.enable_motors_srv = self.create_service(Trigger, 'enable_motors', self._enable_motors_service)
        self.disable_motors_srv = self.create_service(Trigger, 'disable_motors', self._disable_motors_service)
        
        # Timers
        self.odometry_timer = self.create_timer(1.0 / self.publish_rate, self._update_odometry)
        self.control_timer = self.create_timer(0.05, self._control_loop)
        self.joystick_timer = self.create_timer(1.0 / self.joystick_publish_rate, self._read_gpio_joystick)
    
    def _check_scan_connection(self):
        """Check if laser scan data is being received"""
        if self.last_scan is None:
            self.get_logger().warn(f"No laser scan data received from {self.laser_scan_topic}")
            self.get_logger().warn("Make sure the laser scan topic is being published and ROS_DOMAIN_ID matches")
            self.get_logger().info("You can check with: ros2 topic list | grep scan")
        else:
            self.get_logger().info(f"✓ Laser scan connection OK - {len(self.last_scan.ranges)} points")
            self.get_logger().info(f"  Range: {self.last_scan.range_min:.2f}m to {self.last_scan.range_max:.2f}m")
            self.get_logger().info("Collision avoidance is now active!")
            # Cancel the timer once we have scan data
            self.scan_check_timer.cancel()
    
    def _scan_callback(self, msg):
        """Handle laser scan data for collision detection"""
        self.last_scan = msg
        
        # Debug: Log when scan is first received
        if not hasattr(self, '_scan_received'):
            self._scan_received = True
            self.get_logger().info(f"✓ Laser scan received successfully!")
            self.get_logger().info(f"  Points: {len(msg.ranges)}")
            self.get_logger().info(f"  FOV: {math.degrees(msg.angle_min):.1f}° to {math.degrees(msg.angle_max):.1f}°")
            self.get_logger().info(f"  Range: {msg.range_min:.2f}m to {msg.range_max:.2f}m")
            self.get_logger().info(f"  Frame: {msg.header.frame_id}")
        
        # Debug: Occasional update on scan status
        if not hasattr(self, '_scan_count'):
            self._scan_count = 0
        self._scan_count += 1
        if self._scan_count % 100 == 0:  # Every 100 scans
            self.get_logger().info(f"Scan callback working - received {self._scan_count} scans")
        
        if self.collision_avoidance_enabled:
            self._update_collision_detection_from_scan(msg)
    
    def _update_collision_detection_from_scan(self, scan_msg):
        """Update collision detection using laser scan data"""
        if not scan_msg.ranges:
            return
        
        # Convert to numpy array and clean up invalid data
        ranges = np.array(scan_msg.ranges)
        ranges = np.where(np.isinf(ranges), scan_msg.range_max, ranges)
        ranges = np.where(np.isnan(ranges), scan_msg.range_max, ranges)
        
        total_points = len(ranges)
        if total_points == 0:
            return
        
        # Calculate angular resolution
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        
        # Define sectors based on your front-facing 180° LiDAR
        front_half_angle = math.radians(self.front_sector_angle / 2)  # ±30° for front
        side_angle = math.radians(self.side_sector_angle)  # 45° for sides
        
        # Get sectors for front, left, and right
        front_sector = self._get_sector_ranges_radians(ranges, angle_min, angle_increment, 
                                                     -front_half_angle, front_half_angle)
        left_sector = self._get_sector_ranges_radians(ranges, angle_min, angle_increment, 
                                                    front_half_angle, front_half_angle + side_angle)
        right_sector = self._get_sector_ranges_radians(ranges, angle_min, angle_increment, 
                                                     -front_half_angle - side_angle, -front_half_angle)
        
        # Find minimum distances in each sector
        min_front = np.min(front_sector) if len(front_sector) > 0 else float('inf')
        min_left = np.min(left_sector) if len(left_sector) > 0 else float('inf')
        min_right = np.min(right_sector) if len(right_sector) > 0 else float('inf')
        
        # Update collision detection state
        self.min_obstacle_distance = min(min_front, min_left, min_right)
        previous_collision = self.collision_detected
        self.collision_detected = self.min_obstacle_distance < self.warning_distance
        
        if self.collision_detected:
            # Determine primary collision direction
            if min_front < self.warning_distance:
                self.collision_direction = 'front'
            elif min_left < self.warning_distance:
                self.collision_direction = 'left'
            elif min_right < self.warning_distance:
                self.collision_direction = 'right'
            else:
                self.collision_direction = 'front'  # Default
                
            # Debug: Log collision detection changes
            if not previous_collision:
                self.get_logger().warn(f"🚨 COLLISION DETECTED: {self.collision_direction} at {self.min_obstacle_distance:.2f}m")
        else:
            if previous_collision:
                self.get_logger().info("✓ Collision cleared - path is safe")
            self.collision_direction = None
    
    def _get_sector_ranges_radians(self, ranges, angle_min, angle_increment, start_angle, end_angle):
        """Get range values for a specific angular sector using radians"""
        total_points = len(ranges)
        if total_points == 0:
            return np.array([])
        
        # Convert angles to indices
        start_idx = int((start_angle - angle_min) / angle_increment)
        end_idx = int((end_angle - angle_min) / angle_increment)
        
        # Clamp to valid range
        start_idx = max(0, min(start_idx, total_points - 1))
        end_idx = max(0, min(end_idx, total_points - 1))
        
        if start_idx <= end_idx:
            return ranges[start_idx:end_idx+1]
        else:
            return np.array([])  # Invalid range
    
    def _calculate_collision_avoidance_factor(self, forward_speed, turn_speed):
        """Calculate speed reduction factor based on collision detection"""
        if not self.collision_avoidance_enabled or not self.collision_detected:
            return 1.0, 1.0  # No reduction
        
        # Determine intended motion direction
        motion_direction = None
        if abs(forward_speed) > 0.1:
            motion_direction = 'front' if forward_speed > 0 else 'back'
        
        # Check for turning motion
        turning_direction = None
        if abs(turn_speed) > 0.1:
            turning_direction = 'left' if turn_speed > 0 else 'right'
        
        # Calculate speed reduction based on obstacle direction and intended motion
        linear_factor = 1.0
        angular_factor = 1.0
        
        # Handle forward motion (we only have front-facing lidar)
        if motion_direction == 'front' and self.collision_direction in ['front', 'left', 'right']:
            if self.min_obstacle_distance < self.safety_distance:
                linear_factor = 0.0  # Hard stop
            else:
                # Gradual slowdown
                factor = (self.min_obstacle_distance - self.safety_distance) / (self.warning_distance - self.safety_distance)
                linear_factor = max(self.max_deceleration_factor, min(1.0, factor))
        
        # Handle turning motion
        if turning_direction and self.collision_direction == turning_direction:
            if self.min_obstacle_distance < self.safety_distance:
                angular_factor = 0.0  # Hard stop turning
            else:
                # Gradual slowdown for turning
                factor = (self.min_obstacle_distance - self.safety_distance) / (self.warning_distance - self.safety_distance)
                angular_factor = max(self.max_deceleration_factor, min(1.0, factor))
        
        # General proximity slowdown
        if self.min_obstacle_distance < self.warning_distance:
            proximity_factor = max(0.5, self.min_obstacle_distance / self.warning_distance)
            linear_factor = min(linear_factor, proximity_factor)
            angular_factor = min(angular_factor, proximity_factor)
        
        return linear_factor, angular_factor
    
    def _read_gpio_joystick(self):
        """Read GPIO buttons and analog joystick"""
        try:
            current_time = time.time()
            
            # Read button states (active low)
            button_pressed = {
                name: GPIO.input(pin) == 0
                for name, pin in self.gpio_pins.items()
            }
            
            self._handle_button_logic(current_time, button_pressed)
            
            # Read analog joystick
            x_axis = y_axis = 0.0
            if self.analog_available:
                x_axis = self._read_analog_axis(self.x_channel.voltage, 'x')
                y_axis = self._read_analog_axis(self.y_channel.voltage, 'y')
            
            # Determine control mode
            joystick_active = (self.motors_enabled and 
                             (abs(x_axis) > 0.1 or abs(y_axis) > 0.1 or self.braking_active))
            self.joystick_control_active = joystick_active
            self.control_mode = 'joystick' if joystick_active else 'nav2'
            
            if self.joystick_control_active:
                self._handle_joystick_control(x_axis, y_axis)
            
            self._publish_joy_message(x_axis, y_axis, button_pressed)
            
        except Exception as e:
            self.get_logger().error(f"GPIO joystick error: {e}")
    
    def _handle_button_logic(self, current_time, pressed):
        """Handle 3-button control logic"""
        # Speed up button
        if pressed['speed_up'] and self._button_debounced('speed_up', current_time):
            self.current_speed_scale = min(1.0, self.current_speed_scale + self.speed_scale_increment)
            self.get_logger().info(f"Speed increased: {self.current_speed_scale:.1f}")
        
        # Speed down / disable button
        if pressed['speed_down']:
            if self.button_hold_times['speed_down'] is None:
                self.button_hold_times['speed_down'] = current_time
            
            hold_duration = current_time - self.button_hold_times['speed_down']
            if hold_duration >= self.hold_time_disable and self.motors_enabled:
                self.get_logger().warn(f"Motors disabled (held {hold_duration:.1f}s)")
                self._disable_motors()
                self.button_hold_times['speed_down'] = None
        else:
            if self.button_hold_times['speed_down'] is not None:
                hold_duration = current_time - self.button_hold_times['speed_down']
                if (hold_duration < self.hold_time_disable and 
                    self._button_debounced('speed_down', current_time)):
                    self.current_speed_scale = max(0.1, self.current_speed_scale - self.speed_scale_increment)
                    self.get_logger().info(f"Speed decreased: {self.current_speed_scale:.1f}")
                self.button_hold_times['speed_down'] = None
        
        # Brake / enable button
        if pressed['brake']:
            self.braking_active = True
            if self.button_hold_times['brake'] is None:
                self.button_hold_times['brake'] = current_time
            
            hold_duration = current_time - self.button_hold_times['brake']
            if hold_duration >= self.hold_time_enable and not self.motors_enabled:
                self.get_logger().info(f"Motors enabled (held {hold_duration:.1f}s)")
                self._enable_motors()
                self.button_hold_times['brake'] = None
        else:
            self.braking_active = False
            self.button_hold_times['brake'] = None
    
    def _button_debounced(self, button_name, current_time):
        """Check if button press is debounced"""
        if (button_name not in self.last_button_press or 
            current_time - self.last_button_press[button_name] > self.button_debounce_time):
            self.last_button_press[button_name] = current_time
            return True
        return False
    
    def _read_analog_axis(self, voltage, axis):
        """Convert voltage to normalized axis value"""
        cal = self.joystick_cal
        if axis == 'x':
            normalized = (voltage - cal['x_min']) / (cal['x_max'] - cal['x_min'])
        else:
            normalized = (voltage - cal['y_min']) / (cal['y_max'] - cal['y_min'])
        
        axis_value = max(-1.0, min(1.0, (normalized * 2.0) - 1.0))
        
        if abs(axis_value) < self.joystick_deadzone:
            return 0.0
        
        sign = 1 if axis_value > 0 else -1
        return sign * (abs(axis_value) - self.joystick_deadzone) / (1.0 - self.joystick_deadzone)
    
    def _handle_joystick_control(self, x_axis, y_axis):
        """Handle direct joystick motor control with collision avoidance"""
        if self.emergency_stop or not self.motors_enabled:
            return
        
        # Tank drive calculation
        forward = y_axis * self.current_speed_scale  * 0.5
        turn = x_axis * self.current_speed_scale * 0.5
        

        
        # Apply collision avoidance
        if self.collision_avoidance_enabled and self.last_scan is not None:
            linear_factor, angular_factor = self._calculate_collision_avoidance_factor(forward, turn)
            forward *= linear_factor
            turn *= angular_factor
            
            # Log collision warnings with clear feedback
            if self.collision_detected and self.min_obstacle_distance < self.safety_distance:
                self.get_logger().warn(f"🛑 STOPPING: Obstacle at {self.min_obstacle_distance:.2f}m ({self.collision_direction})")
            elif self.collision_detected:
                reduction = int((1.0 - min(linear_factor, angular_factor)) * 100)
                self.get_logger().warn(f"⚠️  SLOWING: {reduction}% speed reduction - obstacle at {self.min_obstacle_distance:.2f}m ({self.collision_direction})")
        elif self.collision_avoidance_enabled and self.last_scan is None:
            # Warning if laser scan not available
            if not hasattr(self, '_scan_warning_shown'):
                self._scan_warning_shown = True
                self.get_logger().warn(f"⚠️  Laser scan not available on {self.laser_scan_topic} - collision avoidance disabled!")
        
        left_speed = forward + turn
        right_speed = forward - turn
        
        # Normalize speeds
        max_magnitude = max(abs(left_speed), abs(right_speed))
        if max_magnitude > 1.0:
            left_speed /= max_magnitude
            right_speed /= max_magnitude
        
        # Convert to RPM and apply braking
        left_rpm = -left_speed * self.max_joystick_speed * 0.5
        right_rpm = right_speed * self.max_joystick_speed * 0.5
        
        if self.braking_active:
            left_rpm *= 0.3
            right_rpm *= 0.3
        
        try:
            self.motor_left.set_target_velocity_rpm(int(left_rpm))
            self.motor_right.set_target_velocity_rpm(int(right_rpm))
        except Exception as e:
            self.get_logger().error(f"Failed to set joystick motor speeds: {e}")
    
    def _publish_joy_message(self, x_axis, y_axis, button_pressed):
        """Publish Joy message"""
        joy_msg = Joy()
        joy_msg.header.stamp = self.get_clock().now().to_msg()
        joy_msg.header.frame_id = 'joystick'
        joy_msg.axes = [x_axis, y_axis, 0.0, 0.0]
        joy_msg.buttons = [
            int(button_pressed['speed_up']),
            int(button_pressed['speed_down']),
            int(button_pressed['brake']),
            int(self.motors_enabled),
            int(self.joystick_control_active),
            int(self.collision_detected)  # Added collision detection status
        ]
        self.joy_pub.publish(joy_msg)
    
    def _cmd_vel_callback(self, msg):
        """Handle Nav2 velocity commands"""
        if self.emergency_stop or self.joystick_control_active:
            return
        
        self.target_linear_vel = max(-self.max_linear_vel, 
                                   min(self.max_linear_vel, msg.linear.x))
        self.target_angular_vel = max(-self.max_angular_vel, 
                                    min(self.max_angular_vel, msg.angular.z))
        self.last_cmd_vel_time = self.get_clock().now()
    
    def _control_loop(self):
        """Main control loop for Nav2 commands"""
        if not self.motors_enabled or self.emergency_stop or self.joystick_control_active:
            return
        
        # Check command timeout
        time_since_cmd = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds / 1e9
        if time_since_cmd > self.cmd_vel_timeout:
            self.target_linear_vel = self.target_angular_vel = 0.0
        
        # Calculate differential drive velocities
        v_left = self.target_linear_vel - (self.target_angular_vel * self.wheel_base) / 2.0
        v_right = self.target_linear_vel + (self.target_angular_vel * self.wheel_base) / 2.0
        
        # Convert to RPM
        rpm_left = -(v_left / self.wheel_circumference) * 60.0
        rpm_right = (v_right / self.wheel_circumference) * 60.0
        
        try:
            self.motor_left.set_target_velocity_rpm(rpm_left)
            self.motor_right.set_target_velocity_rpm(rpm_right)
        except Exception as e:
            self.get_logger().error(f"Failed to set Nav2 motor speeds: {e}")
    
    def _update_odometry(self):
        """Update and publish odometry information"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        if dt <= 0 or not self.motors_enabled:
            return
        
        try:
            left_encoder = self.motor_left.get_actual_position()
            right_encoder = self.motor_right.get_actual_position()
            
            if left_encoder is None or right_encoder is None:
                return
            
            if self.first_reading:
                self.last_left_encoder = left_encoder
                self.last_right_encoder = right_encoder
                self.first_reading = False
                self.last_time = current_time
                return
            
            # Calculate encoder deltas with overflow handling
            delta_left = self._handle_encoder_overflow(left_encoder - self.last_left_encoder)
            delta_right = self._handle_encoder_overflow(right_encoder - self.last_right_encoder)
            
            # Convert to distances
            delta_left_dist = -delta_left * self.meters_per_count
            delta_right_dist = delta_right * self.meters_per_count
            
            # Calculate robot motion
            delta_distance = (delta_left_dist + delta_right_dist) / 2.0
            delta_theta = (delta_right_dist - delta_left_dist) / self.wheel_base
            
            # Update pose using differential drive kinematics
            if abs(delta_theta) < 1e-6:
                self.x += delta_distance * math.cos(self.theta)
                self.y += delta_distance * math.sin(self.theta)
            else:
                radius = delta_distance / delta_theta
                self.x += radius * (math.sin(self.theta + delta_theta) - math.sin(self.theta))
                self.y += radius * (-math.cos(self.theta + delta_theta) + math.cos(self.theta))
            
            self.theta += delta_theta
            self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
            
            # Calculate velocities
            self.vx = delta_distance / dt
            self.vth = delta_theta / dt
            
            # Update wheel states
            self.left_wheel_vel = delta_left_dist / dt
            self.right_wheel_vel = delta_right_dist / dt
            self.left_wheel_pos += delta_left_dist / self.wheel_radius
            self.right_wheel_pos += delta_right_dist / self.wheel_radius
            
            # Store current encoder values
            self.last_left_encoder = left_encoder
            self.last_right_encoder = right_encoder
            
        except Exception as e:
            self.get_logger().error(f"Encoder read error: {e}")
            return
        
        # Publish all data
        self._publish_odometry(current_time)
        self._publish_joint_states(current_time)
        if self.publish_tf:
            self._publish_transform(current_time)
        self._publish_status()
        
        self.last_time = current_time
    
    def _handle_encoder_overflow(self, delta):
        """Handle encoder counter overflow"""
        MAX_ENCODER = 2147483647
        MIN_ENCODER = -2147483648
        
        if abs(delta) > MAX_ENCODER:
            if delta > 0:
                delta -= (MAX_ENCODER - MIN_ENCODER + 1)
            else:
                delta += (MAX_ENCODER - MIN_ENCODER + 1)
        
        return delta
    
    def _publish_odometry(self, current_time):
        """Publish odometry message"""
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id
        
        # Position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        
        # Orientation
        q = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        # Velocity
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.angular.z = self.vth
        
        # Covariances
        odom.pose.covariance = [0.0] * 36
        odom.twist.covariance = [0.0] * 36
        
        for i, val in enumerate(self.pose_covariance_diagonal):
            odom.pose.covariance[i * 7] = val
        for i, val in enumerate(self.twist_covariance_diagonal):
            odom.twist.covariance[i * 7] = val
        
        self.odom_pub.publish(odom)
    
    def _publish_joint_states(self, current_time):
        """Publish joint states"""
        joint_state = JointState()
        joint_state.header.stamp = current_time.to_msg()
        joint_state.name = ['left_wheel_joint', 'right_wheel_joint']
        joint_state.position = [self.left_wheel_pos, self.right_wheel_pos]
        joint_state.velocity = [
            self.left_wheel_vel / self.wheel_radius, 
            self.right_wheel_vel / self.wheel_radius
        ]
        self.joint_state_pub.publish(joint_state)
    
    def _publish_transform(self, current_time):
        """Publish TF transform"""
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.base_frame_id
        
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        
        q = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(t)
    
    def _publish_status(self):
        """Publish motor status"""
        mode = "JOYSTICK" if self.joystick_control_active else "NAV2"
        enabled = "ON" if self.motors_enabled else "OFF"
        estop = "ESTOP" if self.emergency_stop else "OK"
        brake = " BRAKE" if self.braking_active else ""
        analog = " ANALOG" if self.analog_available else " BUTTONS"
        collision = " COLLISION" if self.collision_detected else ""
        
        if self.collision_detected:
            collision_info = f" [{self.collision_direction}:{self.min_obstacle_distance:.2f}m]"
        else:
            collision_info = ""
        
        status = f"{mode} | Motors: {enabled} | {estop} | Speed: {self.current_speed_scale:.1f}{analog}{brake}{collision}{collision_info}"
        
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)
    
    def _initialize_motors(self):
        """Initialize motor hardware"""
        try:
            self.get_logger().info("Initializing motors...")
            
            if not (self.motor_left.initialize() and self.motor_right.initialize()):
                self.get_logger().error("Motor initialization failed")
                return
            
            # Configure motor parameters
            for motor in [self.motor_left, self.motor_right]:
                motor.set_acceleration(0.5)
                motor.set_deceleration(0.5)
            
            self.first_reading = True
            self.get_logger().info("Motors initialized (disabled)")
            
        except Exception as e:
            self.get_logger().error(f"Motor initialization failed: {e}")
    
    def _enable_motors(self):
        """Enable motor operation"""
        try:
            if self.motor_left.enable() and self.motor_right.enable():
                self.motors_enabled = True
                self.emergency_stop = False
                self.get_logger().info("Motors enabled")
                return True
            else:
                self.get_logger().error("Failed to enable motors")
                return False
        except Exception as e:
            self.get_logger().error(f"Error enabling motors: {e}")
            return False
    
    def _disable_motors(self):
        """Disable motor operation"""
        try:
            # Stop motors first
            self.motor_left.set_target_velocity_rpm(0)
            self.motor_right.set_target_velocity_rpm(0)
            time.sleep(0.2)
            
            if self.motor_left.disable() and self.motor_right.disable():
                self.motors_enabled = False
                self.joystick_control_active = False
                self.control_mode = 'nav2'
                self.get_logger().info("Motors disabled")
                return True
            else:
                self.get_logger().error("Failed to disable motors")
                return False
        except Exception as e:
            self.get_logger().error(f"Error disabling motors: {e}")
            return False
    
    def _emergency_stop_service(self, request, response):
        """Emergency stop service handler"""
        self.emergency_stop = True
        self.target_linear_vel = self.target_angular_vel = 0.0
        
        try:
            self.motor_left.set_target_velocity_rpm(0)
            self.motor_right.set_target_velocity_rpm(0)
        except:
            pass
        
        self.get_logger().warn("Emergency stop activated")
        response.success = True
        response.message = "Emergency stop activated"
        return response
    
    def _enable_motors_service(self, request, response):
        """Enable motors service handler"""
        success = self._enable_motors()
        response.success = success
        response.message = "Motors enabled" if success else "Failed to enable motors"
        return response
    
    def _disable_motors_service(self, request, response):
        """Disable motors service handler"""
        success = self._disable_motors()
        response.success = success
        response.message = "Motors disabled" if success else "Failed to disable motors"
        return response
    
    def shutdown(self):
        """Clean shutdown procedure"""
        self.get_logger().info("Shutting down motor controller...")
        
        if self.motors_enabled:
            try:
                self.motor_left.set_target_velocity_rpm(0)
                self.motor_right.set_target_velocity_rpm(0)
                time.sleep(0.5)
                self.motor_left.disable()
                self.motor_right.disable()
            except:
                pass
        
        try:
            GPIO.cleanup()
        except:
            pass
        
        try:
            L2DBMotorDriver.close_bus()
        except:
            pass
        
        self.get_logger().info("Shutdown complete")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = JetsonMotorController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node:
            node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()