#!/usr/bin/env python3
"""
Jetson Motor Control Node with Integrated Collision Avoidance
ROS2 node for differential drive robot control with GPIO interface and lidar safety
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
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


class JetsonMotorController(Node):
    
    def __init__(self):
        super().__init__('jetson_motor_controller')
        
        self._declare_params()
        self._get_params()
        self._init_hardware()
        self._init_state()
        self._init_collision_avoidance()
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
        self.declare_parameter('max_joystick_speed', 50)
        
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
        
        # Frame configuration
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('publish_tf', True)
        
        # Covariance matrices
        self.declare_parameter('pose_covariance_diagonal', [0.01, 0.01, 0.001, 0.001, 0.001, 0.09])
        self.declare_parameter('twist_covariance_diagonal', [0.002, 0.001, 0.001, 0.001, 0.001, 0.02])
        
        # COLLISION AVOIDANCE PARAMETERS
        self.declare_parameter('collision_avoidance_enabled', True)
        self.declare_parameter('wheelchair_length', 1.0)
        self.declare_parameter('wheelchair_width', 0.65)
        self.declare_parameter('safety_margin', 0.15)
        self.declare_parameter('lidar_x_offset', 0.4)
        self.declare_parameter('min_detection_range', 0.1)
        self.declare_parameter('max_detection_range', 3.0)
        self.declare_parameter('critical_distance', 0.8)
        self.declare_parameter('warning_distance', 1.2)
        self.declare_parameter('min_movement_threshold', 0.1)
        self.declare_parameter('scan_timeout', 1.0)
        self.declare_parameter('collision_debug', True)
    
    def _get_params(self):
        """Get parameter values and calculate derived constants"""
        # Original parameters
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
        
        # COLLISION AVOIDANCE PARAMETERS
        self.collision_avoidance_enabled = self.get_parameter('collision_avoidance_enabled').value
        self.wheelchair_length = self.get_parameter('wheelchair_length').value
        self.wheelchair_width = self.get_parameter('wheelchair_width').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.lidar_x_offset = self.get_parameter('lidar_x_offset').value
        self.min_detection_range = self.get_parameter('min_detection_range').value
        self.max_detection_range = self.get_parameter('max_detection_range').value
        self.critical_distance = self.get_parameter('critical_distance').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.min_movement_threshold = self.get_parameter('min_movement_threshold').value
        self.scan_timeout = self.get_parameter('scan_timeout').value
        self.collision_debug = self.get_parameter('collision_debug').value
        
        # Calculate wheelchair footprint bounds from base_link
        self.front_bound = self.wheelchair_length / 2.0 + self.safety_margin
        self.rear_bound = -self.wheelchair_length / 2.0 - self.safety_margin
        self.left_bound = self.wheelchair_width / 2.0 + self.safety_margin
        self.right_bound = -self.wheelchair_width / 2.0 - self.safety_margin
    
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
    
    def _init_collision_avoidance(self):
        """Initialize collision avoidance system"""
        self.last_scan = None
        self.last_scan_time = None
        
        # Sector distances for obstacle detection
        self.sector_distances = {
            'front': float('inf'),
            'front_left': float('inf'),
            'front_right': float('inf'),
            'left': float('inf'),
            'right': float('inf')
        }
        
        self.get_logger().info(f"Collision avoidance {'ENABLED' if self.collision_avoidance_enabled else 'DISABLED'}")
    
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
        
        # COLLISION AVOIDANCE: Subscribe to lidar
        if self.collision_avoidance_enabled:
            lidar_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_callback, lidar_qos)
            
            if self.collision_debug:
                self.collision_debug_pub = self.create_publisher(String, 'collision_debug', 10)
        
        # Services
        self.emergency_stop_srv = self.create_service(Trigger, 'emergency_stop', self._emergency_stop_service)
        self.enable_motors_srv = self.create_service(Trigger, 'enable_motors', self._enable_motors_service)
        self.disable_motors_srv = self.create_service(Trigger, 'disable_motors', self._disable_motors_service)
        
        # Timers
        self.odometry_timer = self.create_timer(1.0 / self.publish_rate, self._update_odometry)
        self.control_timer = self.create_timer(0.05, self._control_loop)
        self.joystick_timer = self.create_timer(1.0 / self.joystick_publish_rate, self._read_gpio_joystick)
    
    def _scan_callback(self, msg):
        """Process incoming laser scan data for collision avoidance"""
        if not self.collision_avoidance_enabled:
            return
        
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()
        
        # Update obstacle sectors
        self._update_obstacle_sectors(msg)
    
    def _update_obstacle_sectors(self, scan):
        """Analyze scan data and update sector distances"""
        if len(scan.ranges) == 0:
            return
        
        # Reset sectors
        for sector in self.sector_distances:
            self.sector_distances[sector] = float('inf')
        
        angle_increment = scan.angle_increment
        angle_min = scan.angle_min
        
        for i, range_val in enumerate(scan.ranges):
            if not (self.min_detection_range <= range_val <= self.max_detection_range):
                continue
            
            # Calculate angle relative to lidar frame
            angle = angle_min + (i * angle_increment)
            
            # Convert to base_link coordinates considering lidar offset
            x_lidar = range_val * math.cos(angle)
            y_lidar = range_val * math.sin(angle)
            
            # Transform from lidar frame to base_link frame
            x_base = x_lidar + self.lidar_x_offset
            y_base = y_lidar
            
            # Skip points inside wheelchair footprint
            if self._is_inside_footprint(x_base, y_base):
                continue
            
            # Calculate distance from base_link
            distance_from_base = math.sqrt(x_base**2 + y_base**2)
            
            # Determine which sector this obstacle belongs to
            sector = self._get_obstacle_sector(x_base, y_base)
            if sector and distance_from_base < self.sector_distances[sector]:
                self.sector_distances[sector] = distance_from_base
    
    def _is_inside_footprint(self, x, y):
        """Check if point is inside wheelchair footprint"""
        return (self.rear_bound < x < self.front_bound and 
                self.right_bound < y < self.left_bound)
    
    def _get_obstacle_sector(self, x, y):
        """Determine which sector a point belongs to - CORRECTED for 270° lidar"""
        # Calculate angle from base_link to obstacle
        angle_from_base = math.atan2(y, x)  # y=left/right, x=front/back from base_link
        
        # Convert to degrees for easier logic
        angle_deg = math.degrees(angle_from_base)
        
        # Lidar covers 270 degrees - blind zone is rear (around 180°)
        # Front is 0°, Left is +90°, Right is -90°
        
        # Define sectors for 270° coverage (no rear detection)
        if -45 <= angle_deg <= 45:
            return 'front'           # Front sector
        elif 45 < angle_deg <= 90:
            return 'front_left'      # Front-left sector  
        elif 90 < angle_deg <= 135:
            return 'left'            # Left side sector
        elif -90 <= angle_deg < -45:
            return 'front_right'     # Front-right sector
        elif -135 <= angle_deg < -90:
            return 'right'           # Right side sector
        else:
            # Blind zone (135° to 180° and -135° to -180°) - rear of wheelchair
            return None
    
    def _is_scan_data_valid(self):
        """Check if scan data is recent and valid"""
        if not self.collision_avoidance_enabled or self.last_scan is None or self.last_scan_time is None:
            return False
        
        time_diff = (self.get_clock().now() - self.last_scan_time).nanoseconds / 1e9
        return time_diff < self.scan_timeout
    
    def _apply_collision_filter(self, x_input, y_input):
        """Apply collision avoidance - CORRECTED for your actual coordinate system"""
        # If collision avoidance disabled or no valid scan data, return original inputs
        if not self.collision_avoidance_enabled or not self._is_scan_data_valid():
            return x_input, y_input
        
        filtered_x = x_input  # RIGHT (+) / LEFT (-)  <- CORRECTED!
        filtered_y = y_input  # FORWARD (+) / BACKWARD (-)
        
        # Skip filtering for very small inputs
        if abs(x_input) < self.min_movement_threshold and abs(y_input) < self.min_movement_threshold:
            return filtered_x, filtered_y
        
        # FORWARD movement filtering (y_input > 0) - check FRONT sectors
        if y_input > self.min_movement_threshold:
            front_distance = min(
                self.sector_distances.get('front', float('inf')),
                self.sector_distances.get('front_left', float('inf')),
                self.sector_distances.get('front_right', float('inf'))
            )
            
            if front_distance < self.critical_distance:
                filtered_y = 0.0
                if self.collision_debug:
                    self.get_logger().warn(f"🚨 FORWARD BLOCKED - front obstacle at {front_distance:.2f}m")
            elif front_distance < self.warning_distance:
                reduction_factor = max(0.3, (front_distance - self.critical_distance) / (self.warning_distance - self.critical_distance))
                filtered_y *= reduction_factor
                if self.collision_debug:
                    self.get_logger().info(f"⚠️ FORWARD REDUCED to {reduction_factor:.2f} - front obstacle at {front_distance:.2f}m")
        
        # BACKWARD movement (y_input < 0) - NO FILTERING (blind zone)
        # Always allow backward movement since lidar can't see behind
        
        # RIGHT movement filtering (x_input > 0) - check RIGHT sectors  
        if x_input > self.min_movement_threshold:
            right_distance = min(
                self.sector_distances.get('right', float('inf')),
                self.sector_distances.get('front_right', float('inf'))
            )
            
            if right_distance < self.critical_distance:
                filtered_x = 0.0
                if self.collision_debug:
                    self.get_logger().warn(f"🚨 RIGHT BLOCKED - right obstacle at {right_distance:.2f}m")
            elif right_distance < self.warning_distance:
                reduction_factor = max(0.3, (right_distance - self.critical_distance) / (self.warning_distance - self.critical_distance))
                filtered_x *= reduction_factor
                if self.collision_debug:
                    self.get_logger().info(f"⚠️ RIGHT REDUCED to {reduction_factor:.2f} - right obstacle at {right_distance:.2f}m")
        
        # LEFT movement filtering (x_input < 0) - check LEFT sectors
        elif x_input < -self.min_movement_threshold:
            left_distance = min(
                self.sector_distances.get('left', float('inf')),
                self.sector_distances.get('front_left', float('inf'))
            )
            
            if left_distance < self.critical_distance:
                filtered_x = 0.0
                if self.collision_debug:
                    self.get_logger().warn(f"🚨 LEFT BLOCKED - left obstacle at {left_distance:.2f}m")
            elif left_distance < self.warning_distance:
                reduction_factor = max(0.3, (left_distance - self.critical_distance) / (self.warning_distance - self.critical_distance))
                filtered_x *= reduction_factor
                if self.collision_debug:
                    self.get_logger().info(f"⚠️ LEFT REDUCED to {reduction_factor:.2f} - left obstacle at {left_distance:.2f}m")
        
        # Debug output if filtering occurred
        if (filtered_x != x_input or filtered_y != y_input) and self.collision_debug:
            self.get_logger().info(f"COLLISION FILTER: Input({x_input:.2f},{y_input:.2f}) -> Output({filtered_x:.2f},{filtered_y:.2f})")
        
        return filtered_x, filtered_y
    
    def _read_gpio_joystick(self):
        """Read GPIO buttons and analog joystick with collision avoidance"""
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
            
            # APPLY COLLISION AVOIDANCE FILTERING
            x_axis_safe, y_axis_safe = self._apply_collision_filter(x_axis, y_axis)
            
            # Determine control mode
            joystick_active = (self.motors_enabled and 
                             (abs(x_axis_safe) > 0.1 or abs(y_axis_safe) > 0.1 or self.braking_active))
            self.joystick_control_active = joystick_active
            self.control_mode = 'joystick' if joystick_active else 'nav2'
            
            if self.joystick_control_active:
                # Use FILTERED values for motor control
                self._handle_joystick_control(x_axis_safe, y_axis_safe)
            
            # Publish joy message (with original values for monitoring)
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
        """Handle direct joystick motor control with inverted backward direction"""
        if self.emergency_stop or not self.motors_enabled:
            return
        
        # Tank drive calculation
        forward = y_axis * self.current_speed_scale
        turn = x_axis * self.current_speed_scale
        
        # Invert turning direction when moving backwards
        if forward < 0:
            turn = -turn
        
        left_speed = forward + turn
        right_speed = forward - turn
        
        # Normalize speeds
        max_magnitude = max(abs(left_speed), abs(right_speed))
        if max_magnitude > 1.0:
            left_speed /= max_magnitude
            right_speed /= max_magnitude
        
        # Convert to RPM and apply braking
        left_rpm = -left_speed * self.max_joystick_speed
        right_rpm = right_speed * self.max_joystick_speed
        
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
            int(self.joystick_control_active)
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
        """Publish motor status with collision avoidance info"""
        mode = "JOYSTICK" if self.joystick_control_active else "NAV2"
        enabled = "ON" if self.motors_enabled else "OFF"
        estop = "ESTOP" if self.emergency_stop else "OK"
        brake = " BRAKE" if self.braking_active else ""
        analog = " ANALOG" if self.analog_available else " BUTTONS"
        
        # Add collision avoidance status
        collision_status = ""
        if self.collision_avoidance_enabled:
            if self._is_scan_data_valid():
                # Count obstacles in sectors
                obstacles = [f"{sector}:{dist:.1f}m" for sector, dist in self.sector_distances.items() 
                           if dist < self.warning_distance]
                
                if any(dist < self.critical_distance for dist in self.sector_distances.values()):
                    collision_status = f" COLLISION-STOP({len(obstacles)})"
                elif obstacles:
                    collision_status = f" COLLISION-WARN({len(obstacles)})"
                else:
                    collision_status = " COLLISION-OK"
            else:
                collision_status = " COLLISION-NO_SCAN"
        else:
            collision_status = " COLLISION-DISABLED"
        
        status = f"{mode} | Motors: {enabled} | {estop} | Speed: {self.current_speed_scale:.1f}{analog}{brake}{collision_status}"
        
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