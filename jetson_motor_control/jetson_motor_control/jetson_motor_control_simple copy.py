#!/usr/bin/env python3
"""
Integrated Jetson Motor Control with Robust Collision Avoidance
Properly handles wheelchair geometry and lidar positioning
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Joy, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from tf2_ros import TransformBroadcaster
import tf_transformations
import math
import time
import threading
import numpy as np
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import Jetson.GPIO as GPIO
from jetson_motor_control.l2db_motor_driver import L2DBMotorDriver


class IntegratedSafeMotorController(Node):
    
    def __init__(self):
        super().__init__('integrated_safe_motor_controller')
        
        self._declare_params()
        self._get_params()
        self._init_hardware()
        self._init_state()
        self._init_collision_system()
        self._setup_ros_interfaces()
        
        self.init_thread = threading.Thread(target=self._initialize_motors, daemon=True)
        self.init_thread.start()
        
        self.get_logger().info("Integrated Safe Motor Controller initialized")
    
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
        
        # Wheelchair geometry (CRITICAL FOR COLLISION DETECTION)
        self.declare_parameter('base_to_lidar_x', 0.54)   # Distance from base to lidar
        self.declare_parameter('lidar_to_front_x', 0.25)   # Distance from lidar to front
        self.declare_parameter('base_to_rear_x', 0.25)    # Distance from base to rear
        self.declare_parameter('wheelchair_width', 0.65)   # Total width
        
        # Collision avoidance parameters
        self.declare_parameter('use_collision_avoidance', True)
        self.declare_parameter('safety_margin', 0.1)       # Buffer around footprint
        self.declare_parameter('critical_distance', 0.15)  # Emergency stop
        self.declare_parameter('safety_distance', 0.25)    # Major slowdown
        self.declare_parameter('slowdown_distance', 0.5)   # Start gentle slowdown
        self.declare_parameter('side_safety_buffer', 0.15) # Extra buffer for sides
        self.declare_parameter('publish_markers', True)
        
        # Frame configuration
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('publish_tf', True)
        
        # Covariance matrices
        self.declare_parameter('pose_covariance_diagonal', [0.01, 0.01, 0.001, 0.001, 0.001, 0.09])
        self.declare_parameter('twist_covariance_diagonal', [0.002, 0.001, 0.001, 0.001, 0.001, 0.02])
    
    def _get_params(self):
        """Get parameter values and calculate derived constants"""
        # Motor parameters
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
        
        # Control parameters
        self.hold_time_enable = self.get_parameter('hold_time_enable_motors').value
        self.hold_time_disable = self.get_parameter('hold_time_disable_motors').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.joystick_publish_rate = self.get_parameter('joystick_publish_rate').value
        self.joystick_deadzone = self.get_parameter('joystick_deadzone').value
        self.speed_scale_increment = self.get_parameter('speed_scale_increment').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.button_debounce_time = self.get_parameter('button_debounce_time').value
        
        # Wheelchair geometry
        self.base_to_lidar_x = self.get_parameter('base_to_lidar_x').value
        self.lidar_to_front_x = self.get_parameter('lidar_to_front_x').value
        self.base_to_rear_x = self.get_parameter('base_to_rear_x').value
        self.wheelchair_width = self.get_parameter('wheelchair_width').value
        
        # Collision parameters
        self.use_collision_avoidance = self.get_parameter('use_collision_avoidance').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.critical_distance = self.get_parameter('critical_distance').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.slowdown_distance = self.get_parameter('slowdown_distance').value
        self.side_safety_buffer = self.get_parameter('side_safety_buffer').value
        self.publish_markers = self.get_parameter('publish_markers').value
        
        # Frame parameters
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
        
        # Calculate wheelchair boundaries in base_link frame
        self.front_edge_x = self.base_to_lidar_x + self.lidar_to_front_x
        self.rear_edge_x = -self.base_to_rear_x
        self.half_width = self.wheelchair_width / 2.0
        
        self.get_logger().info(f"=== Wheelchair Geometry ===")
        self.get_logger().info(f"Front edge (from base): {self.front_edge_x:.2f}m")
        self.get_logger().info(f"Rear edge (from base): {self.rear_edge_x:.2f}m")
        self.get_logger().info(f"Width: {self.wheelchair_width:.2f}m")
        self.get_logger().info(f"Lidar position (from base): {self.base_to_lidar_x:.2f}m forward")
    
    def _init_collision_system(self):
        """Initialize collision avoidance system"""
        self.latest_scan = None
        self.obstacles_in_base_frame = []
        self.collision_zones = {
            'front': [],
            'front_left': [],
            'front_right': [],
            'left': [],
            'right': [],
            'rear': [],
            'rear_left': [],
            'rear_right': []
        }
        self.collision_warning = False
        self.last_collision_check = {'linear_scale': 1.0, 'angular_scale': 1.0}
        
        # For smooth obstacle tracking
        self.obstacle_history = []
        self.history_size = 3  # Keep last 3 scans
        self.min_obstacle_persistence = 2  # Obstacle must appear in at least 2 scans
    
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
        
        # Joystick values
        self.raw_joystick_x = 0.0
        self.raw_joystick_y = 0.0
        self.filtered_joystick_x = 0.0
        self.filtered_joystick_y = 0.0
        
        # Encoder state
        self.last_left_encoder = self.last_right_encoder = 0
        self.first_reading = True
        
        # Wheel state
        self.left_wheel_pos = self.right_wheel_pos = 0.0
        self.left_wheel_vel = self.right_wheel_vel = 0.0
        
        self.last_time = self.get_clock().now()
    
    def _setup_ros_interfaces(self):
        """Setup ROS2 publishers, subscribers, services, and timers"""
        # QoS profile for lidar
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.joy_pub = self.create_publisher(Joy, 'joy', 10)
        self.status_pub = self.create_publisher(String, 'motor_status', 10)
        
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        
        if self.publish_markers and self.use_collision_avoidance:
            self.marker_pub = self.create_publisher(MarkerArray, 'collision_zones', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_callback, 10)
        
        if self.use_collision_avoidance:
            self.scan_sub = self.create_subscription(
                LaserScan,
                '/scan',
                self._scan_callback,
                qos_profile
            )
        
        # Services
        self.emergency_stop_srv = self.create_service(Trigger, 'emergency_stop', self._emergency_stop_service)
        self.enable_motors_srv = self.create_service(Trigger, 'enable_motors', self._enable_motors_service)
        self.disable_motors_srv = self.create_service(Trigger, 'disable_motors', self._disable_motors_service)
        
        # Timers
        self.odometry_timer = self.create_timer(1.0 / self.publish_rate, self._update_odometry)
        self.control_timer = self.create_timer(0.05, self._control_loop)
        self.joystick_timer = self.create_timer(1.0 / self.joystick_publish_rate, self._read_gpio_joystick)
    
    def _transform_lidar_to_base(self, x_lidar, y_lidar):
        """Transform point from lidar frame to base_link frame"""
        x_base = x_lidar + self.base_to_lidar_x
        y_base = y_lidar
        return x_base, y_base
    
    def _is_point_in_footprint(self, x_base, y_base):
        """Check if point is inside wheelchair footprint with margin"""
        margin = self.safety_margin
        
        if (self.rear_edge_x - margin <= x_base <= self.front_edge_x + margin and
            -self.half_width - margin <= y_base <= self.half_width + margin):
            return True
        return False
    
    def _calculate_distance_to_footprint_edge(self, x_base, y_base):
        """Calculate minimum distance from point to wheelchair edge"""
        # Distance to front
        if x_base > self.front_edge_x:
            dist_front = x_base - self.front_edge_x
        else:
            dist_front = 0
        
        # Distance to rear
        if x_base < self.rear_edge_x:
            dist_rear = self.rear_edge_x - x_base
        else:
            dist_rear = 0
        
        # Distance to left
        if y_base > self.half_width:
            dist_left = y_base - self.half_width
        else:
            dist_left = 0
        
        # Distance to right
        if y_base < -self.half_width:
            dist_right = -self.half_width - y_base
        else:
            dist_right = 0
        
        # If outside in both dimensions, use diagonal distance
        if (dist_front > 0 or dist_rear > 0) and (dist_left > 0 or dist_right > 0):
            x_dist = max(dist_front, dist_rear)
            y_dist = max(dist_left, dist_right)
            return math.sqrt(x_dist**2 + y_dist**2)
        
        # Otherwise use the maximum single dimension distance
        return max(dist_front, dist_rear, dist_left, dist_right)
    
    def _classify_obstacle_zone(self, x_base, y_base):
        """Classify obstacle into detailed zones"""
        zones = []
        
        # Determine primary zones
        front_threshold = self.base_to_lidar_x  # Use lidar position as reference
        rear_threshold = 0.0
        
        # Front zones
        if x_base > front_threshold:
            if y_base > self.half_width * 0.5:
                zones.append('front_left')
            elif y_base < -self.half_width * 0.5:
                zones.append('front_right')
            else:
                zones.append('front')
        
        # Rear zones
        elif x_base < rear_threshold:
            if y_base > self.half_width * 0.5:
                zones.append('rear_left')
            elif y_base < -self.half_width * 0.5:
                zones.append('rear_right')
            else:
                zones.append('rear')
        
        # Side zones
        if y_base > self.half_width:
            zones.append('left')
        elif y_base < -self.half_width:
            zones.append('right')
        
        # Default to closest zone if none assigned
        if not zones:
            if abs(x_base - front_threshold) < abs(y_base):
                zones.append('left' if y_base > 0 else 'right')
            else:
                zones.append('front' if x_base > 0 else 'rear')
        
        return zones
    
    def _scan_callback(self, msg):
        """Process laser scan data"""
        self.latest_scan = msg
        self._process_obstacles(msg)
    
    def _process_obstacles(self, scan):
        """Process scan and classify obstacles"""
        self.obstacles_in_base_frame = []
        self.collision_zones = {zone: [] for zone in self.collision_zones.keys()}
        
        if not scan.ranges:
            return
        
        for i, distance in enumerate(scan.ranges):
            # Skip invalid readings
            if math.isnan(distance) or math.isinf(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue
            
            # Skip far obstacles
            if distance > self.slowdown_distance * 2.5:
                continue
            
            # Calculate angle and position in lidar frame
            angle = scan.angle_min + i * scan.angle_increment
            x_lidar = distance * math.cos(angle)
            y_lidar = distance * math.sin(angle)
            
            # Transform to base_link frame
            x_base, y_base = self._transform_lidar_to_base(x_lidar, y_lidar)
            
            # Skip if inside footprint
            if self._is_point_in_footprint(x_base, y_base):
                continue
            
            # Calculate distance to wheelchair edge
            dist_to_edge = self._calculate_distance_to_footprint_edge(x_base, y_base)
            
            # Create obstacle data
            obstacle = {
                'x_base': x_base,
                'y_base': y_base,
                'x_lidar': x_lidar,
                'y_lidar': y_lidar,
                'distance_lidar': distance,
                'distance_to_edge': dist_to_edge,
                'angle': angle
            }
            
            self.obstacles_in_base_frame.append(obstacle)
            
            # Classify into zones
            zones = self._classify_obstacle_zone(x_base, y_base)
            for zone in zones:
                self.collision_zones[zone].append(obstacle)
        
        # Publish visualization if enabled
        if self.publish_markers and self.use_collision_avoidance:
            self._publish_visualization_markers()
    
    def _check_collision_risk(self, joy_x, joy_y):
        """Enhanced collision checking with proper geometry"""
        if not self.obstacles_in_base_frame:
            return 1.0, 1.0
        
        linear_scale = 1.0
        angular_scale = 1.0
        
        # Movement analysis
        # joy_x > 0 is RIGHT turn, joy_x < 0 is LEFT turn
        moving_forward = joy_y > 0.01
        moving_backward = joy_y < -0.01
        turning_right = joy_x > 0.01  # FIXED: positive x is right turn
        turning_left = joy_x < -0.01   # FIXED: negative x is left turn
        
        # CHECK FRONT OBSTACLES
        if moving_forward:
            # Check all front zones
            front_obstacles = (self.collision_zones['front'] + 
                              self.collision_zones['front_left'] + 
                              self.collision_zones['front_right'])
            
            if front_obstacles:
                # Find closest obstacle in front path
                path_obstacles = [o for o in front_obstacles 
                                if abs(o['y_base']) <= self.half_width + 0.1]
                
                if path_obstacles:
                    closest = min(path_obstacles, key=lambda o: o['distance_to_edge'])
                    dist = closest['distance_to_edge']
                    
                    if dist < self.critical_distance:
                        linear_scale = 0.0
                        self.get_logger().warn(f"⛔ STOP! Front obstacle at {dist:.2f}m")
                    elif dist < self.safety_distance:
                        linear_scale = 0.1
                        self.get_logger().warn(f"⚠️ Front obstacle close: {dist:.2f}m")
                    elif dist < self.slowdown_distance:
                        linear_scale = 0.1 + 0.9 * ((dist - self.safety_distance) / 
                                                    (self.slowdown_distance - self.safety_distance))
        
        # CHECK REAR OBSTACLES
        if moving_backward:
            rear_obstacles = (self.collision_zones['rear'] + 
                             self.collision_zones['rear_left'] + 
                             self.collision_zones['rear_right'])
            
            if rear_obstacles:
                path_obstacles = [o for o in rear_obstacles 
                                if abs(o['y_base']) <= self.half_width + 0.2]
                
                if path_obstacles:
                    closest = min(path_obstacles, key=lambda o: o['distance_to_edge'])
                    dist = closest['distance_to_edge']
                    
                    if dist < self.critical_distance:
                        linear_scale = 0.0
                        self.get_logger().warn(f"⛔ STOP! Rear obstacle at {dist:.2f}m")
                    elif dist < self.safety_distance:
                        linear_scale = 0.1
                    elif dist < self.slowdown_distance:
                        linear_scale = 0.1 + 0.9 * ((dist - self.safety_distance) / 
                                                    (self.slowdown_distance - self.safety_distance))
        
        # CHECK SIDE OBSTACLES FOR TURNING
        if turning_left:
            left_obstacles = (self.collision_zones['left'] + 
                             self.collision_zones['front_left'] + 
                             self.collision_zones['rear_left'])
            
            if left_obstacles:
                # Check obstacles along the wheelchair length
                side_obstacles = [o for o in left_obstacles 
                                 if self.rear_edge_x - 0.2 <= o['x_base'] <= self.front_edge_x + 0.2]
                
                if side_obstacles:
                    closest = min(side_obstacles, key=lambda o: abs(o['y_base'] - self.half_width))
                    dist_to_side = abs(closest['y_base']) - self.half_width
                    
                    if dist_to_side < self.critical_distance:
                        angular_scale = 0.0
                        self.get_logger().warn(f"⛔ STOP! Left obstacle at {dist_to_side:.2f}m")
                    elif dist_to_side < self.safety_distance:
                        angular_scale = 0.2
                    elif dist_to_side < self.slowdown_distance:
                        angular_scale = 0.2 + 0.8 * ((dist_to_side - self.safety_distance) / 
                                                     (self.slowdown_distance - self.safety_distance))
        
        if turning_right:
            right_obstacles = (self.collision_zones['right'] + 
                              self.collision_zones['front_right'] + 
                              self.collision_zones['rear_right'])
            
            if right_obstacles:
                side_obstacles = [o for o in right_obstacles 
                                 if self.rear_edge_x - 0.2 <= o['x_base'] <= self.front_edge_x + 0.2]
                
                if side_obstacles:
                    closest = min(side_obstacles, key=lambda o: abs(o['y_base'] + self.half_width))
                    dist_to_side = abs(closest['y_base']) - self.half_width
                    
                    if dist_to_side < self.critical_distance:
                        angular_scale = 0.0
                        self.get_logger().warn(f"⛔ STOP! Right obstacle at {dist_to_side:.2f}m")
                    elif dist_to_side < self.safety_distance:
                        angular_scale = 0.2
                    elif dist_to_side < self.slowdown_distance:
                        angular_scale = 0.2 + 0.8 * ((dist_to_side - self.safety_distance) / 
                                                     (self.slowdown_distance - self.safety_distance))
        
        # Store for status display
        self.last_collision_check = {'linear_scale': linear_scale, 'angular_scale': angular_scale}
        self.collision_warning = (linear_scale < 0.5 or angular_scale < 0.5)
        
        return linear_scale, angular_scale
    
    def _read_gpio_joystick(self):
        """Read GPIO buttons and analog joystick with collision filtering"""
        try:
            current_time = time.time()
            
            # Read button states
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
            
            # Store raw values
            self.raw_joystick_x = x_axis
            self.raw_joystick_y = y_axis
            
            # Apply collision avoidance filtering if enabled
            if self.use_collision_avoidance and self.latest_scan:
                linear_scale, angular_scale = self._check_collision_risk(x_axis, y_axis)
                
                # Apply scaling with heavy smoothing to prevent jitter
                target_x = x_axis * angular_scale
                target_y = y_axis * linear_scale
                
                # Strong smoothing filter to prevent jagged motion
                smooth_factor = 0.8  # Higher = more smoothing
                self.filtered_joystick_x = smooth_factor * self.filtered_joystick_x + (1 - smooth_factor) * target_x
                self.filtered_joystick_y = smooth_factor * self.filtered_joystick_y + (1 - smooth_factor) * target_y
                
                # If we're supposed to stop, actually stop (override smoothing)
                if linear_scale == 0.0:
                    self.filtered_joystick_y = 0.0
                if angular_scale == 0.0:
                    self.filtered_joystick_x = 0.0
            else:
                self.filtered_joystick_x = x_axis
                self.filtered_joystick_y = y_axis
            
            # Determine control mode
            joystick_active = (self.motors_enabled and 
                             (abs(self.filtered_joystick_x) > 0.1 or 
                              abs(self.filtered_joystick_y) > 0.1 or 
                              self.braking_active))
            self.joystick_control_active = joystick_active
            self.control_mode = 'joystick' if joystick_active else 'nav2'
            
            if self.joystick_control_active:
                self._handle_joystick_control(self.filtered_joystick_x, self.filtered_joystick_y)
            
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
        """Handle joystick motor control with collision-filtered values"""
        if self.emergency_stop or not self.motors_enabled:
            return
        
        # Tank drive calculation
        forward = y_axis * self.current_speed_scale
        turn = x_axis * self.current_speed_scale
        
        left_speed = forward + turn * 0.5
        right_speed = forward - turn * 0.5
        
        # Normalize speeds
        max_magnitude = max(abs(left_speed), abs(right_speed))
        if max_magnitude > 1.0:
            left_speed /= max_magnitude
            right_speed /= max_magnitude
        
        # Convert to RPM
        left_rpm = -left_speed * self.max_joystick_speed
        right_rpm = right_speed * self.max_joystick_speed
        
        # Apply braking
        if self.braking_active:
            left_rpm *= 0.0
            right_rpm *= 0.0
        
        try:
            self.motor_left.set_target_velocity_rpm(int(left_rpm))
            self.motor_right.set_target_velocity_rpm(int(right_rpm))
        except Exception as e:
            self.get_logger().error(f"Failed to set motor speeds: {e}")
    
    def _publish_joy_message(self, x_axis, y_axis, button_pressed):
        """Publish Joy message with collision info"""
        joy_msg = Joy()
        joy_msg.header.stamp = self.get_clock().now().to_msg()
        joy_msg.header.frame_id = 'joystick'
        
        # axes[0-1]: raw joystick, axes[2-3]: filtered
        joy_msg.axes = [x_axis, y_axis, self.filtered_joystick_x, self.filtered_joystick_y]
        
        # Button states + collision warning
        joy_msg.buttons = [
            int(button_pressed['speed_up']),
            int(button_pressed['speed_down']),
            int(button_pressed['brake']),
            int(self.motors_enabled),
            int(self.joystick_control_active),
            int(self.collision_warning)
        ]
        
        self.joy_pub.publish(joy_msg)
    
    def _publish_visualization_markers(self):
        """Publish markers for RViz visualization with proper deletion"""
        if not self.publish_markers:
            return
        
        # Initialize marker count if not exists
        if not hasattr(self, '_last_marker_count'):
            self._last_marker_count = 0
            self._marker_initialized = False
        
        current_time = self.get_clock().now().to_msg()
        
        # First, clear old markers if we have any
        if self._last_marker_count > 0 or not self._marker_initialized:
            clear_array = MarkerArray()
            clear_marker = Marker()
            clear_marker.header.frame_id = "base_link"
            clear_marker.header.stamp = current_time
            clear_marker.ns = "collision"
            clear_marker.action = Marker.DELETEALL
            clear_array.markers.append(clear_marker)
            self.marker_pub.publish(clear_array)
            self._marker_initialized = True
            # Small delay to ensure deletion is processed
            time.sleep(0.001)
        
        # Now create new markers
        marker_array = MarkerArray()
        
        # Wheelchair footprint
        footprint = Marker()
        footprint.header.frame_id = "base_link"
        footprint.header.stamp = current_time
        footprint.ns = "collision"
        footprint.id = 0
        footprint.type = Marker.LINE_STRIP
        footprint.action = Marker.ADD
        footprint.scale.x = 0.03
        footprint.color.r = 0.0
        footprint.color.g = 1.0
        footprint.color.b = 0.0
        footprint.color.a = 1.0
        
        # Wheelchair corners
        corners = [
            (self.front_edge_x, self.half_width),
            (self.front_edge_x, -self.half_width),
            (self.rear_edge_x, -self.half_width),
            (self.rear_edge_x, self.half_width),
            (self.front_edge_x, self.half_width)  # Close the rectangle
        ]
        
        for corner in corners:
            p = Point()
            p.x = corner[0]
            p.y = corner[1]
            p.z = 0.0
            footprint.points.append(p)
        
        marker_array.markers.append(footprint)
        
        # Lidar position marker
        lidar_marker = Marker()
        lidar_marker.header.frame_id = "base_link"
        lidar_marker.header.stamp = current_time
        lidar_marker.ns = "collision"
        lidar_marker.id = 1
        lidar_marker.type = Marker.SPHERE
        lidar_marker.action = Marker.ADD
        lidar_marker.pose.position.x = self.base_to_lidar_x
        lidar_marker.pose.position.y = 0.0
        lidar_marker.pose.position.z = 0.1
        lidar_marker.scale.x = lidar_marker.scale.y = lidar_marker.scale.z = 0.1
        lidar_marker.color.r = 0.0
        lidar_marker.color.g = 0.0
        lidar_marker.color.b = 1.0
        lidar_marker.color.a = 1.0
        marker_array.markers.append(lidar_marker)
        
        # Obstacle markers - limit to prevent too many markers
        marker_id = 2
        max_obstacles = min(len(self.obstacles_in_base_frame), 200)  # Limit to 200 obstacles
        
        for i in range(max_obstacles):
            obs = self.obstacles_in_base_frame[i]
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = current_time
            marker.ns = "collision"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obs['x_base']
            marker.pose.position.y = obs['y_base']
            marker.pose.position.z = 0.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            
            # Color based on distance to edge
            dist = obs['distance_to_edge']
            if dist < self.critical_distance:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif dist < self.safety_distance:
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
            elif dist < self.slowdown_distance:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 1.0
            
            marker.color.a = 0.8
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Store count for next iteration
        self._last_marker_count = marker_id
        
        # Publish new markers
        self.marker_pub.publish(marker_array)
    
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
            
            # Calculate encoder deltas
            delta_left = self._handle_encoder_overflow(left_encoder - self.last_left_encoder)
            delta_right = self._handle_encoder_overflow(right_encoder - self.last_right_encoder)
            
            # Convert to distances
            delta_left_dist = -delta_left * self.meters_per_count
            delta_right_dist = delta_right * self.meters_per_count
            
            # Calculate robot motion
            delta_distance = (delta_left_dist + delta_right_dist) / 2.0
            delta_theta = (delta_right_dist - delta_left_dist) / self.wheel_base
            
            # Update pose
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
        
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        
        q = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.angular.z = self.vth
        
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
        """Publish enhanced motor status with collision info"""
        mode = "JOYSTICK" if self.joystick_control_active else "NAV2"
        enabled = "ON" if self.motors_enabled else "OFF"
        estop = "ESTOP" if self.emergency_stop else "OK"
        brake = " BRAKE" if self.braking_active else ""
        analog = " ANALOG" if self.analog_available else " BUTTONS"
        
        # Collision status
        collision_status = ""
        if self.use_collision_avoidance and self.collision_warning:
            collision_status = f" ⚠️COLLISION"
            if self.last_collision_check['linear_scale'] < 1.0:
                collision_status += f" L:{self.last_collision_check['linear_scale']:.0%}"
            if self.last_collision_check['angular_scale'] < 1.0:
                collision_status += f" A:{self.last_collision_check['angular_scale']:.0%}"
        
        # Obstacle info
        obs_info = ""
        if self.use_collision_avoidance and self.obstacles_in_base_frame:
            closest = min(self.obstacles_in_base_frame, key=lambda o: o['distance_to_edge'])
            obs_info = f" | Obs:{len(self.obstacles_in_base_frame)} Closest:{closest['distance_to_edge']:.2f}m"
            
            # Show active zones
            active_zones = [z for z, obs in self.collision_zones.items() if obs]
            if active_zones:
                obs_info += f" [{','.join(active_zones[:3])}]"
        
        status = f"{mode} | Motors:{enabled} | {estop} | Speed:{self.current_speed_scale:.1f}{analog}{brake}{collision_status}{obs_info}"
        
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
        node = IntegratedSafeMotorController()
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