#!/usr/bin/env python3
"""
Jetson Motor Control Node with Dual Side-Mounted LiDAR Collision Avoidance
ROS2 node for differential drive wheelchair control with enhanced safety - Joystick Control Only
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
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
        self._init_collision_avoidance()
        self._init_hardware()
        self._init_state()
        self._setup_ros_interfaces()
        
        self.init_thread = threading.Thread(target=self._initialize_motors, daemon=True)
        self.init_thread.start()
        
        self.get_logger().info("Jetson Motor Controller with Dual LiDAR Collision Avoidance initialized (Joystick Only)")
    
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
        self.declare_parameter('max_joystick_speed', 100)
        
        # GPIO configuration
        self.declare_parameter('gpio_pin_speed_up', 29)
        self.declare_parameter('gpio_pin_speed_down', 31)
        self.declare_parameter('gpio_pin_brake', 33)
        self.declare_parameter('hold_time_enable_motors', 3.0)
        self.declare_parameter('hold_time_disable_motors', 2.0)
        
        # Control parameters
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('joystick_publish_rate', 50.0)
        self.declare_parameter('joystick_deadzone', 0.15)
        self.declare_parameter('speed_scale_increment', 0.1)
        self.declare_parameter('button_debounce_time', 0.3)
        
        # Collision avoidance parameters
        self.declare_parameter('collision_avoidance_enabled', True)
        self.declare_parameter('critical_distance', 0.2)
        self.declare_parameter('safety_distance', 0.3)
        self.declare_parameter('warning_distance', 0.6)
        self.declare_parameter('side_clearance', 0.3)
        self.declare_parameter('use_laser_scan', True)
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
        self.max_joystick_speed = self.get_parameter('max_joystick_speed').value
        self.hold_time_enable = self.get_parameter('hold_time_enable_motors').value
        self.hold_time_disable = self.get_parameter('hold_time_disable_motors').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.joystick_publish_rate = self.get_parameter('joystick_publish_rate').value
        self.joystick_deadzone = self.get_parameter('joystick_deadzone').value
        self.speed_scale_increment = self.get_parameter('speed_scale_increment').value
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
        """Initialize collision avoidance system for dual side-mounted LiDARs"""
        # Set default values
        self.collision_avoidance_enabled = True
        self.use_laser_scan = True
        self.laser_scan_topic = '/scan'
        self.critical_distance = 0.2
        self.safety_distance = 0.3
        self.warning_distance = 0.6
        self.side_clearance = 0.2

        # Override with parameter values if available
        try:
            self.collision_avoidance_enabled = self.get_parameter('collision_avoidance_enabled').value
            self.critical_distance = self.get_parameter('critical_distance').value
            self.safety_distance = self.get_parameter('safety_distance').value
            self.warning_distance = self.get_parameter('warning_distance').value
            self.side_clearance = self.get_parameter('side_clearance').value
            self.use_laser_scan = self.get_parameter('use_laser_scan').value
            self.laser_scan_topic = self.get_parameter('laser_scan_topic').value
        except Exception as e:
            self.get_logger().warn(f"Could not load some collision avoidance parameters, using defaults: {e}")
        
        # Collision detection state
        self.last_scan = None
        self.collision_detected = False
        self.collision_direction = None
        self.min_obstacle_distance = float('inf')
        self.obstacle_points = []  # Store obstacle coordinates as (x,y,distance) tuples
        self.sector_distances = {
            'front': float('inf'),
            'front_left': float('inf'),
            'front_right': float('inf'),
            'left': float('inf'),
            'right': float('inf'),
            'rear': float('inf')
        }
        
        # Wheelchair dimensions for filtering
        self.wheelchair_length = 1.0
        self.wheelchair_width = 0.7
        
        self.get_logger().info(f"Collision avoidance initialized for dual side-mounted LiDARs")
        self.get_logger().info(f"Critical: {self.critical_distance}m, Safety: {self.safety_distance}m, Warning: {self.warning_distance}m")
    
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
        self.current_speed_scale = 0.5
        
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
        
        # Collision avoidance subscribers
        if self.collision_avoidance_enabled and self.use_laser_scan:
            self.get_logger().info(f"Setting up laser scan subscription to {self.laser_scan_topic}...")
            
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            
            self.scan_sub = self.create_subscription(
                LaserScan, 
                self.laser_scan_topic, 
                self._scan_callback, 
                qos_profile
            )
            self.get_logger().info("Laser scan subscriber created - waiting for scan data...")
            self.scan_check_timer = self.create_timer(5.0, self._check_scan_connection)
        
        # Services
        self.emergency_stop_srv = self.create_service(Trigger, 'emergency_stop', self._emergency_stop_service)
        self.enable_motors_srv = self.create_service(Trigger, 'enable_motors', self._enable_motors_service)
        self.disable_motors_srv = self.create_service(Trigger, 'disable_motors', self._disable_motors_service)
        
        # Timers
        self.odometry_timer = self.create_timer(1.0 / self.publish_rate, self._update_odometry)
        self.joystick_timer = self.create_timer(1.0 / self.joystick_publish_rate, self._read_gpio_joystick)
    
    def _check_scan_connection(self):
        """Check if laser scan data is being received"""
        if self.last_scan is None:
            self.get_logger().warn(f"No laser scan data received from {self.laser_scan_topic}")
            self.get_logger().warn("Check that the laser scan is published and ROS_DOMAIN_ID matches")
        else:
            self.get_logger().info(f"✓ Laser scan OK - {len(self.last_scan.ranges)} points, 270° coverage")
            self.scan_check_timer.cancel()
    
    def _scan_callback(self, msg):
        """Handle laser scan data for collision detection"""
        self.last_scan = msg
        
        if not hasattr(self, '_scan_received'):
            self._scan_received = True
            self.get_logger().info(f"✓ Laser scan received! Points: {len(msg.ranges)}")
        
        if self.collision_avoidance_enabled:
            self._update_collision_detection_from_scan(msg)
    
    def _point_in_core_footprint(self, x, y):
        """Check if point is definitely within wheelchair core structure"""
        # Very conservative footprint - only filter obvious wheelchair parts
        return (abs(x) < 0.6 and abs(y) < 0.3)
    
    def _point_in_safety_footprint(self, x, y):
        """Check if point is within wheelchair safety footprint"""
        # Based on actual footprint but slightly reduced to be conservative
        footprint = [
            [0.7, 0.3],    # Front right
            [0.7, -0.3],   # Front left
            [-0.15, -0.35], # Rear left
            [-0.15, 0.35]   # Rear right
        ]
        
        # Point-in-polygon test
        n = len(footprint)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = footprint[i]
            xj, yj = footprint[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def _clear_sector_distances(self):
        """Clear all sector distances"""
        for sector in self.sector_distances:
            self.sector_distances[sector] = float('inf')
    
    def _update_collision_detection_from_scan(self, scan_msg):
        """Simple collision detection using obstacle coordinates (fixed)."""
        if not scan_msg.ranges:
            return

        ranges = np.array(scan_msg.ranges)
        original_count = len(ranges)

        ranges = np.where(np.isinf(ranges), scan_msg.range_max, ranges)
        ranges = np.where(np.isnan(ranges), scan_msg.range_max, ranges)
        ranges = np.where(ranges < 0.05, scan_msg.range_max, ranges)  # ignore extremely noisy zeroes

        valid_readings = np.sum(ranges < scan_msg.range_max)
        close_readings = np.sum(ranges < 2.0)
        self.get_logger().debug(f"Scan: {original_count} pts, {valid_readings} valid, {close_readings} <2m, range_max={scan_msg.range_max}")

        if len(ranges) == 0:
            self.min_obstacle_distance = float('inf')
            self.collision_detected = False
            self.obstacle_points = []
            return

        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        angles = np.array([angle_min + i * angle_increment for i in range(len(ranges))])

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        self.obstacle_points = []
        filtered_count = 0

        # Collect obstacles within warning_distance but allow very close obstacles (no >0.4 cutoff)
        for i in range(len(ranges)):
            r = ranges[i]
            if r >= scan_msg.range_max or r > self.warning_distance:
                continue

            x = xs[i]
            y = ys[i]

            # filter robot body using point-in-footprint
            if self._point_in_core_footprint(x, y):
                continue

            # discard extremely close readings caused by vehicle reflection or forklift corners if desired
            if r < 0.05:
                continue

            self.obstacle_points.append((x, y, r))
            filtered_count += 1

            # debug first few
            if i < 5:
                self.get_logger().debug(f"Point {i}: r={r:.2f}, ang={np.degrees(angles[i]):.1f}°, x={x:.2f}, y={y:.2f}")

        self.get_logger().info(f"Filtered to {filtered_count} obstacles within {self.warning_distance}m")

        if not self.obstacle_points:
            self.min_obstacle_distance = float('inf')
            self.collision_detected = False
            self.collision_direction = None
            self._clear_sector_distances()
            return

        # Update sector distances from collected points
        self._update_sectors_from_points()

        # Determine the minimum distance among sectors (only finite ones)
        finite_sectors = {k: v for k, v in self.sector_distances.items() if v != float('inf')}
        if finite_sectors:
            # find sector with minimum distance
            sector, dist = min(finite_sectors.items(), key=lambda kv: kv[1])
            self.min_obstacle_distance = dist
            self.collision_detected = dist < self.warning_distance
            self.collision_direction = sector
            self.get_logger().info(f"Closest sector: {sector} at {dist:.2f}m")
        else:
            self.min_obstacle_distance = float('inf')
            self.collision_detected = False
            self.collision_direction = None

        # Also log the absolute closest point for clarity
        distances = [p[2] for p in self.obstacle_points]
        if distances:
            idx = int(np.argmin(distances))
            cp = self.obstacle_points[idx]
            self.get_logger().debug(f"Closest point: x={cp[0]:.2f}, y={cp[1]:.2f}, r={cp[2]:.2f}")

    
    def _update_sectors_from_points(self):
        """Update sector distances from obstacle points with improved front detection."""
        for sector in self.sector_distances:
            self.sector_distances[sector] = float('inf')

        for x, y, dist in self.obstacle_points:
            angle = np.degrees(np.arctan2(y, x))  # -180..180

            # FRONT sector: ±30° cone
            if -30 <= angle <= 30:
                self.sector_distances['front'] = min(self.sector_distances['front'], dist)

            # FRONT-LEFT: 30°–80°
            elif 30 < angle <= 80:
                self.sector_distances['front_left'] = min(self.sector_distances['front_left'], dist)

            # FRONT-RIGHT: -80° to -30°
            elif -80 <= angle < -30:
                self.sector_distances['front_right'] = min(self.sector_distances['front_right'], dist)

            # LEFT: 80°–130°
            elif 80 < angle <= 130:
                self.sector_distances['left'] = min(self.sector_distances['left'], dist)

            # RIGHT: -130° to -80°
            elif -130 <= angle < -80:
                self.sector_distances['right'] = min(self.sector_distances['right'], dist)

            # REAR: >130° or < -130°
            else:
                self.sector_distances['rear'] = min(self.sector_distances['rear'], dist)

    
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
            
            # Always handle joystick control when motors are enabled
            if self.motors_enabled:
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
        
        # IMPORTANT: Check if Y-axis needs inversion
        # Positive y_axis should mean forward, negative should mean backward
        # If your joystick is inverted, uncomment the next line:
        # if axis == 'y':
        #     axis_value = -axis_value
        
        if abs(axis_value) < self.joystick_deadzone:
            return 0.0
        
        sign = 1 if axis_value > 0 else -1
        return sign * (abs(axis_value) - self.joystick_deadzone) / (1.0 - self.joystick_deadzone)
    
    def _handle_joystick_control(self, x_axis, y_axis):
        """FIXED joystick control with collision avoidance for side-mounted LiDARs"""
        if self.emergency_stop or not self.motors_enabled:
            return
        
        # Get base speeds from joystick
        forward = y_axis * self.current_speed_scale * 0.8  # Positive = forward, Negative = backward
        turn = x_axis * self.current_speed_scale * 0.7     # Positive = left, Negative = right
        
        # DEBUG: Log joystick inputs occasionally
        if not hasattr(self, '_joystick_debug_counter'):
            self._joystick_debug_counter = 0
        self._joystick_debug_counter += 1
        if self._joystick_debug_counter % 50 == 0:  # Every 50 calls
            self.get_logger().info(f"Joystick: forward={forward:.2f}, turn={turn:.2f}")
        
        # Apply collision avoidance - FIXED LOGIC
        if self.collision_avoidance_enabled and hasattr(self, 'sector_distances'):
            
            # Check for critical obstacles
            if self.min_obstacle_distance < self.critical_distance:
                original_forward = forward
                original_turn = turn
                
                # ACTUALLY CORRECT LOGIC based on your coordinate system:
                # In your system: turn > 0 = RIGHT turn, turn < 0 = LEFT turn
                # So: LEFT obstacle should block LEFT turns (turn < 0)
                # And: RIGHT obstacle should block RIGHT turns (turn > 0)
                
                # CORRECTED forward/backward logic:
                # FRONT obstacle should block FORWARD movement (forward > 0)
                # REAR obstacle should block BACKWARD movement (forward < 0)
                
                if self.collision_direction in ['front', 'front_left', 'front_right']:
                    if forward > 0:  # Block FORWARD into FRONT obstacle
                        forward = 0.0
                    if turn != 0.0:
                        turn = 0.0  # Block turning when front obstacle is critical
                        self.get_logger().error(f"🚨 CRITICAL: Front obstacle - blocking forward!")
                    # Allow backward (forward < 0) away from front obstacle
                elif self.collision_direction == 'rear':
                    if forward < 0:  # Block BACKWARD into REAR obstacle  
                        forward = 0.0 
                        self.get_logger().error(f"🚨 CRITICAL: Rear obstacle - blocking backward!")
                    # Allow forward (forward > 0) away from rear obstacle
                elif self.collision_direction == 'left':
                    if turn < 0:  # Block LEFT turn (negative) when obstacle on LEFT
                        turn = 0.0
                        self.get_logger().error(f"🚨 CRITICAL: Left obstacle - blocking LEFT turn!")
                elif self.collision_direction == 'right':
                    if turn > 0:  # Block RIGHT turn (positive) when obstacle on RIGHT
                        turn = 0.0  
                        self.get_logger().error(f"🚨 CRITICAL: Right obstacle - blocking RIGHT turn!")
                
                # Debug what changed
                if forward != original_forward or turn != original_turn:
                    self.get_logger().warn(f"Movement blocked: was ({original_forward:.2f},{original_turn:.2f}) now ({forward:.2f},{turn:.2f})")
            
            else:
                # Normal collision avoidance with warning/safety distances
                # Forward motion check - only check front obstacles
                if forward > 0.1:  # Moving FORWARD
                    # Check if there are front obstacles to block forward movement
                    front_clear = min(self.sector_distances['front'],
                                     self.sector_distances['front_left'],
                                     self.sector_distances['front_right'])
                    
                    if front_clear < self.safety_distance:
                        forward = 0.0  # Block forward movement into front obstacle
                        self.get_logger().warn(f"🛑 Forward blocked by front obstacle: {front_clear:.2f}m")
                    elif front_clear < self.warning_distance:
                        factor = (front_clear - self.safety_distance) / (self.warning_distance - self.safety_distance)
                        forward *= max(0.4, factor)
                
                # NO backward collision checking - LiDAR blind zone behind wheelchair
                # Always allow backward movement since we can't see obstacles behind
                
                # Turning motion check - CORRECTED for your coordinate system
                if turn != 0.0:
                    if turn > 0:  # RIGHT turn
                        side_clear = self.sector_distances['right']
                        diagonal_clear = self.sector_distances['front_right']
                        min_turn_clearance = min(side_clear, diagonal_clear)

                        if min_turn_clearance < self.critical_distance:
                            turn = 0.0
                            self.get_logger().error(f"🚨 CRITICAL: Right obstacle - blocking RIGHT turn!")
                        elif min_turn_clearance < self.side_clearance:
                            turn = 0.0
                            self.get_logger().warn(f"🛑 Right turn blocked: {min_turn_clearance:.2f}m")
                        elif min_turn_clearance < self.warning_distance:
                            factor = (min_turn_clearance - self.side_clearance) / (self.warning_distance - self.side_clearance)
                            turn *= max(0.4, factor)

                    else:  # LEFT turn
                        side_clear = self.sector_distances['left']
                        diagonal_clear = self.sector_distances['front_left']
                        min_turn_clearance = min(side_clear, diagonal_clear)

                        if min_turn_clearance < self.critical_distance:
                            turn = 0.0
                            self.get_logger().error(f"🚨 CRITICAL: Left obstacle - blocking LEFT turn!")
                        elif min_turn_clearance < self.side_clearance:
                            turn = 0.0
                            self.get_logger().warn(f"🛑 Left turn blocked: {min_turn_clearance:.2f}m")
                        elif min_turn_clearance < self.warning_distance:
                            factor = (min_turn_clearance - self.side_clearance) / (self.warning_distance - self.side_clearance)
                            turn *= max(0.4, factor)

                
                # Overall proximity reduction - less aggressive
                if self.min_obstacle_distance < self.warning_distance:
                    proximity_factor = max(0.3, (self.min_obstacle_distance - self.critical_distance) / 
                                          (self.warning_distance - self.critical_distance))
                    forward *= proximity_factor
                    turn *= proximity_factor
        
        # Calculate differential drive speeds
        left_speed = forward + turn
        right_speed = forward - turn
        
        # Normalize speeds
        max_magnitude = max(abs(left_speed), abs(right_speed))
        if max_magnitude > 1.0:
            left_speed /= max_magnitude
            right_speed /= max_magnitude
        
        # Convert to RPM
        left_rpm = -left_speed * self.max_joystick_speed * 0.8
        right_rpm = right_speed * self.max_joystick_speed * 0.8
        
        # Apply braking
        if self.braking_active:
            left_rpm *= 0.3
            right_rpm *= 0.3
        
        try:
            self.motor_left.set_target_velocity_rpm(int(left_rpm))
            self.motor_right.set_target_velocity_rpm(int(right_rpm))
        except Exception as e:
            self.get_logger().error(f"Failed to set motor speeds: {e}")
    
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
            int(self.collision_detected)
        ]
        self.joy_pub.publish(joy_msg)
    
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
        """Publish enhanced motor status with collision info"""
        enabled = "ON" if self.motors_enabled else "OFF"
        estop = "ESTOP" if self.emergency_stop else "OK"
        brake = " BRAKE" if self.braking_active else ""
        analog = " ANALOG" if self.analog_available else " BUTTONS"
        
        # Enhanced collision status with all sectors
        if self.min_obstacle_distance < self.critical_distance:
            collision = " 🚨CRITICAL"
        elif self.collision_detected:
            collision = " ⚠️OBSTACLE"
        else:
            collision = ""
        
        if self.collision_detected or self.min_obstacle_distance < 2.0:
            # Show sector distances
            collision_info = f" [F:{self.sector_distances['front']:.1f}"
            collision_info += f" FL:{self.sector_distances['front_left']:.1f}"
            collision_info += f" FR:{self.sector_distances['front_right']:.1f}"
            collision_info += f" L:{self.sector_distances['left']:.1f}"
            collision_info += f" R:{self.sector_distances['right']:.1f}"
            collision_info += f" B:{self.sector_distances['rear']:.1f}]m"
        else:
            collision_info = ""
        
        status = f"JOYSTICK | Motors: {enabled} | {estop} | Speed: {self.current_speed_scale:.1f}{analog}{brake}{collision}{collision_info}"
        
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
                motor.set_deceleration(0.9)

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