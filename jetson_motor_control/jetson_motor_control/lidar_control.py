#!/usr/bin/env python3
"""
Simplified Integrated Jetson Motor Control with Collision Avoidance
Motors enabled on launch, controlled via cmd_vel only
Includes analog joystick monitoring with raw value printing
NEW: Collision avoidance can be toggled via ROS2 service
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
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
        
        self.get_logger().info("Integrated Safe Motor Controller initialized (motors will auto-enable)")
        self.get_logger().info("Analog joystick monitoring enabled - raw values will print to terminal")
        self.get_logger().info(f"Collision avoidance: {'ENABLED' if self.collision_avoidance_active else 'DISABLED'}")
    
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
        
        # Control parameters
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('fixed_speed_scale', 0.5)
        self.declare_parameter('joystick_monitor_rate', 10.0)  # Hz for printing
        
        # Wheelchair geometry 
        self.declare_parameter('base_to_lidar_x', 0.54)
        self.declare_parameter('lidar_to_front_x', 0.25)
        self.declare_parameter('base_to_rear_x', 0.25)
        self.declare_parameter('wheelchair_width', 0.65)
        
        # Collision avoidance parameters
        self.declare_parameter('use_collision_avoidance', True)
        self.declare_parameter('safety_margin', 0.1)
        self.declare_parameter('critical_distance', 0.15)
        self.declare_parameter('safety_distance', 0.25)
        self.declare_parameter('slowdown_distance', 0.5)
        self.declare_parameter('side_safety_buffer', 0.15)
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
        
        # Control parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.speed_scale = self.get_parameter('fixed_speed_scale').value
        self.joystick_monitor_rate = self.get_parameter('joystick_monitor_rate').value
        
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
        self.get_logger().info(f"Fixed speed scale: {self.speed_scale}")
    
    def _init_collision_system(self):
        """Initialize collision avoidance system"""
        # Runtime toggle for collision avoidance (can be changed via service)
        self.collision_avoidance_active = self.use_collision_avoidance
        
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
    
    def _init_hardware(self):
        """Initialize all hardware components"""
        self._init_motors()
        self._init_analog_joystick()
    
    def _init_motors(self):
        """Initialize motor drivers"""
        self.motor_left = L2DBMotorDriver(self.can_interface, self.can_channel, self.motor_left_id)
        self.motor_right = L2DBMotorDriver(self.can_interface, self.can_channel, self.motor_right_id)
    
    def _init_analog_joystick(self):
        """Initialize analog joystick via I2C/ADS1115 for monitoring only"""
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(self.i2c)
            self.ads.gain = 1
            
            self.x_channel = AnalogIn(self.ads, ADS.P0)
            self.y_channel = AnalogIn(self.ads, ADS.P1)
            
            self.analog_available = True
            self.get_logger().info("✓ Analog joystick monitoring initialized (I2C/ADS1115)")
            
        except Exception as e:
            self.analog_available = False
            self.get_logger().warn(f"⚠ Analog joystick unavailable: {e}")
    
    def _init_state(self):
        """Initialize robot state variables"""
        # Pose and velocity
        self.x = self.y = self.theta = 0.0
        self.vx = self.vth = 0.0
        
        # Control state
        self.motors_enabled = False
        self.emergency_stop = False
        
        # Command tracking
        self.target_linear_vel = self.target_angular_vel = 0.0
        self.filtered_linear_vel = self.filtered_angular_vel = 0.0
        self.last_cmd_vel_time = self.get_clock().now()
        
        # Joystick monitoring
        self.raw_joystick_x_voltage = 0.0
        self.raw_joystick_y_voltage = 0.0
        
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
        self.enable_collision_avoidance_srv = self.create_service(Trigger, 'enable_collision_avoidance', self._enable_collision_avoidance_service)
        self.disable_collision_avoidance_srv = self.create_service(Trigger, 'disable_collision_avoidance', self._disable_collision_avoidance_service)
        
        # Timers
        self.odometry_timer = self.create_timer(1.0 / self.publish_rate, self._update_odometry)
        self.control_timer = self.create_timer(0.02, self._control_loop)
        
        # Joystick monitoring timer (prints to terminal)
        if self.analog_available:
            self.joystick_monitor_timer = self.create_timer(
                1.0 / self.joystick_monitor_rate, 
                self._monitor_joystick
            )
    
    def _monitor_joystick(self):
        """Read and print raw joystick values to terminal"""
        if not self.analog_available:
            return
        
        try:
            # Read raw voltages
            x_voltage = self.x_channel.voltage
            y_voltage = self.y_channel.voltage
            
            # Store for status
            self.raw_joystick_x_voltage = x_voltage
            self.raw_joystick_y_voltage = y_voltage
            
            # Print to terminal with clear formatting
            print(f"\r🕹️  Joystick Raw | X: {x_voltage:.3f}V | Y: {y_voltage:.3f}V", end='', flush=True)
            
        except Exception as e:
            self.get_logger().error(f"Joystick read error: {e}")
    
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
        front_threshold = self.base_to_lidar_x
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
        # Only process obstacles if collision avoidance is active
        if self.collision_avoidance_active:
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
        if self.publish_markers and self.use_collision_avoidance and self.collision_avoidance_active:
            self._publish_visualization_markers()
    
    def _check_collision_risk(self, linear_vel, angular_vel):
        """Enhanced collision checking with proper geometry"""
        # If collision avoidance is disabled, return full speed
        if not self.collision_avoidance_active:
            return 1.0, 1.0
        
        if not self.obstacles_in_base_frame:
            return 1.0, 1.0
        
        linear_scale = 1.0
        angular_scale = 1.0
        
        # Movement analysis
        moving_forward = linear_vel > 0.01
        moving_backward = linear_vel < -0.01
        turning_right = angular_vel < -0.01
        turning_left = angular_vel > 0.01
        
        # CHECK FRONT OBSTACLES
        if moving_forward:
            front_obstacles = (self.collision_zones['front'] + 
                              self.collision_zones['front_left'] + 
                              self.collision_zones['front_right'])
            
            if front_obstacles:
                path_obstacles = [o for o in front_obstacles 
                                if abs(o['y_base']) <= self.half_width + 0.1]
                
                if path_obstacles:
                    closest = min(path_obstacles, key=lambda o: o['distance_to_edge'])
                    dist = closest['distance_to_edge']
                    
                    if dist < self.critical_distance:
                        linear_scale = 0.0
                        print(f"\n⛔ STOP! Front obstacle at {dist:.2f}m")
                    elif dist < self.safety_distance:
                        linear_scale = 0.1
                        print(f"\n⚠️ Front obstacle close: {dist:.2f}m")
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
                        print(f"\n⛔ STOP! Rear obstacle at {dist:.2f}m")
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
                side_obstacles = [o for o in left_obstacles 
                                 if self.rear_edge_x - 0.2 <= o['x_base'] <= self.front_edge_x + 0.2]
                
                if side_obstacles:
                    closest = min(side_obstacles, key=lambda o: abs(o['y_base'] - self.half_width))
                    dist_to_side = abs(closest['y_base']) - self.half_width
                    
                    if dist_to_side < self.critical_distance:
                        angular_scale = 0.0
                        print(f"\n⛔ STOP! Left obstacle at {dist_to_side:.2f}m")
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
                        print(f"\n⛔ STOP! Right obstacle at {dist_to_side:.2f}m")
                    elif dist_to_side < self.safety_distance:
                        angular_scale = 0.2
                    elif dist_to_side < self.slowdown_distance:
                        angular_scale = 0.2 + 0.8 * ((dist_to_side - self.safety_distance) / 
                                                     (self.slowdown_distance - self.safety_distance))
        
        # Store for status display
        self.last_collision_check = {'linear_scale': linear_scale, 'angular_scale': angular_scale}
        self.collision_warning = (linear_scale < 0.5 or angular_scale < 0.5)
        
        return linear_scale, angular_scale
    
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
            (self.front_edge_x, self.half_width)
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
        max_obstacles = min(len(self.obstacles_in_base_frame), 200)
        
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
        """Handle velocity commands with collision avoidance"""
        if self.emergency_stop or not self.motors_enabled:
            return
        
        # Clamp to max velocities
        self.target_linear_vel = max(-self.max_linear_vel, 
                                   min(self.max_linear_vel, msg.linear.x))
        self.target_angular_vel = max(-self.max_angular_vel, 
                                    min(self.max_angular_vel, msg.angular.z))
        
        # Apply collision avoidance filtering if enabled
        if self.collision_avoidance_active and self.latest_scan:
            linear_scale, angular_scale = self._check_collision_risk(
                self.target_linear_vel, self.target_angular_vel
            )
            
            # Apply scaling with smoothing to prevent jitter
            target_linear = self.target_linear_vel * linear_scale
            target_angular = self.target_angular_vel * angular_scale
            
            # Strong smoothing filter
            smooth_factor = 0.7
            self.filtered_linear_vel = target_linear
            self.filtered_angular_vel = target_angular
            
            # If we're supposed to stop, actually stop (override smoothing)
            if linear_scale == 0.0:
                self.filtered_linear_vel = 0.0
            if angular_scale == 0.0:
                self.filtered_angular_vel = 0.0
        else:
            self.filtered_linear_vel = self.target_linear_vel
            self.filtered_angular_vel = self.target_angular_vel
        
        self.last_cmd_vel_time = self.get_clock().now()
    
    def _control_loop(self):
        """Main control loop"""
        if not self.motors_enabled or self.emergency_stop:
            return
        
        # Check command timeout
        time_since_cmd = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds / 1e9
        if time_since_cmd > self.cmd_vel_timeout:
            self.filtered_linear_vel = self.filtered_angular_vel = 0.0
        
        # Apply fixed speed scale
        scaled_linear = self.filtered_linear_vel * self.speed_scale
        scaled_angular = self.filtered_angular_vel * self.speed_scale
        
        # Calculate differential drive velocities
        v_left = scaled_linear - (scaled_angular * self.wheel_base) / 2.0
        v_right = scaled_linear + (scaled_angular * self.wheel_base) / 2.0
        
        # Convert to RPM
        rpm_left = -(v_left / self.wheel_circumference) * 60.0
        rpm_right = (v_right / self.wheel_circumference) * 60.0
        
        try:
            self.motor_left.set_target_velocity_rpm(rpm_left)
            self.motor_right.set_target_velocity_rpm(rpm_right)
        except Exception as e:
            self.get_logger().error(f"Failed to set motor speeds: {e}")
    
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
        """Publish motor status with collision info"""
        enabled = "ON" if self.motors_enabled else "OFF"
        estop = "ESTOP" if self.emergency_stop else "OK"
        
        # Collision avoidance status
        collision_status_text = f" | CA:{'ON' if self.collision_avoidance_active else 'OFF'}"
        
        # Collision warning
        if self.collision_avoidance_active and self.collision_warning:
            collision_status_text += f" ⚠️"
            if self.last_collision_check['linear_scale'] < 1.0:
                collision_status_text += f" L:{self.last_collision_check['linear_scale']:.0%}"
            if self.last_collision_check['angular_scale'] < 1.0:
                collision_status_text += f" A:{self.last_collision_check['angular_scale']:.0%}"
        
        # Obstacle info
        obs_info = ""
        if self.collision_avoidance_active and self.obstacles_in_base_frame:
            closest = min(self.obstacles_in_base_frame, key=lambda o: o['distance_to_edge'])
            obs_info = f" | Obs:{len(self.obstacles_in_base_frame)} Closest:{closest['distance_to_edge']:.2f}m"
            
            # Show active zones
            active_zones = [z for z, obs in self.collision_zones.items() if obs]
            if active_zones:
                obs_info += f" [{','.join(active_zones[:3])}]"
        
        # Joystick monitoring info
        joy_info = ""
        if self.analog_available:
            joy_info = f" | Joy: X={self.raw_joystick_x_voltage:.3f}V Y={self.raw_joystick_y_voltage:.3f}V"
        
        status = f"CMD_VEL | Motors:{enabled} | {estop} | Speed:{self.speed_scale:.1f}{collision_status_text}{obs_info}{joy_info}"
        
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)
    
    def _initialize_motors(self):
        """Initialize and enable motors"""
        try:
            self.get_logger().info("Initializing motors...")
            
            if not (self.motor_left.initialize() and self.motor_right.initialize()):
                self.get_logger().error("Motor initialization failed")
                return
            
            for motor in [self.motor_left, self.motor_right]:
                motor.set_acceleration(0.5)
                motor.set_deceleration(0.5)
            
            self.first_reading = True
            self.get_logger().info("Motors initialized")
            
            # Auto-enable motors
            time.sleep(1.0)
            self._enable_motors()
            
        except Exception as e:
            self.get_logger().error(f"Motor initialization failed: {e}")
    
    def _enable_motors(self):
        """Enable motor operation"""
        try:
            if self.motor_left.enable() and self.motor_right.enable():
                self.motors_enabled = True
                self.emergency_stop = False
                self.get_logger().info("✓ Motors enabled and ready")
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
        self.filtered_linear_vel = self.filtered_angular_vel = 0.0
        
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
    
    def _enable_collision_avoidance_service(self, request, response):
        """Enable collision avoidance service handler"""
        self.collision_avoidance_active = True
        self.get_logger().info("✓ Collision avoidance ENABLED")
        response.success = True
        response.message = "Collision avoidance enabled"
        return response
    
    def _disable_collision_avoidance_service(self, request, response):
        """Disable collision avoidance service handler"""
        self.collision_avoidance_active = False
        self.collision_warning = False
        self.last_collision_check = {'linear_scale': 1.0, 'angular_scale': 1.0}
        self.get_logger().warn("⚠️ Collision avoidance DISABLED")
        response.success = True
        response.message = "Collision avoidance disabled"
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