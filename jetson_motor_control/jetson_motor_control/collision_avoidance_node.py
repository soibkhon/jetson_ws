#!/usr/bin/env python3
"""
Comprehensive Collision Avoidance Node
Works with existing jetson_motor_controller.py to provide:
- Footprint-based self-detection filtering
- Intelligent directional collision detection
- Speed scaling that doesn't slow down for side obstacles
- Velocity command interception and scaling
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, Polygon, Point32
from std_msgs.msg import Float32, Bool, String
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading


class CollisionAvoidanceNode(Node):
    
    def __init__(self):
        super().__init__('collision_avoidance_node')
        
        self._declare_parameters()
        self._get_parameters()
        self._init_state()
        self._setup_ros_interfaces()
        
        self.get_logger().info("Comprehensive Collision Avoidance Node initialized")
        self.get_logger().info(f"Robot footprint: {self.robot_length:.2f}m x {self.robot_width:.2f}m")
        self.get_logger().info(f"Speed multiplier: {self.base_speed_multiplier:.1f}x")
        self.get_logger().info(f"Safety distances: {self.safety_distance:.2f}m / {self.warning_distance:.2f}m")
    
    def _declare_parameters(self):
        """Declare all ROS2 parameters"""
        # Robot physical dimensions
        self.declare_parameter('robot_width', 0.7)           # Wheelchair width (meters)
        self.declare_parameter('robot_length', 0.95)         # Wheelchair length (meters)
        self.declare_parameter('robot_center_offset', 0.2)   # Distance from rear axle to center
        self.declare_parameter('footprint_margin', 0.1)      # Extra safety margin around footprint
        
        # Collision detection parameters
        self.declare_parameter('safety_distance', 0.2)       # Emergency stop distance
        self.declare_parameter('warning_distance', 1.0)      # Start slowing down distance
        self.declare_parameter('critical_distance', 0.5)     # Critical slowdown distance
        
        # Sector definitions (degrees)
        self.declare_parameter('front_sector_angle', 45.0)   # Front detection cone (±22.5°)
        self.declare_parameter('side_sector_angle', 30.0)    # Side detection zones
        self.declare_parameter('rear_sector_angle', 45.0)    # Rear detection cone
        
        # Speed control parameters
        self.declare_parameter('base_speed_multiplier', 2.0) # Base speed increase
        self.declare_parameter('min_speed_factor', 0.15)     # Minimum speed when obstacle detected
        self.declare_parameter('emergency_stop_enabled', True)
        self.declare_parameter('gradual_slowdown_enabled', True)
        
        # Advanced filtering
        self.declare_parameter('min_valid_range', 0.15)      # Minimum valid laser range
        self.declare_parameter('max_valid_range', 10.0)      # Maximum valid laser range
        self.declare_parameter('noise_filter_enabled', True) # Enable noise filtering
        self.declare_parameter('noise_threshold', 0.05)      # Noise filter threshold
        
        # Topics
        self.declare_parameter('input_scan_topic', '/scan')
        self.declare_parameter('output_scan_topic', '/scan_filtered')
        self.declare_parameter('input_cmd_vel_topic', 'cmd_vel_nav')
        self.declare_parameter('output_cmd_vel_topic', 'cmd_vel')
        
        # Publishing options
        self.declare_parameter('publish_filtered_scan', True)
        self.declare_parameter('publish_collision_markers', True)
        self.declare_parameter('publish_debug_info', True)
        self.declare_parameter('collision_check_frequency', 20.0)
    
    def _get_parameters(self):
        """Retrieve all parameters"""
        # Robot dimensions
        self.robot_width = self.get_parameter('robot_width').value
        self.robot_length = self.get_parameter('robot_length').value
        self.robot_center_offset = self.get_parameter('robot_center_offset').value
        self.footprint_margin = self.get_parameter('footprint_margin').value
        
        # Collision parameters
        self.safety_distance = self.get_parameter('safety_distance').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.critical_distance = self.get_parameter('critical_distance').value
        
        # Sector angles (convert to radians)
        self.front_sector_angle = math.radians(self.get_parameter('front_sector_angle').value)
        self.side_sector_angle = math.radians(self.get_parameter('side_sector_angle').value)
        self.rear_sector_angle = math.radians(self.get_parameter('rear_sector_angle').value)
        
        # Speed control
        self.base_speed_multiplier = self.get_parameter('base_speed_multiplier').value
        self.min_speed_factor = self.get_parameter('min_speed_factor').value
        self.emergency_stop_enabled = self.get_parameter('emergency_stop_enabled').value
        self.gradual_slowdown_enabled = self.get_parameter('gradual_slowdown_enabled').value
        
        # Filtering
        self.min_valid_range = self.get_parameter('min_valid_range').value
        self.max_valid_range = self.get_parameter('max_valid_range').value
        self.noise_filter_enabled = self.get_parameter('noise_filter_enabled').value
        self.noise_threshold = self.get_parameter('noise_threshold').value
        
        # Topics
        self.input_scan_topic = self.get_parameter('input_scan_topic').value
        self.output_scan_topic = self.get_parameter('output_scan_topic').value
        self.input_cmd_vel_topic = self.get_parameter('input_cmd_vel_topic').value
        self.output_cmd_vel_topic = self.get_parameter('output_cmd_vel_topic').value
        
        # Publishing options
        self.publish_filtered_scan = self.get_parameter('publish_filtered_scan').value
        self.publish_collision_markers = self.get_parameter('publish_collision_markers').value
        self.publish_debug_info = self.get_parameter('publish_debug_info').value
        self.collision_check_frequency = self.get_parameter('collision_check_frequency').value
        
        # Calculate robot footprint points (rectangular)
        half_width = (self.robot_width + self.footprint_margin) / 2.0
        half_length = (self.robot_length + self.footprint_margin) / 2.0
        
        self.footprint_points = [
            (half_length, half_width),    # Front right
            (half_length, -half_width),   # Front left
            (-half_length, -half_width),  # Rear left
            (-half_length, half_width)    # Rear right
        ]
    
    def _init_state(self):
        """Initialize state variables"""
        # Collision detection state
        self.last_scan = None
        self.filtered_scan = None
        self.collision_detected = False
        self.collision_sectors = {'front': False, 'left': False, 'right': False, 'rear': False}
        self.sector_distances = {'front': float('inf'), 'left': float('inf'), 'right': float('inf'), 'rear': float('inf')}
        self.min_obstacle_distance = float('inf')
        self.primary_collision_direction = None
        
        # Command velocity state
        self.last_input_cmd_vel = Twist()
        self.current_speed_factors = {'linear': 1.0, 'angular': 1.0}
        self.cmd_vel_lock = threading.Lock()
        
        # Statistics
        self.scan_count = 0
        self.collision_count = 0
        self.filtered_points_count = 0
    
    def _setup_ros_interfaces(self):
        """Setup all ROS2 publishers, subscribers and timers"""
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self._scan_callback,
            sensor_qos
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.input_cmd_vel_topic,
            self._cmd_vel_callback,
            reliable_qos
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.output_cmd_vel_topic,
            reliable_qos
        )
        
        if self.publish_filtered_scan:
            self.filtered_scan_pub = self.create_publisher(
                LaserScan,
                self.output_scan_topic,
                sensor_qos
            )
        
        if self.publish_debug_info:
            self.collision_status_pub = self.create_publisher(Bool, 'collision_detected', reliable_qos)
            self.min_distance_pub = self.create_publisher(Float32, 'min_obstacle_distance', reliable_qos)
            self.collision_direction_pub = self.create_publisher(String, 'collision_direction', reliable_qos)
            self.speed_factor_pub = self.create_publisher(Float32, 'speed_factor', reliable_qos)
        
        # Timers
        self.collision_check_timer = self.create_timer(
            1.0 / self.collision_check_frequency,
            self._collision_check_timer_callback
        )
        
        if self.publish_debug_info:
            self.status_timer = self.create_timer(1.0, self._publish_status)
        
        self.get_logger().info(f"Subscribed to: {self.input_scan_topic} -> {self.input_cmd_vel_topic}")
        self.get_logger().info(f"Publishing to: {self.output_scan_topic} -> {self.output_cmd_vel_topic}")
    
    def _scan_callback(self, msg):
        """Process incoming laser scan data"""
        self.last_scan = msg
        self.scan_count += 1
        
        # Log first scan info
        if self.scan_count == 1:
            self.get_logger().info(f"First scan received:")
            self.get_logger().info(f"  Points: {len(msg.ranges)}")
            self.get_logger().info(f"  FOV: {math.degrees(msg.angle_min):.1f}° to {math.degrees(msg.angle_max):.1f}°")
            self.get_logger().info(f"  Range: {msg.range_min:.2f}m to {msg.range_max:.2f}m")
            self.get_logger().info(f"  Frame: {msg.header.frame_id}")
        
        # Filter the scan data
        self.filtered_scan = self._filter_scan_data(msg)
        
        # Publish filtered scan if enabled
        if self.publish_filtered_scan and self.filtered_scan:
            self.filtered_scan_pub.publish(self.filtered_scan)
        
        # Update collision detection
        self._update_collision_detection()
    
    def _filter_scan_data(self, scan_msg):
        """Filter scan data to remove robot footprint and noise"""
        if not scan_msg.ranges:
            return None
        
        # Convert to numpy arrays
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
        # Handle invalid values
        valid_mask = np.isfinite(ranges)
        ranges = np.where(valid_mask, ranges, scan_msg.range_max)
        
        # Apply range limits
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)
        
        # Filter out points too close (likely noise or self-detection)
        ranges = np.where(ranges < self.min_valid_range, scan_msg.range_max, ranges)
        
        # Footprint filtering - remove points inside robot footprint
        original_count = np.sum(ranges < scan_msg.range_max)
        
        for i, (angle, range_val) in enumerate(zip(angles, ranges)):
            if range_val >= scan_msg.range_max:
                continue
                
            # Convert to cartesian coordinates (laser frame)
            x = range_val * math.cos(angle)
            y = range_val * math.sin(angle)
            
            # Check if point is inside robot footprint
            if self._point_in_footprint(x, y):
                ranges[i] = scan_msg.range_max
        
        # Apply noise filtering if enabled
        if self.noise_filter_enabled:
            ranges = self._apply_noise_filter(ranges, scan_msg.range_max)
        
        # Count filtered points
        filtered_count = np.sum(ranges < scan_msg.range_max)
        self.filtered_points_count = original_count - filtered_count
        
        # Create filtered scan message
        filtered_scan = LaserScan()
        filtered_scan.header = scan_msg.header
        filtered_scan.angle_min = scan_msg.angle_min
        filtered_scan.angle_max = scan_msg.angle_max
        filtered_scan.angle_increment = scan_msg.angle_increment
        filtered_scan.time_increment = scan_msg.time_increment
        filtered_scan.scan_time = scan_msg.scan_time
        filtered_scan.range_min = scan_msg.range_min
        filtered_scan.range_max = scan_msg.range_max
        filtered_scan.ranges = ranges.tolist()
        filtered_scan.intensities = scan_msg.intensities if scan_msg.intensities else []
        
        return filtered_scan
    
    def _point_in_footprint(self, x, y):
        """Check if a point (x,y) is inside the robot footprint using ray casting"""
        # Simple rectangular footprint check
        half_width = (self.robot_width + self.footprint_margin) / 2.0
        half_length = (self.robot_length + self.footprint_margin) / 2.0
        
        # Adjust for robot center offset (laser typically at front)
        adjusted_x = x + self.robot_center_offset
        
        return (abs(y) <= half_width and 
                -half_length <= adjusted_x <= half_length)
    
    def _apply_noise_filter(self, ranges, max_range):
        """Apply median filter to reduce noise"""
        window_size = 3
        filtered_ranges = np.copy(ranges)
        
        for i in range(window_size, len(ranges) - window_size):
            if ranges[i] < max_range:
                window = ranges[i-window_size:i+window_size+1]
                valid_window = window[window < max_range]
                if len(valid_window) >= window_size:
                    median_val = np.median(valid_window)
                    if abs(ranges[i] - median_val) > self.noise_threshold:
                        filtered_ranges[i] = median_val
        
        return filtered_ranges
    
    def _update_collision_detection(self):
        """Update collision detection state based on filtered scan"""
        if not self.filtered_scan:
            return
        
        ranges = np.array(self.filtered_scan.ranges)
        angles = np.linspace(self.filtered_scan.angle_min, self.filtered_scan.angle_max, len(ranges))
        max_range = self.filtered_scan.range_max
        
        # Reset sector states
        for sector in self.collision_sectors:
            self.collision_sectors[sector] = False
            self.sector_distances[sector] = float('inf')
        
        # Define sector boundaries
        sectors = {
            'front': (-self.front_sector_angle/2, self.front_sector_angle/2),
            'left': (self.front_sector_angle/2, self.front_sector_angle/2 + self.side_sector_angle),
            'right': (-self.front_sector_angle/2 - self.side_sector_angle, -self.front_sector_angle/2),
            'rear': (math.pi - self.rear_sector_angle/2, math.pi + self.rear_sector_angle/2)
        }
        
        # Check each sector
        for sector_name, (angle_min, angle_max) in sectors.items():
            sector_ranges = []
            
            for angle, range_val in zip(angles, ranges):
                # Normalize angle to [-pi, pi]
                angle_norm = math.atan2(math.sin(angle), math.cos(angle))
                
                # Check if angle is in sector (handle wraparound for rear)
                if sector_name == 'rear':
                    if angle_norm >= angle_min or angle_norm <= angle_max - 2*math.pi:
                        if range_val < max_range:
                            sector_ranges.append(range_val)
                else:
                    if angle_min <= angle_norm <= angle_max:
                        if range_val < max_range:
                            sector_ranges.append(range_val)
            
            # Update sector state
            if sector_ranges:
                min_distance = min(sector_ranges)
                self.sector_distances[sector_name] = min_distance
                
                if min_distance < self.warning_distance:
                    self.collision_sectors[sector_name] = True
        
        # Update global collision state
        previous_collision = self.collision_detected
        self.collision_detected = any(self.collision_sectors.values())
        self.min_obstacle_distance = min(self.sector_distances.values())
        
        # Determine primary collision direction
        if self.collision_detected:
            min_sector = min(self.sector_distances, key=self.sector_distances.get)
            self.primary_collision_direction = min_sector
            
            if not previous_collision:
                self.collision_count += 1
                self.get_logger().warn(
                    f"Collision detected in {self.primary_collision_direction} sector at {self.min_obstacle_distance:.2f}m"
                )
        else:
            self.primary_collision_direction = None
            if previous_collision:
                self.get_logger().info("All collision sectors clear")
    
    def _cmd_vel_callback(self, msg):
        """Process incoming cmd_vel and apply collision avoidance"""
        with self.cmd_vel_lock:
            self.last_input_cmd_vel = msg
            
            # Apply base speed multiplier first
            scaled_cmd = Twist()
            scaled_cmd.linear.x = msg.linear.x * self.base_speed_multiplier
            scaled_cmd.linear.y = msg.linear.y * self.base_speed_multiplier
            scaled_cmd.linear.z = msg.linear.z
            scaled_cmd.angular.x = msg.angular.x
            scaled_cmd.angular.y = msg.angular.y
            scaled_cmd.angular.z = msg.angular.z * self.base_speed_multiplier
            
            # Apply collision avoidance
            final_cmd = self._apply_collision_avoidance(scaled_cmd)
            
            # Publish processed command
            self.cmd_vel_pub.publish(final_cmd)
    
    def _apply_collision_avoidance(self, cmd_vel):
        """Apply intelligent collision avoidance to cmd_vel"""
        if not self.collision_detected:
            self.current_speed_factors = {'linear': 1.0, 'angular': 1.0}
            return cmd_vel
        
        # Determine motion direction
        linear_vel = cmd_vel.linear.x
        angular_vel = cmd_vel.angular.z
        
        # Calculate speed factors based on motion direction and obstacle location
        linear_factor = self._calculate_linear_speed_factor(linear_vel)
        angular_factor = self._calculate_angular_speed_factor(angular_vel)
        
        # Apply speed factors
        modified_cmd = Twist()
        modified_cmd.linear.x = cmd_vel.linear.x * linear_factor
        modified_cmd.linear.y = cmd_vel.linear.y * linear_factor
        modified_cmd.linear.z = cmd_vel.linear.z
        modified_cmd.angular.x = cmd_vel.angular.x
        modified_cmd.angular.y = cmd_vel.angular.y
        modified_cmd.angular.z = cmd_vel.angular.z * angular_factor
        
        # Store current factors for debugging
        self.current_speed_factors = {'linear': linear_factor, 'angular': angular_factor}
        
        # Log significant speed reductions
        if linear_factor < 0.5 or angular_factor < 0.5:
            self.get_logger().warn(
                f"Speed reduction applied - Linear: {linear_factor:.2f}x, Angular: {angular_factor:.2f}x "
                f"(obstacle: {self.primary_collision_direction} at {self.min_obstacle_distance:.2f}m)"
            )
        
        return modified_cmd
    
    def _calculate_linear_speed_factor(self, linear_vel):
        """Calculate speed factor for linear motion based on collision direction"""
        if abs(linear_vel) < 0.01:  # No significant linear motion
            return 1.0
        
        # Forward motion - check front obstacles
        if linear_vel > 0:
            if self.collision_sectors['front']:
                return self._get_distance_based_factor(self.sector_distances['front'])
            else:
                return 1.0  # No front obstacles, full speed
        
        # Backward motion - check rear obstacles  
        else:
            if self.collision_sectors['rear']:
                return self._get_distance_based_factor(self.sector_distances['rear'])
            else:
                return 1.0  # No rear obstacles, full speed
    
    def _calculate_angular_speed_factor(self, angular_vel):
        """Calculate speed factor for angular motion based on collision direction"""
        if abs(angular_vel) < 0.01:  # No significant angular motion
            return 1.0
        
        # Turning left (positive angular velocity)
        if angular_vel > 0:
            if self.collision_sectors['left']:
                return self._get_distance_based_factor(self.sector_distances['left'])
            else:
                return 1.0  # No left obstacles, full speed
        
        # Turning right (negative angular velocity)
        else:
            if self.collision_sectors['right']:
                return self._get_distance_based_factor(self.sector_distances['right'])
            else:
                return 1.0  # No right obstacles, full speed
    
    def _get_distance_based_factor(self, distance):
        """Calculate speed factor based on obstacle distance"""
        if not self.gradual_slowdown_enabled:
            return 1.0 if distance > self.safety_distance else 0.0
        
        if distance <= self.safety_distance:
            return 0.0 if self.emergency_stop_enabled else self.min_speed_factor
        elif distance <= self.critical_distance:
            # Steep reduction in critical zone
            factor = (distance - self.safety_distance) / (self.critical_distance - self.safety_distance)
            return self.min_speed_factor + (0.5 - self.min_speed_factor) * factor
        elif distance <= self.warning_distance:
            # Gradual reduction in warning zone
            factor = (distance - self.critical_distance) / (self.warning_distance - self.critical_distance)
            return 0.5 + 0.5 * factor
        else:
            return 1.0
    
    def _collision_check_timer_callback(self):
        """Periodic collision check and emergency handling"""
        if self.collision_detected and self.min_obstacle_distance <= self.safety_distance:
            # Emergency situation - ensure stopped
            if (abs(self.last_input_cmd_vel.linear.x) > 0.01 or 
                abs(self.last_input_cmd_vel.angular.z) > 0.01):
                
                emergency_cmd = Twist()  # Zero velocity
                self.cmd_vel_pub.publish(emergency_cmd)
                
                if self.scan_count % int(self.collision_check_frequency) == 0:  # Log once per second
                    self.get_logger().error(
                        f"EMERGENCY STOP: Obstacle at {self.min_obstacle_distance:.2f}m in {self.primary_collision_direction} sector"
                    )
    
    def _publish_status(self):
        """Publish debug information"""
        if not self.publish_debug_info:
            return
        
        # Collision status
        collision_msg = Bool()
        collision_msg.data = self.collision_detected
        self.collision_status_pub.publish(collision_msg)
        
        # Minimum distance
        distance_msg = Float32()
        distance_msg.data = float(self.min_obstacle_distance) if not math.isinf(self.min_obstacle_distance) else 100.0
        self.min_distance_pub.publish(distance_msg)
        
        # Collision direction
        direction_msg = String()
        direction_msg.data = self.primary_collision_direction or "none"
        self.collision_direction_pub.publish(direction_msg)
        
        # Current speed factor
        factor_msg = Float32()
        min_factor = min(self.current_speed_factors['linear'], self.current_speed_factors['angular'])
        factor_msg.data = float(min_factor)
        self.speed_factor_pub.publish(factor_msg)
        
        # Periodic status log
        if self.scan_count % 50 == 0 and self.scan_count > 0:  # Every ~2.5 seconds at 20Hz
            self.get_logger().info(
                f"Status: Scans={self.scan_count}, Collisions={self.collision_count}, "
                f"Filtered_points={self.filtered_points_count}, "
                f"Speed_factor={min_factor:.2f}, "
                f"Min_distance={self.min_obstacle_distance:.2f}m"
            )


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = CollisionAvoidanceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in collision avoidance node: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()