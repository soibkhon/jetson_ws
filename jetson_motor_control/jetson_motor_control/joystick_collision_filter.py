#!/usr/bin/env python3
"""
Simple Collision Monitor
- Two LiDAR sensors fused into one /scan topic with TF frame between them
- Continuously monitors for obstacles and sends immediate stop commands
- Simple footprint removal and direction-based collision detection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class SimpleCollisionMonitor(Node):
    
    def __init__(self):
        super().__init__('simple_collision_monitor')
        
        # Simple parameters
        self.wheelchair_width = 0.7      # Total width
        self.wheelchair_length = 1.0     # Total length  
        self.safety_distance = 0.4       # Stop distance
        self.min_scan_range = 0.1        # Use LiDAR minimum range
        
        # Current movement state
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        self.last_cmd_time = self.get_clock().now()
        
        # Setup ROS interfaces
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_callback, qos)
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel_input', self._cmd_vel_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.collision_pub = self.create_publisher(Bool, 'collision_detected', 10)
        
        # High-frequency timer for continuous monitoring
        self.monitor_timer = self.create_timer(0.02, self._monitor_collision)  # 50Hz
        
        self.scan_data = None
        self.get_logger().info("Simple Collision Monitor initialized")
        self.get_logger().info(f"Safety distance: {self.safety_distance}m, Min range: {self.min_scan_range}m")
    
    def _scan_callback(self, msg):
        """Store latest scan data"""
        self.scan_data = msg
    
    def _cmd_vel_callback(self, msg):
        """Store current movement commands"""
        self.current_linear_x = msg.linear.x
        self.current_angular_z = msg.angular.z
        self.last_cmd_time = self.get_clock().now()
    
    def _monitor_collision(self):
        """Continuously monitor for collisions and stop if needed"""
        if not self.scan_data:
            return
        
        # Check if we have recent movement commands
        time_since_cmd = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if time_since_cmd > 0.5:
            # No recent commands, publish zero velocity
            self._publish_cmd_vel(0.0, 0.0)
            return
        
        # Filter wheelchair footprint from scan
        filtered_ranges = self._filter_footprint(self.scan_data)
        
        # Check for collision in movement direction
        collision_detected = self._check_collision(filtered_ranges)
        
        # Publish collision status
        collision_msg = Bool()
        collision_msg.data = bool(collision_detected)  # Explicit bool conversion
        self.collision_pub.publish(collision_msg)
        
        if collision_detected:
            # IMMEDIATE STOP
            self._publish_cmd_vel(0.0, 0.0)
            self.get_logger().warn("COLLISION DETECTED - STOPPING WHEELCHAIR")
        else:
            # Forward the command
            self._publish_cmd_vel(self.current_linear_x, self.current_angular_z)
    
    def _filter_footprint(self, scan_msg):
        """Remove wheelchair footprint from scan data"""
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
        # Clean invalid data
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges), scan_msg.range_max, ranges)
        
        # Filter wheelchair body (simple rectangular footprint)
        half_width = self.wheelchair_width / 2.0
        half_length = self.wheelchair_length / 2.0
        
        for i, (angle, range_val) in enumerate(zip(angles, ranges)):
            if range_val >= scan_msg.range_max or range_val < self.min_scan_range:
                continue
            
            # Convert to cartesian (TF frame is between the two LiDAR sensors)
            x = range_val * math.cos(angle)
            y = range_val * math.sin(angle)
            
            # Remove points inside wheelchair footprint
            if abs(x) <= half_length and abs(y) <= half_width:
                ranges[i] = scan_msg.range_max
        
        return ranges
    
    def _check_collision(self, ranges):
        """Check for collision based on current movement direction"""
        if not self.scan_data:
            return False
        
        angles = np.linspace(self.scan_data.angle_min, self.scan_data.angle_max, len(ranges))
        
        # Determine movement direction
        if abs(self.current_linear_x) < 0.05 and abs(self.current_angular_z) < 0.05:
            return False  # Not moving significantly
        
        # Check direction based on movement
        if abs(self.current_linear_x) > abs(self.current_angular_z):
            # Linear movement dominant
            if self.current_linear_x > 0:
                # Moving forward - check front
                return self._check_sector(ranges, angles, -math.pi/6, math.pi/6)  # ±30°
            else:
                # Moving backward - check rear
                return self._check_sector(ranges, angles, 5*math.pi/6, -5*math.pi/6)  # ±150°
        else:
            # Angular movement dominant
            if self.current_angular_z > 0:
                # Turning left - check left side
                return self._check_sector(ranges, angles, math.pi/3, 2*math.pi/3)  # 60° to 120°
            else:
                # Turning right - check right side
                return self._check_sector(ranges, angles, -2*math.pi/3, -math.pi/3)  # -120° to -60°
    
    def _check_sector(self, ranges, angles, angle_min, angle_max):
        """Check if any obstacle in sector is closer than safety distance"""
        # Handle angle wraparound
        if angle_min > angle_max:
            # Wraparound case (e.g., rear sector)
            sector_ranges = []
            for angle, range_val in zip(angles, ranges):
                if range_val >= self.scan_data.range_max:
                    continue
                if angle >= angle_min or angle <= angle_max:
                    sector_ranges.append(range_val)
        else:
            # Normal case
            sector_ranges = []
            for angle, range_val in zip(angles, ranges):
                if range_val >= self.scan_data.range_max:
                    continue
                if angle_min <= angle <= angle_max:
                    sector_ranges.append(range_val)
        
        if sector_ranges:
            min_distance = min(sector_ranges)
            return min_distance < self.safety_distance
        
        return False
    
    def _publish_cmd_vel(self, linear_x, angular_z):
        """Publish velocity command"""
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_x
        cmd_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleCollisionMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()