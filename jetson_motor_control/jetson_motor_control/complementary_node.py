#!/usr/bin/env python3
"""
Robust Collision Avoidance Motor Controller with Proper Geometry
Handles rectangular wheelchair shape with correct lidar positioning
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Joy
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
import math

class RobustCollisionController(Node):
    def __init__(self):
        super().__init__('robust_collision_controller')
        
        # Wheelchair dimensions (all in meters)
        # Base frame is at the middle of rear wheels
        self.declare_parameter('base_to_lidar_x', 0.54)  # Distance forward from base to lidar
        self.declare_parameter('lidar_to_front_x', 0.3)  # Distance from lidar to front edge
        self.declare_parameter('base_to_rear_x', 0.25)  # Distance from base to rear edge
        self.declare_parameter('wheelchair_width', 0.65)  # Total width
        
        # Safety parameters
        self.declare_parameter('safety_margin', 0.15)  # Extra safety buffer around wheelchair
        self.declare_parameter('critical_distance', 0.2)  # Emergency stop distance
        self.declare_parameter('safety_distance', 0.35)  # Start major slowdown
        self.declare_parameter('slowdown_distance', 0.6)  # Start gentle slowdown
        
        # Control parameters
        self.declare_parameter('joystick_deadzone', 0.1)
        self.declare_parameter('smooth_factor', 0.7)
        self.declare_parameter('publish_markers', True)  # For visualization in RViz
        
        # Get parameters
        self.base_to_lidar_x = self.get_parameter('base_to_lidar_x').value
        self.lidar_to_front_x = self.get_parameter('lidar_to_front_x').value
        self.base_to_rear_x = self.get_parameter('base_to_rear_x').value
        self.wheelchair_width = self.get_parameter('wheelchair_width').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.critical_distance = self.get_parameter('critical_distance').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.slowdown_distance = self.get_parameter('slowdown_distance').value
        self.joystick_deadzone = self.get_parameter('joystick_deadzone').value
        self.smooth_factor = self.get_parameter('smooth_factor').value
        self.publish_markers = self.get_parameter('publish_markers').value
        
        # Calculate total dimensions
        self.total_length = self.base_to_lidar_x + self.lidar_to_front_x + self.base_to_rear_x
        self.front_edge_x = self.base_to_lidar_x + self.lidar_to_front_x  # In base frame
        self.rear_edge_x = -self.base_to_rear_x  # In base frame
        self.half_width = self.wheelchair_width / 2.0
        
        # Wheelchair corners in base_link frame
        self.wheelchair_corners = [
            (self.front_edge_x, self.half_width),   # Front left
            (self.front_edge_x, -self.half_width),  # Front right
            (self.rear_edge_x, -self.half_width),   # Rear right
            (self.rear_edge_x, self.half_width)     # Rear left
        ]
        
        self.get_logger().info(f"Wheelchair geometry initialized:")
        self.get_logger().info(f"  Total length: {self.total_length:.2f}m")
        self.get_logger().info(f"  Width: {self.wheelchair_width:.2f}m")
        self.get_logger().info(f"  Base to lidar: {self.base_to_lidar_x:.2f}m")
        self.get_logger().info(f"  Front edge (from base): {self.front_edge_x:.2f}m")
        self.get_logger().info(f"  Rear edge (from base): {self.rear_edge_x:.2f}m")
        
        # State variables
        self.latest_scan = None
        self.joystick_x = 0.0
        self.joystick_y = 0.0
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.obstacles_in_base_frame = []
        self.collision_zones = {'front': [], 'rear': [], 'left': [], 'right': []}
        
        # QoS profile for best effort (matching lidar)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )
        
        self.joy_sub = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )
        
        # Publishers
        self.filtered_joy_pub = self.create_publisher(Joy, 'filtered_joy', 10)
        
        if self.publish_markers:
            self.marker_pub = self.create_publisher(MarkerArray, 'collision_zones', 10)
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz
        
        self.get_logger().info("Robust Collision Controller ready")
    
    def scan_callback(self, msg):
        """Process laser scan data and identify obstacles"""
        self.latest_scan = msg
        self.process_obstacles(msg)
    
    def joy_callback(self, msg):
        """Receive joystick commands"""
        if len(msg.axes) >= 2:
            self.joystick_x = msg.axes[0] if abs(msg.axes[0]) > self.joystick_deadzone else 0.0
            self.joystick_y = msg.axes[1] if abs(msg.axes[1]) > self.joystick_deadzone else 0.0
    
    def transform_lidar_to_base(self, x_lidar, y_lidar):
        """Transform point from lidar frame to base_link frame"""
        # Lidar is 0.54m forward from base_link, aligned in y
        x_base = x_lidar + self.base_to_lidar_x
        y_base = y_lidar
        return x_base, y_base
    
    def is_point_in_footprint(self, x_base, y_base):
        """Check if a point (in base frame) is inside the wheelchair footprint"""
        # Add safety margin to footprint
        margin = self.safety_margin
        
        # Check if point is within the rectangular footprint plus margin
        if (self.rear_edge_x - margin <= x_base <= self.front_edge_x + margin and
            -self.half_width - margin <= y_base <= self.half_width + margin):
            return True
        
        return False
    
    def calculate_distance_to_footprint(self, x_base, y_base):
        """Calculate minimum distance from point to wheelchair footprint edge"""
        # Distance to front edge
        dist_front = self.front_edge_x - x_base if x_base < self.front_edge_x else 0
        
        # Distance to rear edge
        dist_rear = x_base - self.rear_edge_x if x_base > self.rear_edge_x else 0
        
        # Distance to left edge
        dist_left = self.half_width - y_base if y_base < self.half_width else 0
        
        # Distance to right edge
        dist_right = y_base + self.half_width if y_base > -self.half_width else 0
        
        # If point is outside footprint in x direction
        if x_base > self.front_edge_x:
            dist_x = x_base - self.front_edge_x
        elif x_base < self.rear_edge_x:
            dist_x = self.rear_edge_x - x_base
        else:
            dist_x = 0
        
        # If point is outside footprint in y direction
        if y_base > self.half_width:
            dist_y = y_base - self.half_width
        elif y_base < -self.half_width:
            dist_y = -self.half_width - y_base
        else:
            dist_y = 0
        
        # Minimum distance to footprint
        if dist_x > 0 and dist_y > 0:
            # Point is diagonal from corner
            return math.sqrt(dist_x**2 + dist_y**2)
        else:
            # Point is aligned with a side
            return max(dist_x, dist_y)
    
    def classify_obstacle_zone(self, x_base, y_base):
        """Classify which zone an obstacle is in relative to wheelchair"""
        # Check primary zone based on position
        zones = []
        
        # Front zone (ahead of wheelchair front)
        if x_base > self.front_edge_x - 0.1:
            zones.append('front')
        
        # Rear zone (behind wheelchair center, accounting for blind zone)
        if x_base < 0:
            zones.append('rear')
        
        # Left zone
        if y_base > self.half_width - 0.1:
            zones.append('left')
        
        # Right zone  
        if y_base < -self.half_width + 0.1:
            zones.append('right')
        
        # If obstacle is within footprint bounds in one dimension
        # but outside in another, classify by the outside dimension
        if not zones:
            if abs(y_base) > abs(x_base - self.base_to_lidar_x):
                if y_base > 0:
                    zones.append('left')
                else:
                    zones.append('right')
            else:
                if x_base > self.base_to_lidar_x:
                    zones.append('front')
                else:
                    zones.append('rear')
        
        return zones
    
    def process_obstacles(self, scan):
        """Process scan data to identify obstacles in base frame"""
        self.obstacles_in_base_frame = []
        self.collision_zones = {'front': [], 'rear': [], 'left': [], 'right': []}
        
        if not scan.ranges:
            return
        
        for i, distance in enumerate(scan.ranges):
            # Skip invalid readings
            if math.isnan(distance) or math.isinf(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue
            
            # Skip very far obstacles
            if distance > self.slowdown_distance * 2:
                continue
            
            # Calculate angle of this reading (in lidar frame)
            angle = scan.angle_min + i * scan.angle_increment
            
            # Convert to cartesian in lidar frame
            x_lidar = distance * math.cos(angle)
            y_lidar = distance * math.sin(angle)
            
            # Transform to base_link frame
            x_base, y_base = self.transform_lidar_to_base(x_lidar, y_lidar)
            
            # Skip if inside footprint
            if self.is_point_in_footprint(x_base, y_base):
                continue
            
            # Calculate actual distance from footprint edge
            dist_to_footprint = self.calculate_distance_to_footprint(x_base, y_base)
            
            # Store obstacle with all information
            obstacle = {
                'x_base': x_base,
                'y_base': y_base,
                'x_lidar': x_lidar,
                'y_lidar': y_lidar,
                'distance_lidar': distance,
                'distance_to_footprint': dist_to_footprint,
                'angle': angle
            }
            
            self.obstacles_in_base_frame.append(obstacle)
            
            # Classify into zones
            zones = self.classify_obstacle_zone(x_base, y_base)
            for zone in zones:
                self.collision_zones[zone].append(obstacle)
        
        # Publish visualization markers if enabled
        if self.publish_markers:
            self.publish_visualization_markers()
    
    def check_collision_risk(self, joy_x, joy_y):
        """Check collision risk and return scaling factors"""
        if not self.obstacles_in_base_frame:
            return 1.0, 1.0
        
        linear_scale = 1.0
        angular_scale = 1.0
        
        # Analyze movement intent
        moving_forward = joy_y > 0.01
        moving_backward = joy_y < -0.01
        turning_left = joy_x > 0.01
        turning_right = joy_x < -0.01
        
        # Check FRONT obstacles when moving forward
        if moving_forward and self.collision_zones['front']:
            closest_front = min(self.collision_zones['front'], 
                              key=lambda o: o['distance_to_footprint'])
            dist = closest_front['distance_to_footprint']
            
            # Only consider obstacles roughly in path
            if abs(closest_front['y_base']) < self.half_width + 0.3:
                if dist < self.critical_distance:
                    linear_scale = 0.0
                    self.get_logger().warn(f"STOP! Front obstacle at {dist:.2f}m")
                elif dist < self.safety_distance:
                    linear_scale = min(linear_scale, 0.2)
                elif dist < self.slowdown_distance:
                    scale = (dist - self.safety_distance) / (self.slowdown_distance - self.safety_distance)
                    linear_scale = min(linear_scale, 0.2 + 0.8 * scale)
        
        # Check REAR obstacles when moving backward
        if moving_backward and self.collision_zones['rear']:
            # Extra caution for blind zone
            closest_rear = min(self.collision_zones['rear'], 
                             key=lambda o: o['distance_to_footprint'])
            dist = closest_rear['distance_to_footprint']
            
            # Only consider obstacles roughly in path
            if abs(closest_rear['y_base']) < self.half_width + 0.3:
                if dist < self.critical_distance:
                    linear_scale = 0.0
                    self.get_logger().warn(f"STOP! Rear obstacle at {dist:.2f}m")
                elif dist < self.safety_distance:
                    linear_scale = min(linear_scale, 0.15)  # Even slower backward
                elif dist < self.slowdown_distance:
                    scale = (dist - self.safety_distance) / (self.slowdown_distance - self.safety_distance)
                    linear_scale = min(linear_scale, 0.15 + 0.85 * scale)
        
        # Check LEFT obstacles when turning left
        if turning_left:
            left_obstacles = self.collision_zones['left']
            # Also check front-left obstacles
            for obs in self.collision_zones['front']:
                if obs['y_base'] > 0:  # Left side of front
                    left_obstacles.append(obs)
            
            if left_obstacles:
                # Focus on obstacles near the wheelchair body
                relevant_obstacles = [o for o in left_obstacles 
                                     if self.rear_edge_x - 0.2 < o['x_base'] < self.front_edge_x + 0.2]
                
                if relevant_obstacles:
                    closest = min(relevant_obstacles, key=lambda o: o['distance_to_footprint'])
                    dist = closest['distance_to_footprint']
                    
                    if dist < self.critical_distance:
                        angular_scale = 0.0
                        self.get_logger().warn(f"STOP! Left obstacle at {dist:.2f}m")
                    elif dist < self.safety_distance:
                        angular_scale = min(angular_scale, 0.3)
                    elif dist < self.slowdown_distance:
                        scale = (dist - self.safety_distance) / (self.slowdown_distance - self.safety_distance)
                        angular_scale = min(angular_scale, 0.3 + 0.7 * scale)
        
        # Check RIGHT obstacles when turning right
        if turning_right:
            right_obstacles = self.collision_zones['right']
            # Also check front-right obstacles
            for obs in self.collision_zones['front']:
                if obs['y_base'] < 0:  # Right side of front
                    right_obstacles.append(obs)
            
            if right_obstacles:
                # Focus on obstacles near the wheelchair body
                relevant_obstacles = [o for o in right_obstacles 
                                     if self.rear_edge_x - 0.2 < o['x_base'] < self.front_edge_x + 0.2]
                
                if relevant_obstacles:
                    closest = min(relevant_obstacles, key=lambda o: o['distance_to_footprint'])
                    dist = closest['distance_to_footprint']
                    
                    if dist < self.critical_distance:
                        angular_scale = 0.0
                        self.get_logger().warn(f"STOP! Right obstacle at {dist:.2f}m")
                    elif dist < self.safety_distance:
                        angular_scale = min(angular_scale, 0.3)
                    elif dist < self.slowdown_distance:
                        scale = (dist - self.safety_distance) / (self.slowdown_distance - self.safety_distance)
                        angular_scale = min(angular_scale, 0.3 + 0.7 * scale)
        
        # Handle diagonal movements (forward/backward with turn)
        if abs(joy_y) > 0.01 and abs(joy_x) > 0.01:
            # For combined movements, check the quadrant
            if moving_forward and turning_left:  # Front-left
                quadrant_obstacles = [o for o in self.obstacles_in_base_frame
                                    if o['x_base'] > self.base_to_lidar_x and o['y_base'] > 0]
                if quadrant_obstacles:
                    closest = min(quadrant_obstacles, key=lambda o: o['distance_to_footprint'])
                    if closest['distance_to_footprint'] < self.safety_distance:
                        linear_scale = min(linear_scale, 0.3)
                        angular_scale = min(angular_scale, 0.4)
            
            elif moving_forward and turning_right:  # Front-right
                quadrant_obstacles = [o for o in self.obstacles_in_base_frame
                                    if o['x_base'] > self.base_to_lidar_x and o['y_base'] < 0]
                if quadrant_obstacles:
                    closest = min(quadrant_obstacles, key=lambda o: o['distance_to_footprint'])
                    if closest['distance_to_footprint'] < self.safety_distance:
                        linear_scale = min(linear_scale, 0.3)
                        angular_scale = min(angular_scale, 0.4)
            
            elif moving_backward and turning_left:  # Rear-right (inverted)
                quadrant_obstacles = [o for o in self.obstacles_in_base_frame
                                    if o['x_base'] < 0 and o['y_base'] < 0]
                if quadrant_obstacles:
                    closest = min(quadrant_obstacles, key=lambda o: o['distance_to_footprint'])
                    if closest['distance_to_footprint'] < self.safety_distance:
                        linear_scale = min(linear_scale, 0.2)
                        angular_scale = min(angular_scale, 0.3)
            
            elif moving_backward and turning_right:  # Rear-left (inverted)
                quadrant_obstacles = [o for o in self.obstacles_in_base_frame
                                    if o['x_base'] < 0 and o['y_base'] > 0]
                if quadrant_obstacles:
                    closest = min(quadrant_obstacles, key=lambda o: o['distance_to_footprint'])
                    if closest['distance_to_footprint'] < self.safety_distance:
                        linear_scale = min(linear_scale, 0.2)
                        angular_scale = min(angular_scale, 0.3)
        
        return linear_scale, angular_scale
    
    def control_loop(self):
        """Main control loop - filter joystick commands based on obstacles"""
        if self.latest_scan is None:
            # No scan data yet, pass through joystick commands
            self.filtered_x = self.joystick_x
            self.filtered_y = self.joystick_y
        else:
            # Check collision risk and get scaling factors
            linear_scale, angular_scale = self.check_collision_risk(self.joystick_x, self.joystick_y)
            
            # Apply scaling to joystick commands
            target_x = self.joystick_x * angular_scale
            target_y = self.joystick_y * linear_scale
            
            # Smooth the filtering to avoid jerky motion
            self.filtered_x = self.smooth_factor * self.filtered_x + (1 - self.smooth_factor) * target_x
            self.filtered_y = self.smooth_factor * self.filtered_y + (1 - self.smooth_factor) * target_y
            
            # Log status if movement is restricted
            if linear_scale < 0.9 or angular_scale < 0.9:
                status = f"Collision avoidance active - Linear: {linear_scale:.1%}, Angular: {angular_scale:.1%}"
                if self.obstacles_in_base_frame:
                    closest = min(self.obstacles_in_base_frame, 
                                key=lambda o: o['distance_to_footprint'])
                    status += f" | Closest: {closest['distance_to_footprint']:.2f}m"
                    
                    # Report which zones have obstacles
                    active_zones = [zone for zone, obs_list in self.collision_zones.items() if obs_list]
                    if active_zones:
                        status += f" | Zones: {', '.join(active_zones)}"
                
                self.get_logger().info(status)
        
        # Publish filtered joy message
        filtered_joy = Joy()
        filtered_joy.header.stamp = self.get_clock().now().to_msg()
        filtered_joy.axes = [self.filtered_x, self.filtered_y, 0.0, 0.0]
        filtered_joy.buttons = []
        
        self.filtered_joy_pub.publish(filtered_joy)
    
    def publish_visualization_markers(self):
        """Publish markers for visualization in RViz"""
        marker_array = MarkerArray()
        
        # Clear old markers
        clear_marker = Marker()
        clear_marker.header.frame_id = "base_link"
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Wheelchair footprint
        footprint_marker = Marker()
        footprint_marker.header.frame_id = "base_link"
        footprint_marker.header.stamp = self.get_clock().now().to_msg()
        footprint_marker.id = 0
        footprint_marker.type = Marker.LINE_STRIP
        footprint_marker.action = Marker.ADD
        footprint_marker.scale.x = 0.02
        footprint_marker.color.r = 0.0
        footprint_marker.color.g = 1.0
        footprint_marker.color.b = 0.0
        footprint_marker.color.a = 1.0
        
        # Add wheelchair corners
        for corner in self.wheelchair_corners + [self.wheelchair_corners[0]]:
            p = Point()
            p.x = corner[0]
            p.y = corner[1]
            p.z = 0.0
            footprint_marker.points.append(p)
        
        marker_array.markers.append(footprint_marker)
        
        # Obstacle markers
        for i, obs in enumerate(self.obstacles_in_base_frame):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i + 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obs['x_base']
            marker.pose.position.y = obs['y_base']
            marker.pose.position.z = 0.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05
            
            # Color based on distance
            dist = obs['distance_to_footprint']
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
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    
    controller = RobustCollisionController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()