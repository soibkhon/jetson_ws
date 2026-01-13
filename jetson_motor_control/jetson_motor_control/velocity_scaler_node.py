#!/usr/bin/env python3
"""
Velocity Scaler Node
Separate ROS2 node that scales cmd_vel based on collision detection
Increases base speeds and provides smooth collision avoidance
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool, String
import math


class VelocityScalerNode(Node):
    
    def __init__(self):
        super().__init__('velocity_scaler_node')
        
        self._declare_params()
        self._get_params()
        self._init_state()
        self._setup_ros_interfaces()
        
        self.get_logger().info("Velocity Scaler Node initialized")
    
    def _declare_params(self):
        """Declare ROS2 parameters"""
        # Speed scaling parameters
        self.declare_parameter('base_linear_speed_multiplier', 2.0)   # Make it faster
        self.declare_parameter('base_angular_speed_multiplier', 2.0)  # Make turning faster
        
        # Collision avoidance parameters
        self.declare_parameter('safety_distance', 0.3)
        self.declare_parameter('warning_distance', 1.0)
        self.declare_parameter('min_speed_factor', 0.1)      # Minimum speed when obstacle detected
        self.declare_parameter('emergency_stop_factor', 0.0) # Complete stop factor
        
        # Joystick speed scaling
        self.declare_parameter('joystick_speed_multiplier', 1.5)  # Make joystick faster too
        
        # Topics
        self.declare_parameter('input_cmd_vel_topic', 'cmd_vel_nav')     # From Nav2
        self.declare_parameter('output_cmd_vel_topic', 'cmd_vel')        # To motor controller
        self.declare_parameter('joystick_cmd_vel_topic', 'cmd_vel_joy')  # Optional: from joystick
    
    def _get_params(self):
        """Get parameter values"""
        self.base_linear_multiplier = self.get_parameter('base_linear_speed_multiplier').value
        self.base_angular_multiplier = self.get_parameter('base_angular_speed_multiplier').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.min_speed_factor = self.get_parameter('min_speed_factor').value
        self.emergency_stop_factor = self.get_parameter('emergency_stop_factor').value
        self.joystick_speed_multiplier = self.get_parameter('joystick_speed_multiplier').value
        
        # Topics
        self.input_cmd_vel_topic = self.get_parameter('input_cmd_vel_topic').value
        self.output_cmd_vel_topic = self.get_parameter('output_cmd_vel_topic').value
        self.joystick_cmd_vel_topic = self.get_parameter('joystick_cmd_vel_topic').value
    
    def _init_state(self):
        """Initialize state variables"""
        self.collision_detected = False
        self.collision_direction = None
        self.min_obstacle_distance = float('inf')
        self.last_cmd_vel_time = self.get_clock().now()
        
        # Latest command velocities
        self.current_cmd_vel = Twist()
    
    def _setup_ros_interfaces(self):
        """Setup ROS2 interfaces"""
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 
            self.input_cmd_vel_topic, 
            self._cmd_vel_callback, 
            10
        )
        
        self.collision_status_sub = self.create_subscription(
            Bool,
            'collision_detected',
            self._collision_status_callback,
            10
        )
        
        self.collision_distance_sub = self.create_subscription(
            Float32,
            'min_obstacle_distance',
            self._collision_distance_callback,
            10
        )
        
        self.collision_direction_sub = self.create_subscription(
            String,
            'collision_direction',
            self._collision_direction_callback,
            10
        )
        
        # Publishers
        self.scaled_cmd_vel_pub = self.create_publisher(Twist, self.output_cmd_vel_topic, 10)
        
        # Timer for processing
        self.processing_timer = self.create_timer(0.05, self._process_and_publish)  # 20 Hz
        
        self.get_logger().info(f"Subscribed to: {self.input_cmd_vel_topic}")
        self.get_logger().info(f"Publishing to: {self.output_cmd_vel_topic}")
        self.get_logger().info(f"Base speed multipliers - Linear: {self.base_linear_multiplier}x, Angular: {self.base_angular_multiplier}x")
    
    def _cmd_vel_callback(self, msg):
        """Handle incoming cmd_vel from Nav2 or other sources"""
        self.current_cmd_vel = msg
        self.last_cmd_vel_time = self.get_clock().now()
    
    def _collision_status_callback(self, msg):
        """Handle collision status updates"""
        try:
            self.collision_detected = bool(msg.data)
        except Exception as e:
            self.get_logger().error(f"Error processing collision status: {e}")
            self.collision_detected = False
    
    def _collision_distance_callback(self, msg):
        """Handle minimum obstacle distance updates"""
        try:
            distance = float(msg.data)
            # Handle large values (converted from inf)
            if distance > 50.0:
                self.min_obstacle_distance = float('inf')
            else:
                self.min_obstacle_distance = distance
        except Exception as e:
            self.get_logger().error(f"Error processing collision distance: {e}")
            self.min_obstacle_distance = float('inf')
    
    def _collision_direction_callback(self, msg):
        """Handle collision direction updates"""
        try:
            direction = str(msg.data)
            self.collision_direction = direction if direction != "none" else None
        except Exception as e:
            self.get_logger().error(f"Error processing collision direction: {e}")
            self.collision_direction = None
    
    def _process_and_publish(self):
        """Process cmd_vel and publish scaled version"""
        # Check if we have recent cmd_vel
        time_since_cmd = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds / 1e9
        if time_since_cmd > 0.5:  # 500ms timeout
            # No recent commands, publish zero
            zero_cmd = Twist()
            self.scaled_cmd_vel_pub.publish(zero_cmd)
            return
        
        # Scale the base speeds (make them faster)
        scaled_cmd = Twist()
        scaled_cmd.linear.x = self.current_cmd_vel.linear.x * self.base_linear_multiplier
        scaled_cmd.linear.y = self.current_cmd_vel.linear.y * self.base_linear_multiplier
        scaled_cmd.linear.z = self.current_cmd_vel.linear.z
        scaled_cmd.angular.x = self.current_cmd_vel.angular.x
        scaled_cmd.angular.y = self.current_cmd_vel.angular.y
        scaled_cmd.angular.z = self.current_cmd_vel.angular.z * self.base_angular_multiplier
        
        # Apply collision avoidance scaling
        if self.collision_detected:
            linear_factor, angular_factor = self._calculate_collision_factors(
                scaled_cmd.linear.x, scaled_cmd.angular.z
            )
            
            scaled_cmd.linear.x *= linear_factor
            scaled_cmd.linear.y *= linear_factor
            scaled_cmd.angular.z *= angular_factor
            
            # Log significant speed reductions
            if linear_factor < 0.5 or angular_factor < 0.5:
                reduction = int((1.0 - min(linear_factor, angular_factor)) * 100)
                self.get_logger().warn(
                    f"Speed reduced {reduction}% - obstacle at {self.min_obstacle_distance:.2f}m ({self.collision_direction})"
                )
        
        # Publish scaled command
        self.scaled_cmd_vel_pub.publish(scaled_cmd)
    
    def _calculate_collision_factors(self, linear_vel, angular_vel):
        """Calculate speed reduction factors based on collision detection"""
        linear_factor = 1.0
        angular_factor = 1.0
        
        if not self.collision_detected or math.isinf(self.min_obstacle_distance):
            return linear_factor, angular_factor
        
        # Determine motion direction
        forward_motion = linear_vel > 0.1
        backward_motion = linear_vel < -0.1
        turning_left = angular_vel > 0.1
        turning_right = angular_vel < -0.1
        
        # Emergency stop if too close
        if self.min_obstacle_distance < self.safety_distance:
            if forward_motion and self.collision_direction in ['front', 'left', 'right']:
                linear_factor = self.emergency_stop_factor
            if turning_left and self.collision_direction == 'left':
                angular_factor = self.emergency_stop_factor
            if turning_right and self.collision_direction == 'right':
                angular_factor = self.emergency_stop_factor
        else:
            # Gradual slowdown in warning zone
            distance_factor = (self.min_obstacle_distance - self.safety_distance) / (self.warning_distance - self.safety_distance)
            distance_factor = max(0.0, min(1.0, distance_factor))
            
            # Calculate speed factor (smooth transition from min_speed_factor to 1.0)
            speed_factor = self.min_speed_factor + (1.0 - self.min_speed_factor) * distance_factor
            
            # Apply to relevant motions
            if forward_motion and self.collision_direction in ['front', 'left', 'right']:
                linear_factor = speed_factor
            
            if turning_left and self.collision_direction == 'left':
                angular_factor = speed_factor
            elif turning_right and self.collision_direction == 'right':
                angular_factor = speed_factor
            
            # General proximity slowdown (affects all motion when very close)
            if self.min_obstacle_distance < self.warning_distance * 0.7:
                proximity_factor = max(0.3, self.min_obstacle_distance / (self.warning_distance * 0.7))
                linear_factor = min(linear_factor, proximity_factor)
                angular_factor = min(angular_factor, proximity_factor)
        
        return linear_factor, angular_factor


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = VelocityScalerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()