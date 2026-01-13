#!/usr/bin/env python3
"""
L2DB Motor Driver - Low Level CAN Controller
Provides CANopen SDO protocol interface for L2DB motor drivers
Updated with proper RPM reading and PID control
"""

import can
import time
import threading
from collections import deque


class L2DBMotorDriver:
    """Low-level CAN driver for L2DB motor controllers with PID control"""
    
    # Class variable to share CAN bus instance between motors
    _bus_instance = None
    
    def __init__(self, interface='socketcan', channel='can1', node_id=1):
        """
        Initialize CAN bus connection for motor driver
        
        Args:
            interface: CAN interface type (default: 'socketcan')
            channel: CAN channel (default: 'can0')
            node_id: Motor node ID
        """
        self.node_id = node_id
        self.tx_cob_id = 0x600 + node_id  # SDO Transmit COB-ID
        self.rx_cob_id = 0x580 + node_id  # SDO Receive COB-ID
        
        # PID control variables
        self.target_rpm = 0
        self.last_error = 0
        self.integral_error = 0
        self.last_time = time.time()
        
        # PID gains (can be tuned)
        self.kp = 1.0
        self.ki = 0.01
        self.kd = 0
        
        # Data logging for plotting
        self.data_lock = threading.Lock()
        self.max_data_points = 1000
        self.time_data = deque(maxlen=self.max_data_points)
        self.target_rpm_data = deque(maxlen=self.max_data_points)
        self.actual_rpm_data = deque(maxlen=self.max_data_points)
        self.error_data = deque(maxlen=self.max_data_points)
        self.pid_output_data = deque(maxlen=self.max_data_points)
        self.current_data = deque(maxlen=self.max_data_points)
        self.logging_enabled = False
        
        # Initialize shared CAN bus if not already done
        if L2DBMotorDriver._bus_instance is None:
            try:
                L2DBMotorDriver._bus_instance = can.interface.Bus(interface=interface, channel=channel)
                print(f"CAN bus initialized on {channel}")
            except Exception as e:
                print(f"Failed to initialize CAN bus: {e}")
                raise
        
        self.bus = L2DBMotorDriver._bus_instance
        print(f"Motor driver created for node ID {node_id}")
    
    def send_sdo_write(self, index, subindex, data, data_size):
        """
        Send SDO write command
        
        Args:
            index: Object index (16-bit)
            subindex: Object subindex (8-bit)
            data: Data to write (up to 4 bytes)
            data_size: Size of data (1, 2, or 4 bytes)
            
        Returns:
            bool: True if write successful, False otherwise
        """
        # Determine command specifier based on data size
        cmd_map = {1: 0x2F, 2: 0x2B, 4: 0x23}
        cmd = cmd_map.get(data_size, 0x23)
        
        # Prepare message data
        msg_data = [0] * 8
        msg_data[0] = cmd
        msg_data[1] = index & 0xFF        # Index LSB
        msg_data[2] = (index >> 8) & 0xFF # Index MSB
        msg_data[3] = subindex
        
        # Add data bytes (little-endian)
        for i in range(data_size):
            msg_data[4 + i] = (data >> (i * 8)) & 0xFF
        
        try:
            # Send message
            message = can.Message(arbitration_id=self.tx_cob_id, data=msg_data, is_extended_id=False)
            self.bus.send(message)
            
            # Wait for response
            response = self._wait_for_response()
            return response is not None
        except Exception as e:
            print(f"SDO write failed for node {self.node_id}: {e}")
            return False
    
    def send_sdo_read(self, index, subindex):
        """
        Send SDO read command
        
        Args:
            index: Object index (16-bit)
            subindex: Object subindex (8-bit)
            
        Returns:
            Response message or None if failed
        """
        msg_data = [0x40, index & 0xFF, (index >> 8) & 0xFF, subindex, 0, 0, 0, 0]
        
        try:
            message = can.Message(arbitration_id=self.tx_cob_id, data=msg_data, is_extended_id=False)
            self.bus.send(message)
            
            return self._wait_for_response()
        except Exception as e:
            print(f"SDO read failed for node {self.node_id}: {e}")
            return None
    
    def _wait_for_response(self, timeout=1.0):
        """
        Wait for SDO response
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Response message or None if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                message = self.bus.recv(timeout=0.1)
                if message and message.arbitration_id == self.rx_cob_id:
                    return message
            except Exception:
                continue
        return None
    
    def set_can_id(self, can_id):
        """
        Set CAN ID (needed when SW3 is all OFF)
        
        Args:
            can_id: CAN node ID to set
            
        Returns:
            bool: True if successful
        """
        return self.send_sdo_write(0x6510, 0x0F, can_id, 1)
    
    def set_operation_mode(self, mode):
        """
        Set operation mode
        
        Args:
            mode: Operation mode 
                  3 = Speed mode with Acc/Dec
                  -3 = Speed mode without Acc/Dec
                  4 = Torque mode
                  1 = Position mode
            
        Returns:
            bool: True if successful
        """
        return self.send_sdo_write(0x6060, 0x00, mode, 1)
    
    def set_control_word(self, control_word):
        """
        Set control word
        
        Args:
            control_word: Control word value
                         0x06 = Disable motor
                         0x0F = Enable motor
                         0x86 = Clear alarm
                         
        Returns:
            bool: True if successful
        """
        return self.send_sdo_write(0x6040, 0x00, control_word, 2)
    
    def set_target_velocity_rpm(self, rpm):
        """
        Set target velocity in RPM
        
        Args:
            rpm: Target speed in RPM (can be int or float, will be converted to int)
                Range: -32768 to 32767
            
        Returns:
            bool: True if successful
        """
        # Convert to integer and clamp to valid range
        rpm = int(round(rpm))  # Convert float to int with rounding
        rpm = max(-32768, min(32767, rpm))
        self.target_rpm = rpm
        
        # Convert to unsigned 16-bit value for transmission
        if rpm < 0:
            rpm_value = (1 << 16) + rpm  # Two's complement for negative values
        else:
            rpm_value = rpm
            
        return self.send_sdo_write(0x2FF0, 0x09, rpm_value, 2)
    
    def set_target_velocity_dec(self, dec_value):
        """
        Set target velocity in DEC format
        
        Args:
            dec_value: Target speed in DEC format (32-bit signed)
            
        Returns:
            bool: True if successful
        """
        # Convert to unsigned 32-bit for negative values
        if dec_value < 0:
            dec_value = (1 << 32) + dec_value
        
        return self.send_sdo_write(0x60FF, 0x00, dec_value, 4)
    
    def rpm_to_dec(self, rpm, resolution=4096):
        """
        Convert RPM to DEC format
        
        Args:
            rpm: Speed in RPM
            resolution: Encoder resolution (default: 4096)
            
        Returns:
            int: DEC value
        """
        return int((rpm * 512 * resolution) / 1875)
    
    def dec_to_rpm(self, dec_value, resolution=4096):
        """
        Convert DEC format to RPM
        
        Args:
            dec_value: DEC value
            resolution: Encoder resolution (default: 4096)
            
        Returns:
            float: Speed in RPM
        """
        return (dec_value * 1875) / (512 * resolution)
    
    def set_acceleration(self, rps_per_s, resolution=4096):
        """
        Set acceleration in rps/s
        
        Args:
            rps_per_s: Acceleration in revolutions per second squared
            resolution: Encoder resolution (default: 4096)
            
        Returns:
            bool: True if successful
        """
        dec_value = int((rps_per_s * 256 * resolution) / 15625)
        return self.send_sdo_write(0x6083, 0x00, dec_value, 4)
    
    def set_deceleration(self, rps_per_s, resolution=4096):
        """
        Set deceleration in rps/s
        
        Args:
            rps_per_s: Deceleration in revolutions per second squared
            resolution: Encoder resolution (default: 4096)
            
        Returns:
            bool: True if successful
        """
        dec_value = int((rps_per_s * 256 * resolution) / 15625)
        return self.send_sdo_write(0x6084, 0x00, dec_value, 4)
    
    def set_pid_gains(self, kp=None, ki=None, kd=None):
        """
        Set PID controller gains and update motor controller parameters
        
        Args:
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
            
        Returns:
            bool: True if all parameters set successfully
        """
        success = True
        
        if kp is not None:
            self.kp = kp
            # Set Kvp0 (speed loop proportional gain) - object 0x60F901
            kvp_value = int(kp * 100)  # Scale appropriately
            kvp_value = max(0, min(65535, kvp_value))  # Clamp to 16-bit unsigned
            if not self.send_sdo_write(0x60F9, 0x01, kvp_value, 2):
                success = False
                print(f"Failed to set Kvp0 for motor {self.node_id}")
        
        if ki is not None:
            self.ki = ki
            # Set Kvi0 (speed loop integral gain) - object 0x60F902
            kvi_value = int(ki * 100)  # Scale appropriately
            kvi_value = max(0, min(65535, kvi_value))  # Clamp to 16-bit unsigned
            if not self.send_sdo_write(0x60F9, 0x02, kvi_value, 2):
                success = False
                print(f"Failed to set Kvi0 for motor {self.node_id}")
        
        if kd is not None:
            self.kd = kd
            # Note: The manual doesn't show a direct derivative gain parameter
            # We'll use software PID for derivative component
        
        if success:
            print(f"PID gains updated for motor {self.node_id}: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
        
        return success
    
    def get_status_word(self):
        """
        Get driver status word
        
        Returns:
            int: Status word value or None if failed
        """
        response = self.send_sdo_read(0x6041, 0x00)
        if response and len(response.data) >= 6:
            status = response.data[4] | (response.data[5] << 8)
            return status
        return None
    
    def get_actual_velocity_rpm(self):
        """
        Get actual velocity in RPM (corrected to use proper object)
        
        Returns:
            int: Actual velocity in RPM or None if failed
        """
        # Use object 0x60F918 for 1rpm resolution (16-bit signed)
        response = self.send_sdo_read(0x60F9, 0x18)
        if response and len(response.data) >= 6:
            # Extract 16-bit velocity value
            velocity = response.data[4] | (response.data[5] << 8)
            
            # Convert from unsigned to signed if necessary
            if velocity > 32767:  # If greater than max signed 16-bit
                velocity -= 65536
                
            return velocity
        return None
    
    def get_actual_velocity_high_res(self):
        """
        Get actual velocity in 0.001 RPM resolution
        
        Returns:
            float: Actual velocity in RPM (0.001 resolution) or None if failed
        """
        # Use object 0x60F919 for 0.001rpm resolution (32-bit signed)
        response = self.send_sdo_read(0x60F9, 0x19)
        if response and len(response.data) >= 8:
            # Extract 32-bit velocity value
            velocity = (response.data[4] | 
                       (response.data[5] << 8) | 
                       (response.data[6] << 16) | 
                       (response.data[7] << 24))
            
            # Convert from unsigned to signed if necessary
            if velocity > 2147483647:  # If greater than max signed 32-bit
                velocity -= 4294967296
            
            return velocity / 1000.0  # Convert to RPM
        return None
    
    def get_actual_velocity_dec(self):
        """
        Get actual velocity in DEC format (original method for reference)
        
        Returns:
            int: Actual velocity in DEC format or None if failed
        """
        response = self.send_sdo_read(0x606C, 0x00)
        if response and len(response.data) >= 8:
            # Extract 32-bit velocity value
            velocity = (response.data[4] | 
                       (response.data[5] << 8) | 
                       (response.data[6] << 16) | 
                       (response.data[7] << 24))
            
            # Convert from unsigned to signed if necessary
            if velocity > 2147483647:  # If greater than max signed 32-bit
                velocity -= 4294967296
                
            return velocity
        return None
    
    # Alias for backwards compatibility
    def get_actual_velocity(self):
        """Get actual velocity - uses RPM method by default"""
        return self.get_actual_velocity_rpm()
    
    def get_actual_position(self):
        """
        Get actual position
        
        Returns:
            int: Actual position in encoder counts or None if failed
        """
        response = self.send_sdo_read(0x6063, 0x00)
        if response and len(response.data) >= 8:
            # Extract 32-bit position value
            position = (response.data[4] | 
                       (response.data[5] << 8) | 
                       (response.data[6] << 16) | 
                       (response.data[7] << 24))
            
            # Convert from unsigned to signed if necessary
            if position > 2147483647:  # If greater than max signed 32-bit
                position -= 4294967296
                
            return position
        return None
    
    def get_actual_current(self):
        """
        Get actual current (Iq)
        
        Returns:
            float: Actual current in Arms or None if failed
        """
        response = self.send_sdo_read(0x6078, 0x00)
        if response and len(response.data) >= 6:
            # Extract 16-bit current value
            current_dec = response.data[4] | (response.data[5] << 8)
            
            # Convert from unsigned to signed if necessary
            if current_dec > 32767:
                current_dec -= 65536
            
            # Convert to Arms using formula from manual
            # [DEC] = [Arms] * 1.414 * 2048 / Peak_current
            # For L2DB4830: Peak_current = 30A (from manual)
            # For L2DB4875: Peak_current = 75A
            peak_current = 30  # Assume L2DB4830, should be configurable
            current_arms = (current_dec * peak_current) / (1.414 * 2048)
            
            return current_arms
        return None
    
    def get_dc_bus_voltage(self):
        """
        Get actual DC bus voltage
        
        Returns:
            int: DC bus voltage in volts or None if failed
        """
        response = self.send_sdo_read(0x60F7, 0x12)
        if response and len(response.data) >= 6:
            voltage = response.data[4] | (response.data[5] << 8)
            # Convert from unsigned to signed if necessary
            if voltage > 32767:
                voltage -= 65536
            return voltage
        return None
    
    def get_driver_temperature(self):
        """
        Get driver temperature
        
        Returns:
            int: Temperature in Celsius or None if failed
        """
        response = self.send_sdo_read(0x60F7, 0x0B)
        if response and len(response.data) >= 6:
            temp = response.data[4] | (response.data[5] << 8)
            # Convert from unsigned to signed if necessary
            if temp > 32767:
                temp -= 65536
            return temp
        return None
    
    def pid_control_step(self, target_rpm):
        """
        Perform one step of PID control
        
        Args:
            target_rpm: Desired RPM
            
        Returns:
            bool: True if control step successful
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            return False
        
        # Get current RPM
        current_rpm = self.get_actual_velocity_rpm()
        if current_rpm is None:
            return False
        
        # Calculate error
        error = target_rpm - current_rpm
        
        # PID calculations
        proportional = self.kp * error
        self.integral_error += error * dt
        integral = self.ki * self.integral_error
        derivative = self.kd * (error - self.last_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output as RPM adjustment
        adjusted_rpm = target_rpm + int(output)
        adjusted_rpm = max(-32768, min(32767, adjusted_rpm))  # Clamp to valid range
        
        # Log data for plotting
        if self.logging_enabled:
            with self.data_lock:
                self.time_data.append(current_time)
                self.target_rpm_data.append(target_rpm)
                self.actual_rpm_data.append(current_rpm)
                self.error_data.append(error)
                self.pid_output_data.append(output)
                
                # Get current for additional monitoring
                current_current = self.get_actual_current()
                self.current_data.append(current_current if current_current is not None else 0.0)
        
        # Send command
        success = self.set_target_velocity_rpm(adjusted_rpm)
        
        # Update for next iteration
        self.last_error = error
        self.last_time = current_time
        
        return success
    
    def enable_data_logging(self):
        """Enable data logging for plotting"""
        self.logging_enabled = True
        print(f"Data logging enabled for motor {self.node_id}")
    
    def disable_data_logging(self):
        """Disable data logging"""
        self.logging_enabled = False
        print(f"Data logging disabled for motor {self.node_id}")
    
    def clear_data_log(self):
        """Clear all logged data"""
        with self.data_lock:
            self.time_data.clear()
            self.target_rpm_data.clear()
            self.actual_rpm_data.clear()
            self.error_data.clear()
            self.pid_output_data.clear()
            self.current_data.clear()
        print(f"Data log cleared for motor {self.node_id}")
    
    def get_plot_data(self):
        """
        Get data for plotting
        
        Returns:
            dict: Dictionary containing all plot data
        """
        with self.data_lock:
            return {
                'time': list(self.time_data),
                'target_rpm': list(self.target_rpm_data),
                'actual_rpm': list(self.actual_rpm_data),
                'error': list(self.error_data),
                'pid_output': list(self.pid_output_data),
                'current': list(self.current_data)
            }
    
    def initialize(self):
        """
        Initialize motor driver for operation (updated sequence from manual)
        
        Returns:
            bool: True if initialization successful
        """
        print(f"Initializing motor driver {self.node_id}...")
        
        # Step 1: Set CAN ID (since SW3 is all OFF)
        if not self.set_can_id(self.node_id):
            print(f"Failed to set CAN ID for motor {self.node_id}")
            return False
        
        time.sleep(0.1)
        
        # Step 2: Set operation mode to speed mode with acceleration/deceleration
        if not self.set_operation_mode(3):
            print(f"Failed to set operation mode for motor {self.node_id}")
            return False
        
        time.sleep(0.1)
        
        # Step 3: Set default acceleration and deceleration (2 rps/s)
        if not self.set_acceleration(1.6):
            print(f"Failed to set acceleration for motor {self.node_id}")
            return False
        
        time.sleep(0.1)
        
        if not self.set_deceleration(0.9):
            print(f"Failed to set deceleration for motor {self.node_id}")
            return False
        
        time.sleep(0.1)
        
        # Step 4: Set default PID gains
        if not self.set_pid_gains(kp=0.5, ki=0.01):
            print(f"Failed to set PID gains for motor {self.node_id}")
            # Don't fail initialization for this
        
        time.sleep(0.1)
        
        # Step 5: Clear any alarms
        if not self.set_control_word(0x86):
            print(f"Failed to clear alarms for motor {self.node_id}")
            return False
        
        time.sleep(0.1)
        
        print(f"Motor driver {self.node_id} initialized successfully")
        return True
    
    def enable(self):
        """
        Enable motor for operation
        
        Returns:
            bool: True if successful
        """
        result = self.set_control_word(0x0F)
        if result:
            print(f"Motor {self.node_id} enabled")
            # Reset PID controller state
            self.last_error = 0
            self.integral_error = 0
            self.last_time = time.time()
        else:
            print(f"Failed to enable motor {self.node_id}")
        return result
    
    def disable(self):
        """
        Disable motor
        
        Returns:
            bool: True if successful
        """
        result = self.set_control_word(0x06)
        if result:
            print(f"Motor {self.node_id} disabled")
        else:
            print(f"Failed to disable motor {self.node_id}")
        return result
    
    def stop(self):
        """
        Stop motor by setting velocity to 0
        
        Returns:
            bool: True if successful
        """
        return self.set_target_velocity_rpm(0)
    
    def emergency_stop(self):
        """
        Emergency stop (using emergency stop function from manual)
        
        Returns:
            bool: True if successful
        """
        # Enable emergency stop - object 0x605A11
        return self.send_sdo_write(0x605A, 0x11, 1, 1)
    
    def release_emergency_stop(self):
        """
        Release emergency stop
        
        Returns:
            bool: True if successful
        """
        # Disable emergency stop - object 0x605A11
        return self.send_sdo_write(0x605A, 0x11, 0, 1)
    
    def is_ready(self):
        """
        Check if motor is ready (no errors)
        
        Returns:
            bool: True if motor is ready
        """
        status = self.get_status_word()
        if status is None:
            return False
        
        # Check for error bit (bit 3)
        return not (status & 0x0008)
    
    def has_error(self):
        """
        Check if motor has error
        
        Returns:
            bool: True if motor has error
        """
        status = self.get_status_word()
        if status is None:
            return True  # Assume error if can't read status
        
        return bool(status & 0x0008)
    
    def get_error_code(self):
        """
        Get detailed error code
        
        Returns:
            int: Error code or None if failed
        """
        response = self.send_sdo_read(0x2601, 0x00)
        if response and len(response.data) >= 6:
            error_code = response.data[4] | (response.data[5] << 8)
            return error_code
        return None
    
    def clear_errors(self):
        """
        Clear motor errors/alarms
        
        Returns:
            bool: True if successful
        """
        return self.set_control_word(0x86)
    
    def get_comprehensive_status(self):
        """
        Get comprehensive motor status for debugging
        
        Returns:
            dict: Status information
        """
        status = {
            'node_id': self.node_id,
            'status_word': self.get_status_word(),
            'actual_rpm': self.get_actual_velocity_rpm(),
            'actual_rpm_high_res': self.get_actual_velocity_high_res(),
            'actual_position': self.get_actual_position(),
            'actual_current': self.get_actual_current(),
            'dc_voltage': self.get_dc_bus_voltage(),
            'temperature': self.get_driver_temperature(),
            'error_code': self.get_error_code(),
            'has_error': self.has_error(),
            'is_ready': self.is_ready()
        }
        return status
    
    @classmethod
    def close_bus(cls):
        """Close the shared CAN bus connection"""
        if cls._bus_instance is not None:
            try:
                cls._bus_instance.shutdown()
                print("CAN bus closed")
            except Exception as e:
                print(f"Error closing CAN bus: {e}")
            finally:
                cls._bus_instance = None


# Example usage and testing
if __name__ == "__main__":
    print("L2DB Motor Driver Test with PID Control")
    
    try:
        motor1 = L2DBMotorDriver(interface='socketcan', channel='can1', node_id=1)
        motor2 = L2DBMotorDriver(interface='socketcan', channel='can1', node_id=2)

        print(f"Voltage: {motor1.get_dc_bus_voltage()} V")
        print(f"Current: {motor1.get_actual_current()} A")
    
    except KeyboardInterrupt:
        print("\nTest interrupted")


    # try:
    #     # Create motor drivers for both motors
    #     motor1 = L2DBMotorDriver(interface='socketcan', channel='can0', node_id=1)
    #     motor2 = L2DBMotorDriver(interface='socketcan', channel='can0', node_id=2)
        
    #     # Initialize both motors
    #     print("Initializing motors...")
    #     motor1_init = motor1.initialize()
    #     time.sleep(0.2)
    #     motor2_init = motor2.initialize()
        
    #     if motor1_init and motor2_init:
    #         print("Both motors initialized successfully")
            
    #         # Set custom PID gains
    #         motor1.set_pid_gains(kp=0.5, ki=0.01, kd=0)
    #         motor2.set_pid_gains(kp=0.5, ki=0.01, kd=0)
            
    #         # Enable motors
    #         motor1.enable()
    #         time.sleep(0.1)
    #         motor2.enable()
            
    #         # Test movement
    #         print("Testing motor movement...")
    #         motor1.set_target_velocity_rpm(5)
    #         time.sleep(3)
    #         motor2.set_target_velocity_rpm(-5)  # Opposite direction
            
    #         # Run for a few seconds with status monitoring
    #         for i in range(2):
    #             time.sleep(0.5)
                
    #             status1 = motor1.get_comprehensive_status()
    #             status2 = motor2.get_comprehensive_status()
                
    #             current1 = status1['actual_current'] if status1['actual_current'] is not None else 0.0
    #             current2 = status2['actual_current'] if status2['actual_current'] is not None else 0.0
                
    #             print(f"Motor 1: RPM={status1['actual_rpm']}, Current={current1:.2f}A")
    #             print(f"Motor 2: RPM={status2['actual_rpm']}, Current={current2:.2f}A")
                
    #             # Optional: Use PID control for precise speed control
    #             # motor1.pid_control_step(50)
    #             # motor2.pid_control_step(-50)
            
    #         # Stop motors
    #         print("Stopping motors...")
    #         motor1.stop()
    #         motor2.stop()
    #         time.sleep(1)
            
    #         # Disable motors
    #         motor1.disable()
    #         motor2.disable()
            
    #     else:
    #         print("Motor initialization failed")
                   
    # except KeyboardInterrupt:
    #     print("\nTest interrupted")
    # except Exception as e:
    #     print(f"Test error: {e}")
    #     import traceback
    #     traceback.print_exc()
    # finally:
    #     L2DBMotorDriver.close_bus()
    #     print("Test completed")