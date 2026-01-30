#!/usr/bin/env python3

"""
Autonomous Drone Target Tracker
- Press T to takeoff to 3 meters
- Detects objects with YOLOv8
- Click on detection to lock target
- Drone autonomously follows the target
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

# IMPORTS FOR TRACKING
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class DroneTracker(Node):
    def __init__(self):
        super().__init__('drone_tracker')
        
        # QoS profile for PX4 messages
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Camera subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.camera_callback,
            10)
        
        # Vehicle status subscriber
        self.status_subscription = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        
        # PX4 Publishers for autonomous control
        self.offboard_mode_publisher = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile)
        
        self.trajectory_publisher = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile)
        
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile)
        
        # Timer to publish control commands at 10Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # Detection and tracking variables
        self.br = CvBridge()
        self.model = YOLO('yolov8n.pt')  # Nano model (lighter)
        self.current_frame = None
        self.detections = []
        
        # Tracking state
        self.target_locked = False
        self.target_class = None
        self.target_position = None  # (x_center, y_center, box_width, box_height)
        self.tracking_enabled = False
        self.offboard_mode_active = False
        self.is_armed = False
        self.nav_state = 0
        
        # Takeoff state machine
        self.takeoff_requested = False
        self.takeoff_altitude = -3.0  # NED frame: negative = up
        self.takeoff_phase = 0  # 0=ground, 1=arming, 2=offboard, 3=climbing, 4=hover
        self.takeoff_timer = 0
        
        # Image dimensions
        self.img_width = 640
        self.img_height = 480
        
        # Control gains 
        self.yaw_gain = 0.003     
        self.altitude_gain = 0.002  
        self.forward_gain = 0.001   
        
        # Target size thresholds 
        self.target_size_min = 5000   
        self.target_size_max = 50000  
        self.target_size_ideal = 20000 
        
        # Mouse callback for target selection
        cv2.namedWindow('Drone Camera - Press T to Takeoff')
        cv2.setMouseCallback('Drone Camera - Press T to Takeoff', self.mouse_callback)
        
        self.get_logger().info('🚁 Drone Tracker Initialized!')
        self.get_logger().info('📹 Waiting for camera feed...')
        self.get_logger().info('')
        self.get_logger().info('=== CONTROLS ===')
        self.get_logger().info('Press T: Takeoff to 3 meters')
        self.get_logger().info('Click on detection: Lock target')
        self.get_logger().info('Press SPACE: Enable/disable tracking')
        self.get_logger().info('Press R: Reset target lock')
        self.get_logger().info('Press L: Land')
        self.get_logger().info('Press Q: Quit')

    def vehicle_status_callback(self, msg):
        """Monitor vehicle status"""
        self.is_armed = (msg.arming_state == 2)
        self.nav_state = msg.nav_state

    def camera_callback(self, data):
        """Receive camera images and run detection"""
        try:
            # Convert ROS image to OpenCV
            self.current_frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
            self.img_height, self.img_width = self.current_frame.shape[:2]
            
            # Run YOLO detection
            results = self.model(self.current_frame, verbose=False)
            self.detections = []
            
            # Draw detections
            annotated_frame = self.current_frame.copy()
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls]
                    
                    # Store detection
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    detection = {
                        'class': class_name,
                        'class_id': cls,
                        'confidence': float(conf),
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (x_center, y_center),
                        'size': width * height
                    }
                    self.detections.append(detection)
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green for normal detections
                    
                    # Check if this is our locked target
                    if self.target_locked and self.target_class == class_name:
                        # Find closest detection to last known position
                        if self.target_position:
                            dist = np.sqrt((x_center - self.target_position[0])**2 + 
                                         (y_center - self.target_position[1])**2)
                            if dist < 100:  # Within 100 pixels
                                color = (0, 0, 255)  # Red for locked target
                                self.target_position = (x_center, y_center, width, height)
                    
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(annotated_frame, label, (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw crosshair at image center
            cv2.line(annotated_frame, (self.img_width//2 - 20, self.img_height//2),
                    (self.img_width//2 + 20, self.img_height//2), (255, 255, 255), 1)
            cv2.line(annotated_frame, (self.img_width//2, self.img_height//2 - 20),
                    (self.img_width//2, self.img_height//2 + 20), (255, 255, 255), 1)
            
            # Draw status text
            status_text = []
            
            # Takeoff status
            if self.takeoff_phase == 0:
                status_text.append("⬇️ ON GROUND - Press T to takeoff")
            elif self.takeoff_phase == 1:
                status_text.append("⚡ Arming...")
            elif self.takeoff_phase == 2:
                status_text.append("🤖 Enabling offboard mode...")
            elif self.takeoff_phase == 3:
                status_text.append("🚀 Taking off...")
            elif self.takeoff_phase == 4:
                status_text.append("✅ HOVERING - Ready to track!")
            
            # Target lock status
            if self.target_locked:
                status_text.append(f"🎯 LOCKED: {self.target_class}")
            else:
                status_text.append("❌ No Target (Click to lock)")
            
            # Tracking status
            if self.tracking_enabled:
                status_text.append("▶️ TRACKING ACTIVE")
            else:
                status_text.append("⏸️ Tracking OFF (Press SPACE)")
            
            y_offset = 30
            for text in status_text:
                cv2.putText(annotated_frame, text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
            
            # Show target position error if locked
            if self.target_locked and self.target_position:
                x_err = self.target_position[0] - self.img_width//2
                y_err = self.target_position[1] - self.img_height//2
                size = self.target_position[2] * self.target_position[3]
                
                cv2.putText(annotated_frame, f"X Error: {x_err}px", (10, self.img_height - 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Y Error: {y_err}px", (10, self.img_height - 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Size: {size}px²", (10, self.img_height - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('Drone Camera - Press T to Takeoff', annotated_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quitting...')
                rclpy.shutdown()
            elif key == ord('t') or key == ord('T'):
                if self.takeoff_phase == 0:
                    self.get_logger().info('🚀 TAKEOFF INITIATED')
                    self.takeoff_requested = True
                    self.takeoff_phase = 1
            elif key == ord(' '):  # Spacebar
                if self.takeoff_phase == 4:  # Only allow tracking when hovering
                    self.tracking_enabled = not self.tracking_enabled
                    if self.tracking_enabled:
                        self.get_logger().info('▶️ Tracking ENABLED')
                    else:
                        self.get_logger().info('⏸️ Tracking PAUSED')
                else:
                    self.get_logger().warn('⚠️ Wait for takeoff to complete first!')
            elif key == ord('r'):  # Reset
                self.target_locked = False
                self.target_class = None
                self.target_position = None
                self.get_logger().info('🔄 Target lock RESET')
            elif key == ord('l'):  # Land
                self.get_logger().info('🛬 Landing...')
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.takeoff_phase = 0
                self.tracking_enabled = False
                
        except Exception as e:
            self.get_logger().error(f'Error in camera callback: {str(e)}')

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select target"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.takeoff_phase != 4:
                self.get_logger().warn('⚠️ Wait for takeoff first!')
                return
                
            # Find which detection was clicked
            for det in self.detections:
                x1, y1, x2, y2 = det['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.target_locked = True
                    self.target_class = det['class']
                    self.target_position = (det['center'][0], det['center'][1],
                                          x2 - x1, y2 - y1)
                    self.get_logger().info(f'🎯 Target LOCKED: {self.target_class}')
                    break

    def control_loop(self):
        """Main control loop - runs at 10Hz"""
        # Always publish offboard mode heartbeat
        self.publish_offboard_mode()
        
        # Handle takeoff sequence
        if self.takeoff_requested and self.takeoff_phase < 4:
            self.handle_takeoff()
            return
        
        # If tracking is enabled and target is locked, compute control
        if self.tracking_enabled and self.target_locked and self.target_position:
            self.track_target()
        elif self.takeoff_phase == 4:
            # Hovering - maintain altitude
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0, 0.0)

    def handle_takeoff(self):
        """State machine for takeoff sequence"""
        self.takeoff_timer += 1
        
        if self.takeoff_phase == 1:  # Arming
            if self.takeoff_timer < 10:  # Wait 1 second
                return
            self.get_logger().info('⚡ Sending ARM command...')
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.takeoff_phase = 2
            self.takeoff_timer = 0
            
        elif self.takeoff_phase == 2:  # Enable offboard
            if self.takeoff_timer < 10:  # Wait 1 second
                return
            self.get_logger().info('🤖 Enabling offboard mode...')
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.offboard_mode_active = True
            self.takeoff_phase = 3
            self.takeoff_timer = 0
            
        elif self.takeoff_phase == 3:  # Climbing
            # Send climb command
            self.publish_trajectory_setpoint(0.0, 0.0, -0.5, 0.0)  # Up at 0.5 m/s
            
            if self.takeoff_timer > 80:  # Climb for 8 seconds (should reach 3-4m)
                self.get_logger().info('✅ Takeoff complete! Ready to track.')
                self.takeoff_phase = 4
                self.takeoff_requested = False
                self.takeoff_timer = 0

    def track_target(self):
        """Calculate and send control commands to follow target"""
        x_center, y_center, width, height = self.target_position
        
        # Calculate errors
        x_error = x_center - (self.img_width / 2)   # Pixels left/right of center
        y_error = y_center - (self.img_height / 2)  # Pixels above/below center
        target_size = width * height
        
        # YAW control (rotate to center target horizontally)
        yawspeed = -x_error * self.yaw_gain
        yawspeed = np.clip(yawspeed, -0.5, 0.5)
        
        # ALTITUDE control (move up/down to center target vertically)
        z_velocity = y_error * self.altitude_gain
        z_velocity = np.clip(z_velocity, -0.3, 0.3)
        
        # FORWARD/BACK control (maintain ideal distance)
        if target_size < self.target_size_min:
            x_velocity = 0.3
        elif target_size > self.target_size_max:
            x_velocity = -0.3
        else:
            size_error = target_size - self.target_size_ideal
            x_velocity = size_error * self.forward_gain
            x_velocity = np.clip(x_velocity, -0.3, 0.3)
        
        # Send command
        self.publish_trajectory_setpoint(x_velocity, 0.0, z_velocity, yawspeed)
        
    def publish_offboard_mode(self):
        """Publish offboard control mode heartbeat"""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, vx, vy, vz, yawspeed):
        """Publish velocity command"""
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.velocity[0] = vx
        msg.velocity[1] = vy
        msg.velocity[2] = vz
        msg.yawspeed = yawspeed
        self.trajectory_publisher.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Publish a vehicle command"""
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    tracker = DroneTracker()
    
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
