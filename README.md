This project implements a fully autonomous drone tracking system that:

Detects objects in real-time using YOLOv8 neural network
Locks onto user-selected targets via mouse interaction
Follows targets autonomously using velocity-based control
Maintains optimal distance and centering through closed-loop feedback

Key Features
 Real-time object detection with YOLOv8
 Interactive target selection (click-to-lock)
 Autonomous 3-axis tracking (yaw, altitude, forward/back)
 Visual feedback with bounding boxes and tracking status
 Software-in-the-loop (SITL) simulation with Gazebo
 PX4 offboard control via MAVSDK and ROS2

 System Architecture

 ┌─────────────────────────────────────────────────────────────┐
│                     System Components                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │   Gazebo     │─────▶│  PX4 SITL    │                     │
│  │  Simulator   │      │  Autopilot   │                     │
│  └──────────────┘      └──────┬───────┘                     │
│         │                     │                             │
│         │ Camera Feed         │ MAVLink                     │
│         ▼                     ▼                             │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │ ROS2 Image   │      │ Micro-XRCE   │                     │
│  │   Bridge     │      │  DDS Agent   │                     │
│  └──────┬───────┘      └──────┬───────┘                     │
│         │                     │                             │
│         │                     │ ROS2 Messages               │
│         └─────────┬───────────┘                             │
│                   ▼                                         │
│         ┌─────────────────────┐                             │
│         │  Tracking Node      │                             │
│         │  (uav_camera_det)   │                             │
│         ├─────────────────────┤                             │
│         │ • YOLOv8 Detection  │                             │
│         │ • Target Locking    │                             │
│         │ • Control Algorithm │                             │
│         │ • Visual Overlay    │                             │
│         └─────────────────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Data Flow

Gazebo renders 3D world and drone camera feed
PX4 runs flight controller firmware (SITL mode)
ROS2 Bridge converts Gazebo camera images to ROS2 messages
Tracking Node processes images with YOLOv8, calculates control commands
Control Commands sent back to PX4 via ROS2/DDS for autonomous flight

🚀 Installation
1. Clone Repository
git clone https://github.com/monemati/PX4-ROS2-Gazebo-YOLOv8.git
cd PX4-ROS2-Gazebo-YOLOv8

2. Install PX4 Autopilot
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
cd PX4-Autopilot/
make px4_sitl

3. Install Micro-XRCE-DDS Agent
cd ~
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build && cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/

4. Build ROS2 Workspace
mkdir -p ~/ws_offboard_control/src/
cd ~/ws_offboard_control/src/
git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/PX4/px4_ros_com.git
cd ..
source /opt/ros/humble/setup.bash
colcon build

5. Install Python Dependencies
pip3 install opencv-python ultralytics numpy mavsdk aioconsole pygame
sudo apt install ros-humble-ros-gz-image ros-humble-ros-gzgarden

6. Configure Environment
Add to ~/.bashrc:
source /opt/ros/humble/setup.bash
export GZ_SIM_RESOURCE_PATH=~/.gz/models

7. Copy Models
cp -r PX4-ROS2-Gazebo-YOLOv8/models/* ~/.gz/models/
cp PX4-ROS2-Gazebo-YOLOv8/worlds/default.sdf ~/PX4-Autopilot/Tools/simulation/gz/worlds/

8. Adjust Camera Angle
Edit ~/PX4-Autopilot/Tools/simulation/gz/models/x500_depth/model.sdf:
xml<!-- Line 9: Change from -->
<pose>.12 .03 .242 0 0 0</pose>
<!-- To -->
<pose>.15 .029 .21 0 0.7854 0</pose>

🎮 Usage
Launch System (4 Terminals Required)

Terminal 1: DDS Agent
cd ~/Micro-XRCE-DDS-Agent
MicroXRCEAgent udp4 -p 8888

Terminal 2: PX4 + Gazebo
cd ~/PX4-Autopilot
PX4_SYS_AUTOSTART=4002 \
PX4_GZ_MODEL_POSE="268.08,-128.22,3.86,0.00,0,-0.7" \
PX4_GZ_MODEL=x500_depth \
./build/px4_sitl_default/bin/px4

Terminal 3: Image Bridge
source /opt/ros/humble/setup.bash
ros2 run ros_gz_image image_bridge \
/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image \
--ros-args -r /world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera

Terminal 4: Tracking Node
source /opt/ros/humble/setup.bash
source ~/ws_offboard_control/install/setup.bash
cd ~/PX4-ROS2-Gazebo-YOLOv8
python3 uav_camera_det.py

Wait for all terminals to initialize (~30 seconds)
Camera window opens showing drone view
Press T to takeoff
Wait for "✅ HOVERING - Ready to track!" message
Click on any detected object (person, car, etc.)
Press SPACE to enable tracking
Drone autonomously follows the target!

🙏 Acknowledgments
Original Repository by monemati for baseline integration:
https://github.com/monemati/PX4-ROS2-Gazebo-YOLOv8
