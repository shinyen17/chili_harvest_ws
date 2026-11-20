"""Robot bring-up for chili harvesting (replaces cocoabot.launch.py).

Starts MoveIt (+RViz) and, on the real robot, the servo bridge. Does NOT start
the cocoa YOLO node. The camera runs in its own terminal.

  ros2 launch chili_harvest chili_robot.launch.py                       # real arm
  ros2 launch chili_harvest chili_robot.launch.py use_hardware:=false   # RViz only
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import (AnyLaunchDescriptionSource,
                                               PythonLaunchDescriptionSource)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_hw = LaunchConfiguration('use_hardware')
    use_lift = LaunchConfiguration('use_lift')
    use_base = LaunchConfiguration('use_base')

    moveit = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('cocoabot_moveit_config'), 'launch', 'demo.launch.py')))

    servo_board = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_robot_controller'), 'launch',
            'ros_robot_controller.launch.xml')),
        condition=IfCondition(use_hw))

    arm_bridge = Node(package='ros_robot_controller', executable='robot_arm_controller',
                      name='robot_arm_controller', output='screen',
                      condition=IfCondition(use_hw))

    lift_joystick = Node(package='cocoabot_sub_modules', executable='extension',
                         name='extension', output='screen', condition=IfCondition(use_lift))

    actions = [
        DeclareLaunchArgument('use_hardware', default_value='true',
                              description='Start servo board + joint->servo bridge'),
        DeclareLaunchArgument('use_lift', default_value='true',
                              description='Joystick -> Arduino lift node'),
        DeclareLaunchArgument('use_base', default_value='false',
                              description='Start the tank base driver'),
        servo_board, arm_bridge, lift_joystick, moveit,
    ]

    try:  # tank base lives in another workspace; only include if it is sourced
        base_share = get_package_share_directory('turn_on_dlrobot_robot')
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(base_share, 'launch', 'tank.launch.py')),
            condition=IfCondition(use_base)))
    except Exception:  # noqa: BLE001
        pass
    return LaunchDescription(actions)
