"""Chili perception + harvest controller.

  ros2 launch chili_harvest chili_harvest.launch.py                  # dry run (safe)
  ros2 launch chili_harvest chili_harvest.launch.py dry_run:=false   # arm moves
  ros2 launch chili_harvest chili_harvest.launch.py control:=false   # perception only
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('chili_harvest'),
                                  'config', 'chili_params.yaml')
    params = LaunchConfiguration('params_file')
    dry_run = ParameterValue(LaunchConfiguration('dry_run'), value_type=bool)

    perception = Node(package='chili_harvest', executable='chili_perception',
                      name='chili_perception', output='screen', parameters=[params],
                      condition=IfCondition(LaunchConfiguration('perception')))
    control = Node(package='chili_harvest', executable='harvest_control',
                   name='chili_harvest_control', output='screen',
                   parameters=[params, {'dry_run': dry_run}],
                   condition=IfCondition(LaunchConfiguration('control')))

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('dry_run', default_value='true'),
        DeclareLaunchArgument('perception', default_value='true'),
        DeclareLaunchArgument('control', default_value='true'),
        perception, control,
    ])
