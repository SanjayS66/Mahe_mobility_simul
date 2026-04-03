import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    pkg_share = get_package_share_directory('mini_r1_v1_gz')
    
    #
    default_params_path = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    # 3. Use that full path as the default value
    params_file = LaunchConfiguration('params_file')

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_path,
        description='Full path to the ROS2 parameters file to use'
    )

    lifecycle_nodes = [
        'controller_server',
        'smoother_server',
        'behavior_server',
        # 'bt_navigator',
        'velocity_smoother',
    ]

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        parameters=[params_file]
    )

    smoother_server = Node(
        package='nav2_smoother',
        executable='smoother_server',
        output='screen',
        parameters=[params_file]
    )

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        output='screen',
        parameters=[params_file]
    )

    # bt_navigator = Node(
    #     package='nav2_bt_navigator',
    #     executable='bt_navigator',
    #     output='screen',
    #     parameters=[params_file]
    # )

    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        output='screen',
        parameters=[params_file]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': lifecycle_nodes
        }]
    )

    return LaunchDescription([
        declare_params,
        controller_server,
        smoother_server,
        behavior_server,
        # bt_navigator,
        velocity_smoother,
        lifecycle_manager,
    ])